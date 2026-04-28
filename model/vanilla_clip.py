from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prototype.projector import MLPProjector
from .prototype.token_mask import TokenMaskBuilder


class IdentityProjector(nn.Module):
    def __init__(self, normalize_output: bool = True):
        super().__init__()
        self.normalize_output = bool(normalize_output)

    def forward(self, inputs: torch.Tensor, return_debug: bool = False):
        projected_raw = inputs
        projected = F.normalize(projected_raw, dim=-1) if self.normalize_output else projected_raw
        if not return_debug:
            return projected
        return projected, {
            'projected_features': projected,
            'projected_features_norm': projected,
            'projected_features_pre_norm': projected_raw,
            'projected_features_raw': projected_raw,
            'projector_type': 'identity',
        }


class VanillaClipLosses(nn.Module):
    def __init__(
        self,
        temperature_init: float = 0.07,
        normalize_embeddings: bool = True,
        use_loss_ret: bool = True,
        lambda_ret: float = 1.0,
        retrieval_mode: str = 'clip_bidirectional',
    ):
        super().__init__()
        if temperature_init <= 0:
            raise ValueError('temperature_init must be positive.')
        self.normalize_embeddings = bool(normalize_embeddings)
        self.use_loss_ret = bool(use_loss_ret)
        self.lambda_ret = float(lambda_ret)
        self.retrieval_mode = str(retrieval_mode).lower()
        if self.retrieval_mode not in {'clip_bidirectional', 'surrogate_i2t'}:
            raise ValueError(
                f"Unsupported retrieval_mode={retrieval_mode!r}. Allowed values: ['surrogate_i2t', 'clip_bidirectional']."
            )
        initial_logit_scale = torch.log(torch.tensor(1.0 / temperature_init, dtype=torch.float32))
        self.register_buffer('logit_scale', initial_logit_scale.clone())

    def get_logit_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def get_retrieval_temperature(self) -> torch.Tensor:
        return torch.reciprocal(self.get_logit_scale())

    def _prepare_embeddings(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor):
        if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
            raise ValueError('Similarity inputs must have shape [B, D].')
        if image_embeddings.size(-1) != text_embeddings.size(-1):
            raise ValueError('Image and text embeddings must have the same feature dimension.')
        if self.normalize_embeddings:
            image_embeddings = F.normalize(image_embeddings, dim=-1)
            text_embeddings = F.normalize(text_embeddings, dim=-1)
        return image_embeddings, text_embeddings

    def compute_logits_i2t(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        image_embeddings, text_embeddings = self._prepare_embeddings(image_embeddings, text_embeddings)
        logit_scale = self.get_logit_scale().to(device=image_embeddings.device, dtype=image_embeddings.dtype)
        return (image_embeddings @ text_embeddings.t()) * logit_scale

    def compute_similarity_t2i(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        return self.compute_logits_i2t(image_embeddings, text_embeddings).t().contiguous()

    def retrieval_loss(self, logits_i2t: torch.Tensor) -> Dict[str, torch.Tensor]:
        if logits_i2t.ndim != 2 or logits_i2t.size(0) != logits_i2t.size(1):
            raise ValueError(
                'logits_i2t must have shape [B, B] for paired CLIP-style retrieval; '
                f'received {tuple(logits_i2t.shape)}.'
            )
        targets = torch.arange(logits_i2t.size(0), device=logits_i2t.device)
        loss_i2t = F.cross_entropy(logits_i2t, targets)
        if self.retrieval_mode == 'clip_bidirectional':
            loss_t2i = F.cross_entropy(logits_i2t.t(), targets)
            loss = 0.5 * (loss_i2t + loss_t2i)
        else:
            loss_t2i = logits_i2t.new_zeros(())
            loss = loss_i2t
        return {
            'loss': loss,
            'loss_i2t': loss_i2t,
            'loss_t2i': loss_t2i,
            'logits_i2t': logits_i2t,
        }

    def _pairwise_debug_metrics(self, logits_i2t: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        if logits_i2t is None:
            return {}
        batch_size = logits_i2t.size(0)
        if batch_size == 0:
            return {}
        scale = self.get_logit_scale().to(device=logits_i2t.device, dtype=logits_i2t.dtype)
        cosine = logits_i2t / scale.clamp_min(1e-12)
        positive = cosine.diagonal()
        positive_logits = logits_i2t.diagonal()
        if batch_size <= 1:
            hardest_negative = torch.zeros_like(positive)
            hardest_negative_logits = torch.zeros_like(positive_logits)
        else:
            negative_mask = ~torch.eye(batch_size, device=logits_i2t.device, dtype=torch.bool)
            hardest_negative = cosine.masked_fill(~negative_mask, float('-inf')).max(dim=1).values
            hardest_negative_logits = logits_i2t.masked_fill(~negative_mask, float('-inf')).max(dim=1).values
        return {
            'clip_pairwise_positive_cosine_mean': positive.mean().detach(),
            'clip_pairwise_hardest_negative_cosine_mean': hardest_negative.mean().detach(),
            'clip_pairwise_margin_mean': (positive - hardest_negative).mean().detach(),
            'clip_pairwise_positive_logit_mean': positive_logits.mean().detach(),
            'clip_pairwise_hardest_negative_logit_mean': hardest_negative_logits.mean().detach(),
        }

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        logits_i2t: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
            raise ValueError('Vanilla CLIP losses expect image/text embeddings with shape [B, D].')
        if image_embeddings.shape != text_embeddings.shape:
            raise ValueError(
                'Vanilla CLIP training expects paired image/text embeddings with shared shape [B, D]; '
                f'got image={tuple(image_embeddings.shape)} text={tuple(text_embeddings.shape)}.'
            )
        if self.use_loss_ret:
            if logits_i2t is None:
                raise ValueError('logits_i2t must be provided when use_loss_ret is enabled.')
            ret_info = self.retrieval_loss(logits_i2t)
            loss_ret = ret_info['loss']
            loss_ret_i2t = ret_info['loss_i2t']
            loss_ret_t2i = ret_info['loss_t2i']
        else:
            zero = image_embeddings.new_zeros(())
            ret_info = None
            loss_ret = zero
            loss_ret_i2t = zero
            loss_ret_t2i = zero
        loss_total = self.lambda_ret * loss_ret
        zero = image_embeddings.new_zeros(())
        outputs = {
            'loss_total': loss_total,
            'host_loss': loss_total,
            'host_loss_i2t': loss_ret_i2t,
            'host_loss_t2i': loss_ret_t2i,
            'loss_proxy': zero,
            'loss_proxy_image': zero,
            'loss_proxy_text': zero,
            'loss_proxy_text_exact': zero,
            'loss_ret': loss_ret,
            'loss_ret_i2t': loss_ret_i2t,
            'loss_ret_t2i': loss_ret_t2i,
            'loss_align': zero,
            'loss_diag': zero,
            'loss_support': zero,
            'loss_diversity': zero,
            'loss_balance': zero,
            'loss_proxy_image_weighted': zero,
            'loss_proxy_text_weighted': zero,
            'loss_proxy_text_exact_weighted': zero,
            'loss_proxy_weighted': zero,
            'loss_ret_weighted': self.lambda_ret * loss_ret,
            'loss_align_weighted': zero,
            'loss_diag_weighted': zero,
            'loss_support_weighted': zero,
            'loss_diversity_weighted': zero,
            'loss_balance_weighted': zero,
            'lambda_proxy': zero,
            'lambda_proxy_image': zero,
            'lambda_proxy_text': zero,
            'lambda_proxy_text_exact': zero,
            'use_loss_proxy_image': zero,
            'use_loss_proxy_text': zero,
            'use_loss_proxy_text_exact': zero,
            'use_loss_ret': image_embeddings.new_tensor(float(self.use_loss_ret)),
            'lambda_ret': image_embeddings.new_tensor(self.lambda_ret),
            'use_loss_align': zero,
            'lambda_align': zero,
            'use_loss_diag': zero,
            'lambda_diag': zero,
            'use_loss_support': zero,
            'lambda_support': zero,
            'lambda_div': zero,
            'lambda_bal': zero,
            'proxy_temperature': zero,
            'retrieval_temperature': self.get_retrieval_temperature().to(device=image_embeddings.device, dtype=image_embeddings.dtype),
            'logit_scale': self.get_logit_scale().to(device=image_embeddings.device, dtype=image_embeddings.dtype),
            'debug_metrics': {
                **self._pairwise_debug_metrics(ret_info['logits_i2t'] if ret_info is not None else None),
            },
        }
        if return_debug and ret_info is not None:
            outputs['retrieval_logits_i2t'] = ret_info['logits_i2t']
        return outputs


class VanillaCLIPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        projector_output_dim: int,
        projector_hidden_dim: Optional[int] = None,
        projector_dropout: float = 0.0,
        projector_type: str = 'mlp2',
        normalize_projector_outputs: bool = True,
        use_custom_projector: bool = False,
        special_token_ids: Optional[Dict[str, object]] = None,
        error_on_empty_kept_tokens: bool = True,
        contrastive_temperature_init: float = 0.07,
        use_loss_ret: bool = True,
        lambda_ret: float = 1.0,
        retrieval_mode: str = 'clip_bidirectional',
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.projector_output_dim = int(projector_output_dim)
        self.projector_hidden_dim = int(projector_hidden_dim or projector_output_dim)
        self.normalize_projector_outputs = bool(normalize_projector_outputs)
        self.use_custom_projector = bool(use_custom_projector)
        self.token_mask_builder = TokenMaskBuilder(
            token_policy='eos_only',
            special_token_ids=special_token_ids,
            error_on_empty_kept_tokens=error_on_empty_kept_tokens,
        )
        projector_cls = MLPProjector if self.use_custom_projector else IdentityProjector
        projector_kwargs = {
            'normalize_output': self.normalize_projector_outputs,
        }
        if self.use_custom_projector:
            projector_kwargs = {
                'input_dim': self.input_dim,
                'hidden_dim': self.projector_hidden_dim,
                'output_dim': self.projector_output_dim,
                'dropout': projector_dropout,
                'normalize_output': self.normalize_projector_outputs,
                'projector_type': projector_type,
            }
        self.image_projector = projector_cls(**projector_kwargs)
        self.text_projector = projector_cls(**projector_kwargs)
        self.losses = VanillaClipLosses(
            temperature_init=contrastive_temperature_init,
            normalize_embeddings=self.normalize_projector_outputs,
            use_loss_ret=use_loss_ret,
            lambda_ret=lambda_ret,
            retrieval_mode=retrieval_mode,
        )

    @property
    def output_dim(self) -> int:
        return self.projector_output_dim if self.use_custom_projector else self.input_dim

    def _empty_routing(self, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_empty((reference.size(0), 0))

    def _mode_metrics(self, reference: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            'vanilla_clip_mode': reference.new_tensor(1.0),
            'vanilla_clip_custom_projector': reference.new_tensor(float(self.use_custom_projector)),
        }

    def encode_image_branch(self, image_embeddings: torch.Tensor, return_debug: bool = False) -> Dict[str, object]:
        image_projected, image_projector_debug = self.image_projector(image_embeddings, return_debug=True)
        outputs = {
            'image_embedding': image_embeddings,
            'image_proxy_features': image_embeddings,
            'routing_weights': self._empty_routing(image_embeddings),
            'summary': image_embeddings,
            'image_projected': image_projected,
            'image_projected_raw': image_projector_debug['projected_features_raw'],
            'image_projector_debug': image_projector_debug,
            'router_debug': {},
            'aggregator_debug': {},
        }
        if return_debug:
            outputs['debug'] = {
                **image_projector_debug,
                **self._mode_metrics(image_embeddings),
            }
        return outputs

    def encode_text_branch(self, text_embeddings: torch.Tensor, return_debug: bool = False) -> Dict[str, object]:
        text_projected, text_projector_debug = self.text_projector(text_embeddings, return_debug=True)
        outputs = {
            'text_embedding': text_embeddings,
            'text_projected': text_projected,
            'text_projected_raw': text_projector_debug['projected_features_raw'],
            'text_projector_debug': text_projector_debug,
        }
        if return_debug:
            outputs['debug'] = {
                **text_projector_debug,
                **self._mode_metrics(text_embeddings),
            }
        return outputs

    def compute_similarity_matrix(self, image_projected: torch.Tensor, text_projected: torch.Tensor) -> torch.Tensor:
        return self.losses.compute_similarity_t2i(image_projected, text_projected)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        return_debug: bool = False,
    ) -> Dict[str, object]:
        image_outputs = self.encode_image_branch(image_embeddings, return_debug=return_debug)
        text_outputs = self.encode_text_branch(text_embeddings, return_debug=return_debug)
        logits_i2t = self.losses.compute_logits_i2t(image_outputs['image_projected'], text_outputs['text_projected']) if self.losses.use_loss_ret else None
        loss_outputs = self.losses(
            image_outputs['image_projected'],
            text_outputs['text_projected'],
            logits_i2t=logits_i2t,
            return_debug=return_debug,
        )
        outputs = {
            'routing_weights': image_outputs['routing_weights'],
            'summary': image_outputs['summary'],
            'image_projected': image_outputs['image_projected'],
            'image_projected_raw': image_outputs['image_projected_raw'],
            'surrogate_text_projected': text_outputs['text_projected'],
            'surrogate_text_projected_raw': text_outputs['text_projected_raw'],
            'exact_text_projected': text_outputs['text_projected'],
            'exact_text_projected_raw': text_outputs['text_projected_raw'],
            'surrogate_pairwise_logits': logits_i2t,
            'losses': loss_outputs,
            'metrics': {
                **self._mode_metrics(image_embeddings),
                **loss_outputs.get('debug_metrics', {}),
            },
        }
        outputs['debug'] = dict(outputs['metrics'])
        if return_debug:
            outputs['debug'].update({
                **image_outputs.get('debug', {}),
                **text_outputs.get('debug', {}),
            })
            if logits_i2t is not None:
                outputs['debug']['retrieval_logits_i2t'] = logits_i2t.detach()
        return outputs
