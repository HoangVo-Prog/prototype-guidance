from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vanilla_clip import VanillaCLIPHead


def _module_compute_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    """Resolve the module parameter dtype used for explicit input casting."""
    weight = getattr(module, 'weight', None)
    if isinstance(weight, torch.Tensor):
        return weight.dtype
    for parameter in module.parameters():
        return parameter.dtype
    return fallback


def _adapter_aligned_input_dtype(tensor: torch.Tensor, module: nn.Module) -> torch.dtype:
    """
    Mirror original ITSELF adapter behavior:
    - prefer fp16 compute under CUDA autocast;
    - otherwise match module parameter dtype.
    """
    if tensor.is_cuda and torch.is_autocast_enabled():
        return torch.float16
    return _module_compute_dtype(module, fallback=tensor.dtype)


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.pow(x, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    return torch.div(x, norm)


def maxk(x: torch.Tensor, dim: int, k: int) -> torch.Tensor:
    index = x.topk(k, dim=dim)[1]
    return x.gather(dim, index)


def maxk_pool1d_var(x: torch.Tensor, dim: int, k: int, lengths: torch.Tensor) -> torch.Tensor:
    results = []
    lengths_list = [int(value) for value in lengths.detach().cpu().tolist()]
    for idx, length in enumerate(lengths_list):
        effective_k = min(k, max(length, 1))
        max_k_i = maxk(x[idx, :max(length, 1), :], dim - 1, effective_k).mean(dim - 1)
        results.append(max_k_i)
    return torch.stack(results, dim=0)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.output_dim = int(output_dim)
        self.num_layers = int(num_layers)
        hidden_dims = [hidden_dim] * (self.num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + hidden_dims, hidden_dims + [output_dim]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in hidden_dims + [output_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, width = x.size()
        x = x.reshape(batch_size * num_tokens, width)
        for layer_index, (bn, layer) in enumerate(zip(self.bns, self.layers)):
            x = layer(x)
            if layer_index < self.num_layers - 1:
                x = F.relu(bn(x))
        return x.view(batch_size, num_tokens, self.output_dim)


class TextualEmbeddingLayer(nn.Module):
    def __init__(self, input_dim: int = 512, embed_dim: int = 4096, ratio: float = 0.4):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.linear = nn.Linear(input_dim, embed_dim)
        self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)
        self.ratio = float(ratio)

    def forward(
        self,
        features: torch.Tensor,
        text: torch.Tensor,
        attention: torch.Tensor,
        current_step: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> torch.Tensor:
        mask = (text != 0).to(dtype=features.dtype)
        lengths = mask.sum(dim=1) - 2
        sequence_length = attention.size(1)
        if current_step is not None:
            ratio_start = 0.65
            ratio_end = 0.5
            effective_total_steps = total_steps if total_steps is not None else (10 * 145)
            current_step = min(max(int(current_step), 1), effective_total_steps)
            ratio = current_step / float(effective_total_steps)
            k = int((sequence_length - 2) * ratio_start * ((ratio_end / ratio_start) ** ratio))
        else:
            k = int((sequence_length - 2) * self.ratio)
        k = max(k, 1)

        batch_size = features.size(0)
        eos_positions = text.argmax(dim=-1)
        attention = attention.clone()
        attention[torch.arange(batch_size, device=attention.device), :, eos_positions] = -1
        attention[torch.arange(batch_size, device=attention.device), :, 0] = -1
        attention = attention[torch.arange(batch_size, device=attention.device), eos_positions, :]
        attention = attention * mask

        effective_k = min(k, attention.size(-1))
        topk_indices = attention.topk(dim=-1, k=effective_k)[1].unsqueeze(-1).expand(batch_size, effective_k, features.size(2))
        selected = torch.gather(input=features, dim=1, index=topk_indices)
        selected = l2norm(selected, dim=-1)

        lengths = torch.tensor(
            [max(min(int(lengths[i].item()), selected.size(1)), 1) for i in range(batch_size)],
            device=selected.device,
            dtype=selected.dtype,
        )
        # Keep adapter-like fp16 input behavior when autocast is active.
        linear_dtype = _adapter_aligned_input_dtype(selected, self.linear)
        mlp_dtype = _adapter_aligned_input_dtype(selected, self.mlp)
        base = self.linear(selected.to(dtype=linear_dtype))
        refined = self.mlp(selected.to(dtype=mlp_dtype))
        if base.dtype != refined.dtype:
            base = base.to(dtype=refined.dtype)
        refined = refined + base
        return maxk_pool1d_var(refined, 1, 1, lengths).float()


class VisualEmbeddingLayer(nn.Module):
    def __init__(self, input_dim: int = 512, embed_dim: int = 4096, ratio: float = 0.4):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.fc = nn.Linear(input_dim, embed_dim)
        self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)
        self.ratio = float(ratio)

    def forward(
        self,
        features: torch.Tensor,
        attention: torch.Tensor,
        current_step: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> torch.Tensor:
        sequence_length = attention.size(1)
        if current_step is not None:
            ratio_start = 0.65
            ratio_end = 0.5
            effective_total_steps = total_steps if total_steps is not None else (10 * 145)
            current_step = min(max(int(current_step), 1), effective_total_steps)
            ratio = current_step / float(effective_total_steps)
            k = int((sequence_length - 1) * ratio_start * ((ratio_end / ratio_start) ** ratio))
        else:
            k = int((sequence_length - 1) * self.ratio)
        k = max(k, 1)

        batch_size = features.size(0)
        attention = attention.clone()
        attention[torch.arange(batch_size, device=attention.device), :, 0] = -1
        effective_k = min(k, attention.size(-1))
        indices = attention[:, 0].topk(dim=-1, k=effective_k)[1]
        indices = indices.unsqueeze(-1).expand(batch_size, effective_k, features.size(2))
        selected = torch.gather(input=features, dim=1, index=indices)
        selected = l2norm(selected, dim=-1)
        feat_lengths = torch.full((batch_size,), float(selected.size(1)), device=selected.device, dtype=selected.dtype)
        # Keep adapter-like fp16 input behavior when autocast is active.
        fc_dtype = _adapter_aligned_input_dtype(selected, self.fc)
        mlp_dtype = _adapter_aligned_input_dtype(selected, self.mlp)
        base = self.fc(selected.to(dtype=fc_dtype))
        refined = self.mlp(selected.to(dtype=mlp_dtype))
        if base.dtype != refined.dtype:
            base = base.to(dtype=refined.dtype)
        refined = refined + base
        return maxk_pool1d_var(refined, 1, 1, feat_lengths).float()


def compute_tal_components(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    pid: torch.Tensor,
    tau: float = 0.015,
    margin: float = 0.1,
):
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scores = text_norm @ image_norm.t()
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1))
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().to(scores.device)
    mask = 1.0 - labels
    exp_i2t = (scores / tau).exp()
    exp_t2i = (scores.t() / tau).exp()
    alpha_i2t = (exp_i2t * labels / (exp_i2t * labels).sum(dim=1, keepdim=True).clamp_min(1e-12)).detach()
    alpha_t2i = (exp_t2i * labels / (exp_t2i * labels).sum(dim=1, keepdim=True).clamp_min(1e-12)).detach()
    loss_i2t = (
        - (alpha_i2t * scores).sum(1)
        + tau * (exp_i2t * mask).sum(1).clamp(max=1e36).log()
        + margin
    ).clamp(min=0).sum()
    loss_t2i = (
        - (alpha_t2i * scores.t()).sum(1)
        + tau * (exp_t2i * mask).sum(1).clamp(max=1e36).log()
        + margin
    ).clamp(min=0).sum()
    return loss_i2t + loss_t2i, loss_i2t, loss_t2i

def cid_similarity_matrix(image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    image_norm = F.normalize(image_features, p=2, dim=1)
    text_norm = F.normalize(text_features, p=2, dim=1)
    return image_norm @ text_norm.t()   # image x text, match ITSELF


def cosine_similarity_matrix(image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    image_norm = F.normalize(image_features, p=2, dim=1)
    text_norm = F.normalize(text_features, p=2, dim=1)
    return text_norm @ image_norm.t()


def sample_hard_negatives(similarity: torch.Tensor, labels: torch.Tensor) -> Dict[str, list]:
    similarity_cpu = similarity.detach().float().cpu()
    labels_cpu = labels.detach().view(-1).cpu()
    num_samples = similarity_cpu.size(0)
    hard_negatives = {'visual_negatives': [], 'text_negatives': []}
    for i in range(num_samples):
        sorted_text_idx = torch.argsort(similarity_cpu[i], descending=True)
        text_negative = None
        for j in sorted_text_idx.tolist():
            if int(labels_cpu[i].item()) != int(labels_cpu[j].item()):
                text_negative = int(j)
                break
        if text_negative is None:
            text_negative = int(i)
        hard_negatives['text_negatives'].append(text_negative)

        sorted_visual_idx = torch.argsort(similarity_cpu[:, i], descending=True)
        visual_negative = None
        for j in sorted_visual_idx.tolist():
            if int(labels_cpu[i].item()) != int(labels_cpu[j].item()):
                visual_negative = int(j)
                break
        if visual_negative is None:
            visual_negative = int(i)
        hard_negatives['visual_negatives'].append(visual_negative)
    return hard_negatives


def update_labels_for_negatives(labels: torch.Tensor, hard_negatives: Dict[str, list], max_label: int) -> torch.Tensor:
    new_labels = labels.clone()
    num_samples = len(labels)
    for i in range(num_samples):
        new_labels[hard_negatives['text_negatives'][i]] = max_label + 1
        new_labels[hard_negatives['visual_negatives'][i]] = max_label + 1
    return new_labels


def create_sample_pairs(image_features: torch.Tensor, text_features: torch.Tensor, hard_negatives: Dict[str, list], new_labels: torch.Tensor, labels: torch.Tensor):
    num_samples = image_features.size(0)
    visual_feats = []
    textual_feats = []
    all_labels = []
    for i in range(num_samples):
        visual_feats.append(image_features[i])
        textual_feats.append(text_features[i])
        all_labels.append(labels[i])
        neg_idx = hard_negatives['visual_negatives'][i]
        visual_feats.append(image_features[neg_idx])
        textual_feats.append(text_features[i])
        all_labels.append(new_labels[neg_idx])
        neg_idx = hard_negatives['text_negatives'][i]
        visual_feats.append(image_features[i])
        textual_feats.append(text_features[neg_idx])
        all_labels.append(new_labels[neg_idx])
    return torch.stack(visual_feats), torch.stack(textual_feats), torch.stack(all_labels)


def compute_cid(logits_1: torch.Tensor, logits_2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    criterion = nn.CrossEntropyLoss(reduction='mean')
    return 0.5 * (criterion(logits_1, labels) + criterion(logits_2, labels))


def compute_id(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    criterion = nn.CrossEntropyLoss(reduction='mean')
    return criterion(logits, labels)

class CLIPHostAdapter(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.core = VanillaCLIPHead(**kwargs)
        self.output_dim = self.core.output_dim

    def freeze_trainable_head(self):
        for module in (self.core.image_projector, self.core.text_projector):
            for parameter in module.parameters():
                parameter.requires_grad = False

    def encode_image_branch(self, image_output, return_debug: bool = False, current_step: Optional[int] = None, total_steps: Optional[int] = None) -> Dict[str, object]:
        del current_step, total_steps
        outputs = self.core.encode_image_branch(image_output.projected_pooled, return_debug=return_debug)
        outputs['global_image_embedding'] = image_output.projected_pooled.float()
        outputs['grab_image_embedding'] = None
        outputs['host_similarity_logits'] = None
        return outputs

    def encode_text_branch(self, text_output, token_ids: torch.Tensor, return_debug: bool = False, current_step: Optional[int] = None, total_steps: Optional[int] = None) -> Dict[str, object]:
        del token_ids, current_step, total_steps
        outputs = self.core.encode_text_branch(text_output.projected_pooled, return_debug=return_debug)
        outputs['global_text_embedding'] = text_output.projected_pooled.float()
        outputs['grab_text_embedding'] = None
        return outputs

    def compute_similarity_matrix(self, image_features: Dict[str, torch.Tensor], text_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.core.compute_similarity_matrix(image_features.get('image_projected'), text_features.get('text_projected'))

    def forward(self, image_output, text_output, token_ids: torch.Tensor, pids: Optional[torch.Tensor] = None, return_debug: bool = False, current_step: Optional[int] = None, total_steps: Optional[int] = None):
        del pids, token_ids, current_step, total_steps
        outputs = self.core(
            image_embeddings=image_output.projected_pooled,
            text_embeddings=text_output.projected_pooled,
            return_debug=return_debug,
        )
        outputs['global_image_embedding'] = image_output.projected_pooled.float()
        outputs['global_text_embedding'] = text_output.projected_pooled.float()
        outputs['grab_image_embedding'] = None
        outputs['grab_text_embedding'] = None
        outputs['host_similarity_logits'] = outputs.get('surrogate_pairwise_logits')
        return outputs


class ITSELFHostHead(nn.Module):
    def __init__(self, args, input_dim: int, num_classes: int):
        super().__init__()
        self.args = args
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.grab_embed_dim = int(getattr(args, 'itself_grab_embed_dim', 4096))
        self.select_ratio = float(getattr(args, 'itself_select_ratio', 0.4))
        self.only_global = bool(getattr(args, 'itself_only_global', False))
        self.topk_type = str(getattr(args, 'itself_topk_type', 'mean')).lower()
        self.layer_index = int(getattr(args, 'itself_layer_index', -1))
        self.modify_k = bool(getattr(args, 'itself_modify_k', False))
        self.score_weight_global = float(getattr(args, 'itself_score_weight_global', 0.68))
        self.tau_itself = float(getattr(args, 'itself_tau', 0.015))
        self.margin = float(getattr(args, 'itself_margin', 0.1))
        loss_names = str(getattr(args, 'itself_loss_names', 'tal+cid')).lower()
        self.loss_names = {name.strip() for name in loss_names.split('+') if name.strip()}
        self.use_host_loss = bool(getattr(args, 'use_host_loss', True))

        if not self.only_global:
            self.visual_embedding_layer = VisualEmbeddingLayer(input_dim=self.input_dim, embed_dim=self.grab_embed_dim, ratio=self.select_ratio)
            self.textual_embedding_layer = TextualEmbeddingLayer(input_dim=self.input_dim, embed_dim=self.grab_embed_dim, ratio=self.select_ratio)
        else:
            self.visual_embedding_layer = None
            self.textual_embedding_layer = None
            
        if 'cid' in self.loss_names:
            effective_classes = self.num_classes + 1
            self.classifier_global = nn.Linear(self.input_dim, effective_classes)
            self.mlp_global = nn.Sequential(
                nn.Linear(2 * self.input_dim, self.input_dim),
                nn.LayerNorm(self.input_dim),
                nn.GELU(),
            )
            self.classifier_id_global = nn.Linear(self.input_dim, effective_classes)
            nn.init.normal_(self.classifier_global.weight.data, std=0.001)
            nn.init.constant_(self.classifier_global.bias.data, val=0.0)
            nn.init.normal_(self.classifier_id_global.weight.data, std=0.001)
            nn.init.constant_(self.classifier_id_global.bias.data, val=0.0)
            if not self.only_global:
                self.classifier_grab = nn.Linear(self.grab_embed_dim, effective_classes)
                self.mlp_grab = nn.Sequential(
                    nn.Linear(2 * self.grab_embed_dim, self.grab_embed_dim),
                    nn.LayerNorm(self.grab_embed_dim),
                    nn.GELU(),
                )
                self.classifier_id_grab = nn.Linear(self.grab_embed_dim, effective_classes)
                nn.init.normal_(self.classifier_grab.weight.data, std=0.001)
                nn.init.constant_(self.classifier_grab.bias.data, val=0.0)
                nn.init.normal_(self.classifier_id_grab.weight.data, std=0.001)
                nn.init.constant_(self.classifier_id_grab.bias.data, val=0.0)
        else:
            self.classifier_global = None
            self.mlp_global = None
            self.classifier_id_global = None
            self.classifier_grab = None
            self.mlp_grab = None
            self.classifier_id_grab = None

        self.output_dim = self.input_dim

    def freeze_trainable_head(self):
        for parameter in self.parameters():
            parameter.requires_grad = False

    def _normalize_attention(self, attention: Optional[torch.Tensor], batch_size: int, num_tokens: int, device, dtype) -> torch.Tensor:
        if attention is None:
            return torch.ones(batch_size, num_tokens, num_tokens, device=device, dtype=dtype)
        if attention.ndim == 4:
            attention = attention.mean(dim=1)
        return attention.to(device=device, dtype=dtype)

    def _rollout(self, attentions: torch.Tensor) -> torch.Tensor:
        if attentions.ndim == 5:
            num_layers, batch_size, _, num_tokens, _ = attentions.shape
        else:
            num_layers, batch_size, num_tokens, _ = attentions.shape
        result = torch.eye(num_tokens, device=attentions.device, dtype=attentions.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        discard_ratios = [0.25, 1.0, 1.0, 1.0, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 0.25, 0.25]
        for layer in range(min(4, num_layers), num_layers):
            if layer in {5, 6, 7, 8, 9, 10}:
                continue
            attn = attentions[layer]
            if attn.ndim == 4:
                attn = attn.mean(dim=1)
            flat = attn.view(batch_size, -1)
            num_to_discard = int(flat.size(-1) * discard_ratios[min(layer, len(discard_ratios) - 1)])
            if num_to_discard > 0:
                _, indices = flat.topk(num_to_discard, dim=-1, largest=False)
                for batch_index in range(batch_size):
                    idx = indices[batch_index]
                    idx = idx[idx != 0]
                    flat[batch_index, idx] = 0
                attn = flat.view(batch_size, num_tokens, num_tokens)
            identity = torch.eye(num_tokens, device=attentions.device, dtype=attentions.dtype).unsqueeze(0).expand(batch_size, -1, -1)
            attn = (attn + identity) / 2.0
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            result = torch.bmm(attn, result)
        return result

    def _prepare_attention(self, attention_weights, batch_size: int, num_tokens: int, device, dtype) -> torch.Tensor:
        if isinstance(attention_weights, (list, tuple)):
            layers = [self._normalize_attention(attn, batch_size, num_tokens, device, dtype) for attn in attention_weights]
            stacked = torch.stack(layers, dim=0)
        elif isinstance(attention_weights, torch.Tensor) and attention_weights.ndim >= 3:
            stacked = attention_weights.unsqueeze(0) if attention_weights.ndim in {3, 4} else attention_weights
            stacked = stacked.to(device=device, dtype=dtype)
        else:
            return torch.ones(batch_size, num_tokens, num_tokens, device=device, dtype=dtype)

        if self.topk_type == 'std' and stacked.size(0) > 1:
            reduced = stacked.std(dim=0, unbiased=False)
        elif self.topk_type == 'layer_index' and 0 <= self.layer_index < stacked.size(0):
            reduced = stacked[self.layer_index]
        elif self.topk_type == 'custom' and stacked.size(0) > 1:
            return self._rollout(stacked)
        else:
            reduced = stacked.mean(dim=0)
        if reduced.ndim == 4:
            reduced = reduced.mean(dim=1)
        return self._normalize_attention(reduced, batch_size, num_tokens, device, dtype)

    def encode_image_branch(self, image_output, return_debug: bool = False, current_step: Optional[int] = None, total_steps: Optional[int] = None) -> Dict[str, object]:    
        global_embedding = image_output.projected_pooled.float()
        outputs = {
            'image_embedding': global_embedding,
            'image_projected': global_embedding,
            'image_projected_raw': global_embedding,
            'image_proxy_features': global_embedding,
            'routing_weights': global_embedding.new_empty((global_embedding.size(0), 0)),
            'summary': global_embedding,
            'global_image_embedding': global_embedding,
            'grab_image_embedding': None,
            'router_debug': {},
            'aggregator_debug': {},
        }
        if not self.only_global:
            attention = self._prepare_attention(
                image_output.attention_weights,
                batch_size=image_output.projected_tokens.size(0),
                num_tokens=image_output.projected_tokens.size(1),
                device=image_output.projected_tokens.device,
                dtype=image_output.projected_tokens.dtype,
            )
            outputs['grab_image_embedding'] = self.visual_embedding_layer(
                image_output.projected_tokens.float(),
                attention.float(),
                current_step=current_step if self.modify_k else None,
                total_steps=total_steps if self.modify_k else None,
            )
        if return_debug:
            outputs['debug'] = {
                'itself_host_type': global_embedding.new_tensor(1.0),
                'itself_only_global': global_embedding.new_tensor(float(self.only_global)),
            }
        return outputs

    def encode_text_branch(self, text_output, token_ids: torch.Tensor, return_debug: bool = False, current_step: Optional[int] = None, total_steps: Optional[int] = None) -> Dict[str, object]:
        global_embedding = text_output.projected_pooled.float()
        outputs = {
            'text_embedding': global_embedding,
            'text_projected': global_embedding,
            'text_projected_raw': global_embedding,
            'global_text_embedding': global_embedding,
            'grab_text_embedding': None,
        }
        if not self.only_global:
            attention = self._prepare_attention(
                text_output.attention_weights,
                batch_size=text_output.projected_tokens.size(0),
                num_tokens=text_output.projected_tokens.size(1),
                device=text_output.projected_tokens.device,
                dtype=text_output.projected_tokens.dtype,
            )
            outputs['grab_text_embedding'] = self.textual_embedding_layer(
                text_output.projected_tokens.float(),
                token_ids.long(),
                attention.float(),
                current_step=current_step if self.modify_k else None,
                total_steps=total_steps if self.modify_k else None,
            )
        if return_debug:
            outputs['debug'] = {
                'itself_host_type': global_embedding.new_tensor(1.0),
                'itself_only_global': global_embedding.new_tensor(float(self.only_global)),
            }
        return outputs

    def compute_similarity_matrix(self, image_features: Dict[str, torch.Tensor], text_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        global_similarity = cosine_similarity_matrix(image_features['global_image_embedding'], text_features['global_text_embedding'])
        if self.only_global or image_features.get('grab_image_embedding') is None or text_features.get('grab_text_embedding') is None:
            return global_similarity
        grab_similarity = cosine_similarity_matrix(image_features['grab_image_embedding'], text_features['grab_text_embedding'])
        return (self.score_weight_global * global_similarity) + ((1.0 - self.score_weight_global) * grab_similarity)

    def _empty_loss_outputs(self, reference: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = reference.new_zeros(())
        return {
            'loss_total': zero,
            'loss_ret': zero,
            'loss_ret_i2t': zero,
            'loss_ret_t2i': zero,
            'loss_proxy': zero,
            'loss_proxy_image': zero,
            'loss_proxy_text': zero,
            'loss_proxy_text_exact': zero,
            'loss_align': zero,
            'loss_diag': zero,
            'loss_support': zero,
            'loss_diversity': zero,
            'loss_balance': zero,
            'loss_proxy_image_weighted': zero,
            'loss_proxy_text_weighted': zero,
            'loss_proxy_text_exact_weighted': zero,
            'loss_proxy_weighted': zero,
            'loss_ret_weighted': zero,
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
            'use_loss_ret': zero,
            'lambda_ret': zero,
            'use_loss_align': zero,
            'lambda_align': zero,
            'use_loss_diag': zero,
            'lambda_diag': zero,
            'use_loss_support': zero,
            'lambda_support': zero,
            'lambda_div': zero,
            'lambda_bal': zero,
            'proxy_temperature': zero,
            'retrieval_temperature': reference.new_tensor(self.tau_itself),
            'logit_scale': reference.new_tensor(1.0 / max(self.tau_itself, 1e-12)),
            'debug_metrics': {},
        }

    # def _compute_cid_loss(self, image_features: torch.Tensor, text_features: torch.Tensor, pids: torch.Tensor, mlp: nn.Module, pair_classifier: nn.Module, id_classifier: nn.Module) -> torch.Tensor:
    #     max_supported_label = int(pair_classifier.out_features) - 2
    #     batch_min_label = int(pids.min().item())
    #     batch_max_label = int(pids.max().item())

    #     similarity = cid_similarity_matrix(image_features, text_features)
    #     hard_negatives = sample_hard_negatives(similarity, pids)
    #     max_label = int(pids.max().item())
    #     new_labels = update_labels_for_negatives(pids, hard_negatives, max_label)
    #     ni_feats, nt_feats, nlabels = create_sample_pairs(image_features, text_features, hard_negatives, new_labels, pids)
    #     z_feats1 = mlp(torch.cat([ni_feats.float(), nt_feats.float()], dim=1))
    #     z_feats2 = mlp(torch.cat([nt_feats.float(), ni_feats.float()], dim=1))
    #     cross_modal_logits1 = pair_classifier(z_feats1.float())
    #     cross_modal_logits2 = pair_classifier(z_feats2.float())
    #     cid_pair = compute_cid(cross_modal_logits1, cross_modal_logits2, nlabels.to(cross_modal_logits1.device))
        
    #     image_logits = id_classifier(image_features.float())
    #     text_logits = id_classifier(text_features.float())
    #     cid_id = compute_id(image_logits, pids) + compute_id(text_logits, pids)
        
    #     return cid_pair + cid_id
    
    def _compute_cid_loss_components(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        pids: torch.Tensor,
        mlp: nn.Module,
        classifier: nn.Module,
        classifier_id: nn.Module,
    ):
        max_supported_label = int(classifier.out_features) - 2
        batch_min_label = int(pids.min().item())
        batch_max_label = int(pids.max().item())

        similarity = cid_similarity_matrix(image_features, text_features)
        hard_negatives = sample_hard_negatives(similarity, pids)

        if not hard_negatives:
            zero = image_features.sum() * 0.0
            return {
                "total": zero,
                "pair": zero,
                "id_image": zero,
                "id_text": zero,
                "pair_acc": zero,
                "id_image_acc": zero,
                "id_text_acc": zero,
            }

        max_label = int(pids.max().item())
        new_labels = update_labels_for_negatives(pids, hard_negatives, max_label)

        ni_feats, nt_feats, nlabels = create_sample_pairs(
            image_features, text_features, hard_negatives, new_labels, pids
        )

        z_feats1 = mlp(torch.cat([ni_feats.float(), nt_feats.float()], dim=1))
        z_feats2 = mlp(torch.cat([nt_feats.float(), ni_feats.float()], dim=1))

        cross_modal_logits1 = classifier(z_feats1.float())
        cross_modal_logits2 = classifier(z_feats2.float())

        cid_pair = compute_cid(
            cross_modal_logits1,
            cross_modal_logits2,
            nlabels.to(cross_modal_logits1.device),
        )

        classifier_id_dtype = _adapter_aligned_input_dtype(image_features, classifier_id)
        image_logits = classifier_id(image_features.to(dtype=classifier_id_dtype)).float()
        text_logits = classifier_id(text_features.to(dtype=classifier_id_dtype)).float()

        cid_id_image = compute_id(image_logits, pids)
        cid_id_text = compute_id(text_logits, pids)
        cid_total = cid_pair + cid_id_image + cid_id_text

        with torch.no_grad():
            pair_pred1 = cross_modal_logits1.argmax(dim=1)
            pair_pred2 = cross_modal_logits2.argmax(dim=1)
            pair_acc1 = (pair_pred1 == nlabels.to(pair_pred1.device)).float().mean()
            pair_acc2 = (pair_pred2 == nlabels.to(pair_pred2.device)).float().mean()
            pair_acc = 0.5 * (pair_acc1 + pair_acc2)

            id_image_acc = (image_logits.argmax(dim=1) == pids).float().mean()
            id_text_acc = (text_logits.argmax(dim=1) == pids).float().mean()

        return {
            "total": cid_total,
            "pair": cid_pair,
            "id_image": cid_id_image,
            "id_text": cid_id_text,
            "pair_acc": pair_acc,
            "id_image_acc": id_image_acc,
            "id_text_acc": id_text_acc,
            "min_pids": pids.min(),
            "max_pids": pids.max(),
            "num_unique_pids": pids.unique().numel(),
            "num_classes": self.num_classes
        }


    def _compute_cid_loss(
        self,
        image_features,
        text_features,
        pids,
        mlp,
        classifier,
        classifier_id,
    ):
        parts = self._compute_cid_loss_components(
            image_features,
            text_features,
            pids,
            mlp,
            classifier,
            classifier_id,
        )
        return parts["total"]

    def forward(
        self,
        image_output,
        text_output,
        token_ids: torch.Tensor,
        pids: Optional[torch.Tensor] = None,
        return_debug: bool = False,
        current_step: Optional[int] = None,
        total_steps: Optional[int] = None,
    ):
        image_features = self.encode_image_branch(
            image_output,
            return_debug=return_debug,
            current_step=current_step,
            total_steps=total_steps,
        )
        text_features = self.encode_text_branch(
            text_output,
            token_ids,
            return_debug=return_debug,
            current_step=current_step,
            total_steps=total_steps,
        )

        similarity = self.compute_similarity_matrix(image_features, text_features)

        base_tensor = image_features["global_image_embedding"]
        zero = base_tensor.new_zeros(())
        losses = self._empty_loss_outputs(base_tensor)

        tal_loss = zero
        tal_loss_i2t = zero
        tal_loss_t2i = zero
        cid_loss = zero

        compute_host_losses = self.use_host_loss and (pids is not None)
        compute_tal = compute_host_losses and ("tal" in self.loss_names)
        compute_cid = (
            compute_host_losses
            and self.training
            and ("cid" in self.loss_names)
            and (self.classifier_global is not None)
        )

        cid_log = {
            "loss_host_cid_pair_global": zero.detach(),
            "loss_host_cid_id_image_global": zero.detach(),
            "loss_host_cid_id_text_global": zero.detach(),
            "loss_host_cid_global": zero.detach(),
            "loss_host_cid_pair_grab": zero.detach(),
            "loss_host_cid_id_image_grab": zero.detach(),
            "loss_host_cid_id_text_grab": zero.detach(),
            "loss_host_cid_grab": zero.detach(),
        }

        cid_acc_log = {}

        # TAL
        if compute_tal:
            tal_total_global, tal_i2t_global, tal_t2i_global = compute_tal_components(
                image_features["global_image_embedding"],
                text_features["global_text_embedding"],
                pids,
                tau=self.tau_itself,
                margin=self.margin,
            )
            tal_loss = tal_total_global
            tal_loss_i2t = tal_i2t_global
            tal_loss_t2i = tal_t2i_global

            if (
                not self.only_global
                and image_features.get("grab_image_embedding") is not None
                and text_features.get("grab_text_embedding") is not None
            ):
                tal_total_grab, tal_i2t_grab, tal_t2i_grab = compute_tal_components(
                    image_features["grab_image_embedding"],
                    text_features["grab_text_embedding"],
                    pids,
                    tau=self.tau_itself,
                    margin=self.margin,
                )
                tal_loss = tal_loss + tal_total_grab
                tal_loss_i2t = tal_loss_i2t + tal_i2t_grab
                tal_loss_t2i = tal_loss_t2i + tal_t2i_grab

        # CID
        if compute_cid:
            cid_global = self._compute_cid_loss_components(
                image_features["global_image_embedding"],
                text_features["global_text_embedding"],
                pids,
                self.mlp_global,
                self.classifier_global,
                self.classifier_id_global,
            )

            cid_loss = cid_loss + cid_global["total"]

            cid_log["loss_host_cid_pair_global"] = cid_global["pair"].detach()
            cid_log["loss_host_cid_id_image_global"] = cid_global["id_image"].detach()
            cid_log["loss_host_cid_id_text_global"] = cid_global["id_text"].detach()
            cid_log["loss_host_cid_global"] = cid_global["total"].detach()

            if "pair_acc" in cid_global:
                cid_acc_log["cid_pair_acc_global"] = cid_global["pair_acc"].detach()
            if "id_image_acc" in cid_global:
                cid_acc_log["cid_id_image_acc_global"] = cid_global["id_image_acc"].detach()
            if "id_text_acc" in cid_global:
                cid_acc_log["cid_id_text_acc_global"] = cid_global["id_text_acc"].detach()

            if (
                not self.only_global
                and self.classifier_grab is not None
                and image_features.get("grab_image_embedding") is not None
                and text_features.get("grab_text_embedding") is not None
            ):
                cid_grab = self._compute_cid_loss_components(
                    image_features["grab_image_embedding"],
                    text_features["grab_text_embedding"],
                    pids,
                    self.mlp_grab,
                    self.classifier_grab,
                    self.classifier_id_grab,
                )

                cid_loss = cid_loss + cid_grab["total"]

                cid_log["loss_host_cid_pair_grab"] = cid_grab["pair"].detach()
                cid_log["loss_host_cid_id_image_grab"] = cid_grab["id_image"].detach()
                cid_log["loss_host_cid_id_text_grab"] = cid_grab["id_text"].detach()
                cid_log["loss_host_cid_grab"] = cid_grab["total"].detach()

                if "pair_acc" in cid_grab:
                    cid_acc_log["cid_pair_acc_grab"] = cid_grab["pair_acc"].detach()
                if "id_image_acc" in cid_grab:
                    cid_acc_log["cid_id_image_acc_grab"] = cid_grab["id_image_acc"].detach()
                if "id_text_acc" in cid_grab:
                    cid_acc_log["cid_id_text_acc_grab"] = cid_grab["id_text_acc"].detach()

        loss_total = tal_loss + cid_loss

        debug_metrics = {
            "itself_score_weight_global": similarity.new_tensor(self.score_weight_global).detach(),
            "itself_score_weight_grab": similarity.new_tensor(1.0 - self.score_weight_global).detach(),
            "itself_loss_tal": tal_loss.detach(),
            "itself_loss_cid": cid_loss.detach(),
            "itself_global_similarity_mean": cosine_similarity_matrix(
                image_features["global_image_embedding"],
                text_features["global_text_embedding"],
            ).mean().detach(),
            "min_pids": cid_grab["min_pids"] if compute_cid and not self.only_global and self.classifier_grab is not None else zero.detach(),
            "max_pids": cid_grab["max_pids"] if compute_cid and not self.only_global and self.classifier_grab is not None else zero.detach(),
            "num_unique_pids": cid_grab["num_unique_pids"] if compute_cid and not self.only_global and self.classifier_grab is not None else zero.detach(),
            "num_classes": cid_grab["num_classes"] if compute_cid and not self.only_global and self.classifier_grab is not None else zero.detach(),
        }

        if (
            not self.only_global
            and image_features.get("grab_image_embedding") is not None
            and text_features.get("grab_text_embedding") is not None
        ):
            debug_metrics["itself_grab_similarity_mean"] = cosine_similarity_matrix(
                image_features["grab_image_embedding"],
                text_features["grab_text_embedding"],
            ).mean().detach()

        debug_metrics.update(cid_log)
        debug_metrics.update(cid_acc_log)

        losses.update(
            {
                "loss_total": loss_total,
                "loss_ret": tal_loss,
                "loss_ret_weighted": tal_loss,
                "loss_ret_i2t": tal_loss_i2t,
                "loss_ret_t2i": tal_loss_t2i,
                "loss_cid": cid_loss,
                "loss_host_cid": cid_loss,
                "use_loss_ret": tal_loss.new_tensor(float("tal" in self.loss_names and self.use_host_loss)),
                "lambda_ret": tal_loss.new_tensor(1.0),
                "debug_metrics": debug_metrics,
            }
        )

        # nếu logger của bạn đọc scalar trực tiếp từ losses,
        # thêm breakdown CID vào losses luôn cho chắc
        losses.update(cid_log)
        losses.update(cid_acc_log)

        return {
            "routing_weights": image_features["routing_weights"],
            "summary": image_features["summary"],
            "image_projected": image_features["global_image_embedding"],
            "image_projected_raw": image_features["global_image_embedding"],
            "surrogate_text_projected": text_features["global_text_embedding"],
            "surrogate_text_projected_raw": text_features["global_text_embedding"],
            "exact_text_projected": text_features["global_text_embedding"],
            "exact_text_projected_raw": text_features["global_text_embedding"],
            "losses": losses,
            "metrics": dict(debug_metrics),
            "debug": dict(debug_metrics),
            "surrogate_pairwise_logits": similarity,
            "host_similarity_logits": similarity,
            "global_image_embedding": image_features["global_image_embedding"],
            "global_text_embedding": text_features["global_text_embedding"],
            "grab_image_embedding": image_features.get("grab_image_embedding"),
            "grab_text_embedding": text_features.get("grab_text_embedding"),
        }

def build_host_head(args, input_dim: int, num_classes: int):
    host_type = str(getattr(args, 'host_type', 'clip')).lower()
    if host_type == 'clip':
        return CLIPHostAdapter(
            input_dim=input_dim,
            projector_output_dim=getattr(args, 'projection_dim', input_dim),
            projector_hidden_dim=getattr(args, 'projector_hidden_dim', input_dim),
            projector_dropout=getattr(args, 'projector_dropout', 0.0),
            projector_type=getattr(args, 'projector_type', 'mlp2'),
            normalize_projector_outputs=getattr(args, 'normalize_projector_outputs', True),
            use_custom_projector=getattr(args, 'use_custom_projector', True),
            special_token_ids=getattr(args, 'special_token_ids', None),
            error_on_empty_kept_tokens=getattr(args, 'error_on_empty_kept_tokens', True),
            contrastive_temperature_init=getattr(args, 'temperature', 0.07),
            use_loss_ret=getattr(args, 'use_host_loss', True),
            lambda_ret=1.0,
            retrieval_mode='clip_bidirectional',
        )
    if host_type == 'itself':
        return ITSELFHostHead(args=args, input_dim=input_dim, num_classes=num_classes)
    raise ValueError(f"Unsupported host_type={host_type!r}. Allowed values: ['clip', 'itself']")
