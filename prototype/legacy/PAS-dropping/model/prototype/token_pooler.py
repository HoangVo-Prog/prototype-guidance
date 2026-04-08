from typing import Dict, Tuple

import torch
import torch.nn as nn


class MaskedTokenPooler(nn.Module):
    def compute_weights(self, logits: torch.Tensor, keep_mask: torch.Tensor, return_debug: bool = False):
        if logits.ndim != 2:
            raise ValueError('logits must have shape [B, L].')
        if keep_mask.ndim != 2:
            raise ValueError('keep_mask must have shape [B, L].')
        if logits.shape != keep_mask.shape:
            raise ValueError('logits and keep_mask must share shape [B, L].')
        if not keep_mask.any(dim=-1).all():
            raise ValueError('Each sample must have at least one kept token.')

        beta_logits_masked = logits.masked_fill(~keep_mask, float('-inf'))
        beta = torch.softmax(beta_logits_masked, dim=-1)
        beta = beta.masked_fill(~keep_mask, 0.0)
        beta_row_sums = beta.sum(dim=-1, keepdim=True)
        if torch.any(beta_row_sums <= 0):
            raise FloatingPointError('MaskedTokenPooler received an invalid normalization term.')
        beta = beta / beta_row_sums
        if not torch.isfinite(beta).all():
            raise FloatingPointError('MaskedTokenPooler produced non-finite token weights.')

        if not return_debug:
            return beta, beta_logits_masked
        return beta, beta_logits_masked, {
            'beta_logits_masked': beta_logits_masked,
            'masked_logits': beta_logits_masked,
            'token_weights': beta,
            'beta_max_prob': beta.max(dim=-1).values.mean().detach(),
            'token_pool_entropy': (-(beta * beta.clamp_min(1e-12).log()).sum(dim=-1).mean()).detach(),
        }

    def pool(self, token_states: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        if token_states.ndim != 3:
            raise ValueError('token_states must have shape [B, L, D].')
        if beta.ndim != 2:
            raise ValueError('beta must have shape [B, L].')
        if token_states.shape[:2] != beta.shape:
            raise ValueError('token_states leading dimensions must match beta.')
        pooled = torch.bmm(beta.unsqueeze(1), token_states).squeeze(1)
        if not torch.isfinite(pooled).all():
            raise FloatingPointError('MaskedTokenPooler produced non-finite pooled text outputs.')
        return pooled

    def forward(self, token_scores: torch.Tensor, token_states: torch.Tensor, valid_mask: torch.Tensor, return_debug: bool = False):
        if token_scores.ndim != 2:
            raise ValueError('token_scores must have shape [B, L].')
        if token_states.ndim != 3:
            raise ValueError('token_states must have shape [B, L, D].')
        if valid_mask.ndim != 2:
            raise ValueError('valid_mask must have shape [B, L].')
        if token_scores.shape != valid_mask.shape:
            raise ValueError('token_scores and valid_mask must share shape [B, L].')
        if token_states.shape[:2] != token_scores.shape:
            raise ValueError('token_states leading dimensions must match token_scores.')

        beta, beta_logits_masked, weight_debug = self.compute_weights(token_scores, valid_mask, return_debug=True)
        pooled = self.pool(token_states, beta)

        if not return_debug:
            return pooled, beta
        return pooled, beta, {
            **weight_debug,
            'pooled_text': pooled,
        }
