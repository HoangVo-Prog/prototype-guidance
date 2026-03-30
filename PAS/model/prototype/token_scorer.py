from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


TOKEN_SCORING_ALIASES = {
    'cosine': 'cosine',
    'normalized_dot': 'cosine',
    'dot': 'dot',
    'prototype_summary_dot': 'dot',
}


class TokenScorer(nn.Module):
    def __init__(self, scoring_type: str = 'cosine', temperature: float = 0.07):
        super().__init__()
        self.scoring_type = TOKEN_SCORING_ALIASES.get(str(scoring_type).lower())
        if self.scoring_type is None:
            raise ValueError(f'Unsupported token scoring type: {scoring_type}')
        if temperature <= 0:
            raise ValueError('temperature must be positive.')
        self.temperature = float(temperature)

    def forward(self, query: torch.Tensor, token_states: torch.Tensor, return_debug: bool = False):
        if query.ndim != 2:
            raise ValueError('query must have shape [B, D].')
        if token_states.ndim != 3:
            raise ValueError('token_states must have shape [B, L, D].')
        if query.size(0) != token_states.size(0) or query.size(-1) != token_states.size(-1):
            raise ValueError('query and token_states must share batch and feature dimensions.')

        if self.scoring_type == 'cosine':
            query = F.normalize(query, dim=-1)
            token_states = F.normalize(token_states, dim=-1)

        similarity = (token_states * query.unsqueeze(1)).sum(dim=-1)
        scores = similarity / self.temperature
        if not torch.isfinite(scores).all():
            raise FloatingPointError('TokenScorer produced non-finite token scores.')
        if not return_debug:
            return scores
        return scores, {
            'token_similarity': similarity,
            'token_scores': scores,
            'token_scores_scaled': scores,
        }
