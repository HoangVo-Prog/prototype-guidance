from typing import Dict, Optional

import torch


TOKEN_POLICY_ALIASES = {
    'all': 'content_plus_special',
    'content_plus_special': 'content_plus_special',
    'content_only': 'content_only',
    'content': 'content_only',
    'non_special': 'content_only',
    'eos_only': 'eos_only',
}


class TokenMaskBuilder:
    def __init__(self, token_policy: str = 'content_only', error_on_empty_kept_tokens: bool = True):
        self.token_policy = TOKEN_POLICY_ALIASES.get(str(token_policy).lower())
        if self.token_policy is None:
            raise ValueError(f'Unsupported token policy: {token_policy}')
        self.error_on_empty_kept_tokens = bool(error_on_empty_kept_tokens)

    def get_special_token_positions(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        special_token_positions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = token_ids.size(0)
        device = token_ids.device
        if special_token_positions is not None:
            cls_positions = special_token_positions.get('cls')
            eos_positions = special_token_positions.get('eos')
            if cls_positions is None or eos_positions is None:
                raise ValueError('special_token_positions must include both `cls` and `eos`.')
            return {
                'cls': cls_positions.to(device=device, dtype=torch.long),
                'eos': eos_positions.to(device=device, dtype=torch.long),
            }

        token_valid_mask = attention_mask.bool() if attention_mask is not None else token_ids != 0
        eos_positions = token_ids.argmax(dim=-1)
        if not token_valid_mask[torch.arange(batch_size, device=device), eos_positions].all():
            raise ValueError('Derived EOS positions must point to valid tokens.')
        return {
            'cls': torch.zeros(batch_size, dtype=torch.long, device=device),
            'eos': eos_positions,
        }

    def build(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        special_token_positions: Optional[Dict[str, torch.Tensor]] = None,
        return_debug: bool = False,
    ):
        if token_ids.ndim != 2:
            raise ValueError('token_ids must have shape [B, L].')
        token_valid_mask = attention_mask.bool() if attention_mask is not None else token_ids != 0
        positions = self.get_special_token_positions(
            token_ids,
            attention_mask=attention_mask,
            special_token_positions=special_token_positions,
        )
        token_keep_mask = token_valid_mask.clone()
        batch_indices = torch.arange(token_ids.size(0), device=token_ids.device)

        if self.token_policy == 'content_only':
            token_keep_mask[:, 0] = False
            token_keep_mask[batch_indices, positions['eos']] = False
        elif self.token_policy == 'content_plus_special':
            pass
        elif self.token_policy == 'eos_only':
            token_keep_mask = torch.zeros_like(token_keep_mask)
            token_keep_mask[batch_indices, positions['eos']] = True
        else:
            raise ValueError(f'Unsupported token policy: {self.token_policy}')

        token_keep_mask &= token_valid_mask
        if self.error_on_empty_kept_tokens and not token_keep_mask.any(dim=-1).all():
            raise ValueError('Each sample must have at least one valid token after applying the token policy.')

        if not return_debug:
            return token_keep_mask
        return token_keep_mask, {
            'token_valid_mask': token_valid_mask,
            'token_keep_mask': token_keep_mask,
            'valid_mask': token_keep_mask,
            'special_token_positions': positions,
        }
