from typing import Dict, Iterable, Optional

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
    SPECIAL_TOKEN_KEY_ALIASES = {
        'bos': 'bos',
        'bos_token_id': 'bos',
        'sos': 'bos',
        'sos_token_id': 'bos',
        'start_of_text': 'bos',
        'startoftext': 'bos',
        'cls': 'cls',
        'cls_token_id': 'cls',
        'eos': 'eos',
        'eos_token_id': 'eos',
        'eot': 'eos',
        'eot_token_id': 'eos',
        'end_of_text': 'eos',
        'endoftext': 'eos',
        'pad': 'pad',
        'pad_token_id': 'pad',
    }

    def __init__(
        self,
        token_policy: str = 'content_only',
        special_token_ids: Optional[Dict[str, object]] = None,
        error_on_empty_kept_tokens: bool = True,
    ):
        self.token_policy = TOKEN_POLICY_ALIASES.get(str(token_policy).lower())
        if self.token_policy is None:
            raise ValueError(f'Unsupported token policy: {token_policy}')
        self.special_token_ids = self._canonicalize_special_token_ids(special_token_ids)
        self.error_on_empty_kept_tokens = bool(error_on_empty_kept_tokens)

    def _canonicalize_special_token_ids(self, special_token_ids: Optional[Dict[str, object]]) -> Dict[str, tuple]:
        canonical = {
            'bos': (),
            'cls': (),
            'eos': (),
            'pad': (),
        }
        if special_token_ids is None:
            return canonical
        if not isinstance(special_token_ids, dict):
            raise ValueError('special_token_ids must be a mapping of token kinds to token ids.')
        for raw_key, raw_value in special_token_ids.items():
            canonical_key = self.SPECIAL_TOKEN_KEY_ALIASES.get(str(raw_key).lower())
            if canonical_key is None:
                raise ValueError(f'Unsupported special token key: {raw_key}')
            values = raw_value if isinstance(raw_value, (list, tuple, set)) else [raw_value]
            normalized = []
            for value in values:
                if value is None:
                    continue
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f'Special token id for `{raw_key}` must be an integer or list of integers.')
                normalized.append(int(value))
            canonical[canonical_key] = tuple(normalized)
        return canonical

    def _mask_for_token_ids(self, token_ids: torch.Tensor, values: Iterable[int]) -> torch.Tensor:
        values = tuple(int(value) for value in values)
        if not values:
            return torch.zeros_like(token_ids, dtype=torch.bool)
        mask = torch.zeros_like(token_ids, dtype=torch.bool)
        for value in values:
            mask |= token_ids.eq(value)
        return mask

    def _normalize_special_positions(
        self,
        token_ids: torch.Tensor,
        token_valid_mask: torch.Tensor,
        special_token_positions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if special_token_positions is None:
            return {}
        if not isinstance(special_token_positions, dict):
            raise ValueError('special_token_positions must be a dictionary when provided.')

        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        positions = {}
        for raw_key, value in special_token_positions.items():
            if value is None:
                continue
            key = str(raw_key).lower()
            if key == 'bos':
                key = 'cls'
            if key not in {'cls', 'eos'}:
                continue
            if not isinstance(value, torch.Tensor):
                raise ValueError(f'special_token_positions[{raw_key}] must be a tensor.')
            tensor = value.to(device=device, dtype=torch.long)
            if tensor.ndim != 1 or tensor.numel() != batch_size:
                raise ValueError(
                    f'special_token_positions[{raw_key}] must have shape [B], received {tuple(tensor.shape)}.'
                )
            if torch.any((tensor < 0) | (tensor >= seq_len)):
                raise ValueError(f'special_token_positions[{raw_key}] contains out-of-range indices.')
            if not token_valid_mask[torch.arange(batch_size, device=device), tensor].all():
                raise ValueError(f'special_token_positions[{raw_key}] must point to valid tokens.')
            positions[key] = tensor
        return positions

    def _resolve_first_match_positions(
        self,
        token_ids: torch.Tensor,
        token_valid_mask: torch.Tensor,
        values: Iterable[int],
        kind: str,
    ) -> Optional[torch.Tensor]:
        match_mask = self._mask_for_token_ids(token_ids, values) & token_valid_mask
        if not match_mask.any():
            return None
        if not match_mask.any(dim=-1).all():
            raise ValueError(f'Each sample must contain a valid `{kind}` token.')
        seq_len = token_ids.size(1)
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand_as(token_ids)
        masked_positions = positions.masked_fill(~match_mask, seq_len)
        return masked_positions.min(dim=-1).values

    def build_valid_mask(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        special_token_positions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError('token_ids must have shape [B, L].')
        if attention_mask is not None:
            if attention_mask.shape != token_ids.shape:
                raise ValueError('attention_mask must match token_ids shape [B, L].')
            return attention_mask.bool()

        pad_mask = self._mask_for_token_ids(token_ids, self.special_token_ids['pad'])
        if pad_mask.any():
            return ~pad_mask

        normalized_positions = self._normalize_special_positions(
            token_ids,
            torch.ones_like(token_ids, dtype=torch.bool),
            special_token_positions=special_token_positions,
        )
        eos_positions = normalized_positions.get('eos')
        if eos_positions is None:
            eos_positions = self._resolve_first_match_positions(
                token_ids,
                torch.ones_like(token_ids, dtype=torch.bool),
                self.special_token_ids['eos'],
                kind='eos',
            )
        if eos_positions is None:
            raise ValueError(
                'Unable to derive token_valid_mask without attention_mask, pad_token_id, or eos token metadata.'
            )

        positions = torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0)
        return positions <= eos_positions.unsqueeze(1)

    def _build_explicit_position_mask(
        self,
        token_ids: torch.Tensor,
        positions: Dict[str, torch.Tensor],
        key: str,
    ) -> torch.Tensor:
        position_tensor = positions.get(key)
        if position_tensor is None:
            return torch.zeros_like(token_ids, dtype=torch.bool)
        mask = torch.zeros_like(token_ids, dtype=torch.bool)
        batch_indices = torch.arange(token_ids.size(0), device=token_ids.device)
        mask[batch_indices, position_tensor] = True
        return mask

    def get_special_token_positions(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        special_token_positions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        token_valid_mask = self.build_valid_mask(
            token_ids,
            attention_mask=attention_mask,
            special_token_positions=special_token_positions,
        )
        positions = self._normalize_special_positions(
            token_ids,
            token_valid_mask,
            special_token_positions=special_token_positions,
        )

        if 'cls' not in positions:
            cls_positions = self._resolve_first_match_positions(
                token_ids,
                token_valid_mask,
                self.special_token_ids['cls'],
                kind='cls',
            )
            if cls_positions is None:
                cls_positions = self._resolve_first_match_positions(
                    token_ids,
                    token_valid_mask,
                    self.special_token_ids['bos'],
                    kind='bos',
                )
            if cls_positions is not None:
                positions['cls'] = cls_positions

        if 'eos' not in positions:
            eos_positions = self._resolve_first_match_positions(
                token_ids,
                token_valid_mask,
                self.special_token_ids['eos'],
                kind='eos',
            )
            if eos_positions is not None:
                positions['eos'] = eos_positions

        return positions

    def build(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        special_token_positions: Optional[Dict[str, torch.Tensor]] = None,
        return_debug: bool = False,
    ):
        if token_ids.ndim != 2:
            raise ValueError('token_ids must have shape [B, L].')
        token_valid_mask = self.build_valid_mask(
            token_ids,
            attention_mask=attention_mask,
            special_token_positions=special_token_positions,
        )
        positions = self.get_special_token_positions(
            token_ids,
            attention_mask=attention_mask,
            special_token_positions=special_token_positions,
        )
        token_keep_mask = token_valid_mask.clone()

        leading_special_mask = self._mask_for_token_ids(
            token_ids,
            self.special_token_ids['cls'] + self.special_token_ids['bos'],
        )
        eos_mask = self._mask_for_token_ids(token_ids, self.special_token_ids['eos'])
        leading_special_mask |= self._build_explicit_position_mask(token_ids, positions, 'cls')
        eos_mask |= self._build_explicit_position_mask(token_ids, positions, 'eos')

        if self.token_policy == 'content_only':
            has_leading_special = leading_special_mask.any(dim=-1)
            if not has_leading_special.all():
                raise ValueError(
                    'token_policy=content_only requires every sample to provide a valid BOS/CLS token through '
                    'special_token_ids or explicit special_token_positions.'
                )
            has_eos = eos_mask.any(dim=-1)
            if not has_eos.all():
                raise ValueError(
                    'token_policy=content_only requires every sample to provide a valid EOS token through '
                    'special_token_ids or explicit special_token_positions.'
                )
            token_keep_mask &= ~leading_special_mask
            token_keep_mask &= ~eos_mask
        elif self.token_policy == 'content_plus_special':
            pass
        elif self.token_policy == 'eos_only':
            has_eos = eos_mask.any(dim=-1)
            if not has_eos.all():
                raise ValueError(
                    'token_policy=eos_only requires every sample to provide a valid EOS token through '
                    'special_token_ids or explicit special_token_positions.'
                )
            token_keep_mask = eos_mask & token_valid_mask
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
