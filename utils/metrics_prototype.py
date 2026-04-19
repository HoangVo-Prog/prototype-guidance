from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def compute_proto_label_nmi(proto_ids: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute NMI between prototype top-1 assignments and labels."""
    try:
        from sklearn.metrics import normalized_mutual_info_score
    except Exception:
        return float('nan')

    proto_ids_np = proto_ids.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    return float(normalized_mutual_info_score(labels_np, proto_ids_np))


def compute_proto_cosine_offdiag_max(prototype_bank: torch.Tensor) -> float:
    """Maximum off-diagonal cosine similarity across prototype vectors."""
    if not isinstance(prototype_bank, torch.Tensor) or prototype_bank.ndim != 2:
        return float('nan')
    if prototype_bank.size(0) <= 1:
        return 0.0
    prototypes = F.normalize(prototype_bank.detach().float(), dim=-1)
    similarity = torch.matmul(prototypes, prototypes.t())
    count = similarity.size(0)
    mask = ~torch.eye(count, dtype=torch.bool, device=similarity.device)
    offdiag = similarity[mask]
    if offdiag.numel() == 0:
        return 0.0
    return float(offdiag.max().item())


def compute_proto_high_sim_ratio(prototype_bank: torch.Tensor, threshold: float = 0.8) -> float:
    """Fraction of off-diagonal prototype pairs with cosine > threshold."""
    if not isinstance(prototype_bank, torch.Tensor) or prototype_bank.ndim != 2:
        return float('nan')
    if prototype_bank.size(0) <= 1:
        return 0.0
    prototypes = F.normalize(prototype_bank.detach().float(), dim=-1)
    similarity = torch.matmul(prototypes, prototypes.t())
    count = similarity.size(0)
    mask = ~torch.eye(count, dtype=torch.bool, device=similarity.device)
    offdiag = similarity[mask]
    total = int(offdiag.numel())
    if total == 0:
        return 0.0
    high = float((offdiag > float(threshold)).float().sum().item())
    return float(high / float(total))


def compute_assignment_overlap_mean(proto_ids: torch.Tensor, num_prototypes: int) -> float:
    """Mean pairwise Jaccard overlap between prototype assignment sets."""
    if not isinstance(proto_ids, torch.Tensor) or proto_ids.ndim != 1:
        return float('nan')
    if int(num_prototypes) <= 1:
        return 0.0

    detached = proto_ids.detach().cpu()
    proto_to_samples = {int(k): set() for k in range(int(num_prototypes))}
    for sample_index, proto_index in enumerate(detached):
        proto_to_samples[int(proto_index.item())].add(int(sample_index))

    overlaps = []
    keys = list(proto_to_samples.keys())
    for left in range(len(keys)):
        for right in range(left + 1, len(keys)):
            a = proto_to_samples[keys[left]]
            b = proto_to_samples[keys[right]]
            if len(a) == 0 or len(b) == 0:
                continue
            inter = len(a & b)
            union = len(a | b)
            if union > 0:
                overlaps.append(float(inter) / float(union))
    if not overlaps:
        return 0.0
    return float(sum(overlaps) / float(len(overlaps)))


def resolve_prototype_bank_tensor(model) -> Optional[torch.Tensor]:
    """Best-effort resolver for the active prototype bank tensor."""
    core_model = model.module if hasattr(model, 'module') else model
    prototype_head = getattr(core_model, 'prototype_head', None)
    if prototype_head is None:
        return None
    prototype_bank = getattr(prototype_head, 'prototype_bank', None)
    if prototype_bank is None:
        return None
    if hasattr(prototype_bank, 'get_prototypes'):
        return prototype_bank.get_prototypes()
    if isinstance(prototype_bank, torch.Tensor):
        return prototype_bank
    return None
