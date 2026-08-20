from abc import ABC, abstractmethod

import torch


EPS = 1e-12


def validate_updates(updates):
    """Return a finite [client, parameter] tensor."""
    if not torch.is_tensor(updates):
        raise TypeError("Client updates must be provided as a torch.Tensor.")
    if updates.ndim != 2 or updates.shape[0] == 0 or updates.shape[1] == 0:
        raise ValueError("Client updates must be a non-empty two-dimensional tensor.")
    if not torch.is_floating_point(updates):
        raise TypeError("Client updates must use a floating-point dtype.")
    if not torch.all(torch.isfinite(updates)):
        raise ValueError("Client updates contain non-finite values.")
    return updates


def validate_byzantine_count(byzantine_count, client_count, require_honest_majority=False):
    if isinstance(byzantine_count, bool):
        raise ValueError("byzantine_count must be a non-negative integer.")
    try:
        value = int(byzantine_count)
    except (TypeError, ValueError) as exc:
        raise ValueError("byzantine_count must be a non-negative integer.") from exc
    if value != byzantine_count or value < 0:
        raise ValueError("byzantine_count must be a non-negative integer.")
    if value >= client_count:
        raise ValueError("byzantine_count must be smaller than the number of client updates.")
    if require_honest_majority and 2 * value >= client_count:
        raise ValueError(
            "This aggregator requires more than 2 * byzantine_count client updates."
        )
    return value


class Aggregator(ABC):
    """Interface for aggregating flattened client gradients/updates."""

    name = "aggregator"

    @abstractmethod
    def aggregate(self, updates, weights=None, byzantine_count=0):
        raise NotImplementedError

class Mean(Aggregator):
    name = "mean"

    def aggregate(self, updates, weights=None, byzantine_count=0):
        updates = validate_updates(updates)
        if weights is None:
            return updates.mean(dim=0)

        weights = torch.as_tensor(weights, device=updates.device, dtype=updates.dtype).reshape(-1)
        if weights.numel() != updates.shape[0]:
            raise ValueError("Aggregation weights must match the number of client updates.")
        if not torch.all(torch.isfinite(weights)):
            raise ValueError("Aggregation weights contain non-finite values.")
        denominator = weights.sum()
        if (
            not torch.isfinite(denominator)
            or float(torch.abs(denominator).detach().cpu().item()) <= EPS
        ):
            return updates.mean(dim=0)
        return (weights / denominator) @ updates
