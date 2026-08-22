import math
from itertools import combinations

import torch

from components.aggregator.base import Aggregator, EPS, validate_byzantine_count, validate_updates


class CoordinateWiseTrimmedMean(Aggregator):
    """Coordinate-wise trimmed mean (CWTM)."""

    name = "cwtm"

    def aggregate(self, updates, weights=None, byzantine_count=0):
        updates = validate_updates(updates)
        f = validate_byzantine_count(
            byzantine_count,
            updates.shape[0],
            require_honest_majority=True,
        )
        if f == 0:
            return updates.mean(dim=0)
        sorted_updates = torch.sort(updates, dim=0).values
        return sorted_updates[f:-f].mean(dim=0)


class CoordinateWiseMedian(Aggregator):
    """Coordinate-wise median (CWM), averaging the middle pair for even n."""

    name = "cwm"

    def aggregate(self, updates, weights=None, byzantine_count=0):
        updates = validate_updates(updates)
        sorted_updates = torch.sort(updates, dim=0).values
        client_count = sorted_updates.shape[0]
        midpoint = client_count // 2
        if client_count % 2:
            return sorted_updates[midpoint]
        return 0.5 * (sorted_updates[midpoint - 1] + sorted_updates[midpoint])


class NormBasedScreening(Aggregator):
    """Average client updates after removing the largest L2 norms."""

    name = "nbs"

    def aggregate(self, updates, weights=None, byzantine_count=0):
        updates = validate_updates(updates)
        screen_count = validate_byzantine_count(
            byzantine_count,
            updates.shape[0],
        )
        if screen_count == 0:
            return updates.mean(dim=0)

        norms = torch.linalg.vector_norm(updates, dim=1)
        keep_count = updates.shape[0] - screen_count
        keep_indices = torch.argsort(norms)[:keep_count]
        return updates[keep_indices].mean(dim=0)


class GeometricMedian(Aggregator):
    """Euclidean geometric median using a modified Weiszfeld iteration."""

    name = "median"

    def __init__(self, max_iterations=100, tolerance=1e-6):
        if isinstance(max_iterations, bool) or int(max_iterations) != max_iterations:
            raise ValueError("median_max_iterations must be a positive integer.")
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        if self.max_iterations <= 0:
            raise ValueError("median_max_iterations must be a positive integer.")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("aggregator_tolerance must be positive and finite.")

    def aggregate(self, updates, weights=None, byzantine_count=0):
        updates = validate_updates(updates)
        # CWM is a more robust starting point than the arithmetic mean.
        estimate = CoordinateWiseMedian().aggregate(updates)
        numeric_epsilon = max(EPS, torch.finfo(updates.dtype).eps)

        for _ in range(self.max_iterations):
            offsets = updates - estimate
            distances = torch.linalg.vector_norm(offsets, dim=1)
            nonzero = distances > numeric_epsilon
            zero_count = int((~nonzero).sum().detach().cpu().item())

            if not torch.any(nonzero):
                break

            nonzero_offsets = offsets[nonzero]
            nonzero_distances = distances[nonzero]
            inverse_distances = nonzero_distances.reciprocal()
            candidate = (
                inverse_distances.unsqueeze(1) * updates[nonzero]
            ).sum(dim=0) / inverse_distances.sum()

            if zero_count:
                residual = torch.linalg.vector_norm(
                    (nonzero_offsets / nonzero_distances.unsqueeze(1)).sum(dim=0)
                )
                residual_value = float(residual.detach().cpu().item())
                if residual_value <= zero_count:
                    break
                mixing = min(1.0, zero_count / residual_value)
                candidate = mixing * estimate + (1.0 - mixing) * candidate

            change = torch.linalg.vector_norm(candidate - estimate)
            scale = max(1.0, float(torch.linalg.vector_norm(estimate).detach().cpu().item()))
            estimate = candidate
            if float(change.detach().cpu().item()) <= self.tolerance * scale:
                break

        return estimate


class FABA(Aggregator):
    """Remove the update farthest from the current mean, f times."""

    name = "faba"

    def aggregate(self, updates, weights=None, byzantine_count=0):
        updates = validate_updates(updates)
        f = validate_byzantine_count(
            byzantine_count,
            updates.shape[0],
            require_honest_majority=True,
        )
        remaining = updates
        for _ in range(f):
            center = remaining.mean(dim=0)
            distances = torch.linalg.vector_norm(remaining - center, dim=1)
            remove_index = int(torch.argmax(distances).detach().cpu().item())
            keep = torch.ones(remaining.shape[0], dtype=torch.bool, device=remaining.device)
            keep[remove_index] = False
            remaining = remaining[keep]
        return remaining.mean(dim=0)


class Krum(Aggregator):
    """Select the update with the smallest Krum nearest-neighbor score."""

    name = "krum"

    def aggregate(self, updates, weights=None, byzantine_count=0):
        updates = validate_updates(updates)
        client_count = updates.shape[0]
        f = validate_byzantine_count(
            byzantine_count,
            client_count,
            require_honest_majority=True,
        )
        if client_count < 2 * f + 3:
            raise ValueError(
                "Krum requires at least 2 * byzantine_count + 3 client updates."
            )

        pairwise_squared_distances = torch.cdist(updates, updates).square()
        pairwise_squared_distances.fill_diagonal_(float("inf"))
        neighbor_count = client_count - f - 2
        nearest_distances = torch.topk(
            pairwise_squared_distances,
            k=neighbor_count,
            dim=1,
            largest=False,
            sorted=False,
        ).values
        scores = nearest_distances.sum(dim=1)
        selected_index = int(torch.argmin(scores).detach().cpu().item())
        return updates[selected_index]


class MinimumDiameterAveraging(Aggregator):
    """Average the size-(n-f) subset with minimum pairwise diameter."""

    name = "mda"

    def aggregate(self, updates, weights=None, byzantine_count=0):
        updates = validate_updates(updates)
        client_count = updates.shape[0]
        f = validate_byzantine_count(
            byzantine_count,
            client_count,
            require_honest_majority=True,
        )
        if f == 0:
            return updates.mean(dim=0)

        subset_size = client_count - f
        distances = torch.cdist(updates, updates)
        best_subset = None
        best_diameter = None
        for candidate in combinations(range(client_count), subset_size):
            indices = torch.as_tensor(candidate, device=updates.device)
            diameter = distances.index_select(0, indices).index_select(1, indices).max()
            diameter_value = float(diameter.detach().cpu().item())
            if best_diameter is None or diameter_value < best_diameter:
                best_diameter = diameter_value
                best_subset = candidate

        selected_indices = torch.as_tensor(best_subset, device=updates.device)
        return updates.index_select(0, selected_indices).mean(dim=0)


class CenteredClipping(Aggregator):
    """Stateful centered clipping of client update vectors."""

    name = "centered_clipping"

    def __init__(self, clipping_radius=10.0, iterations=1):
        self.clipping_radius = float(clipping_radius)
        if not math.isfinite(self.clipping_radius) or self.clipping_radius <= 0.0:
            raise ValueError("cc_tau must be positive and finite.")
        if isinstance(iterations, bool) or int(iterations) != iterations or int(iterations) <= 0:
            raise ValueError("cc_iterations must be a positive integer.")
        self.iterations = int(iterations)
        self.center = None

    def reset(self):
        self.center = None

    def aggregate(self, updates, weights=None, byzantine_count=0):
        updates = validate_updates(updates)
        if self.center is None:
            center = torch.zeros_like(updates[0])
        else:
            if self.center.shape != updates.shape[1:]:
                raise ValueError("Centered-Clipping update dimension changed between rounds.")
            center = self.center.to(device=updates.device, dtype=updates.dtype)

        for _ in range(self.iterations):
            residuals = updates - center
            norms = torch.linalg.vector_norm(residuals, dim=1)
            scales = torch.clamp(
                self.clipping_radius / torch.clamp(norms, min=EPS),
                max=1.0,
            )
            center = center + (residuals * scales.unsqueeze(1)).mean(dim=0)

        self.center = center.detach().clone()
        return center
