import torch.nn as nn
import torch.nn.functional as functional


class HalfSquaredBatchLoss(nn.Module):
    """Return half the squared L2 residual for a theorem-sized batch."""

    def __init__(self, theorem_batch_size):
        super().__init__()
        theorem_batch_size = int(theorem_batch_size)
        if theorem_batch_size <= 0:
            raise ValueError("theorem_batch_size must be positive.")
        self.theorem_batch_size = theorem_batch_size

    def forward(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError(
                f"Regression prediction shape {tuple(prediction.shape)} does not "
                f"match target shape {tuple(target.shape)}."
            )
        return 0.5 * self.theorem_batch_size * functional.mse_loss(
            prediction,
            target,
            reduction="mean",
        )
