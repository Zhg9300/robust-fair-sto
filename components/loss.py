import torch.nn as nn
import torch.nn.functional as functional


class HalfSquaredBatchLoss(nn.Module):
    """Return half the squared L2 residual at a fixed reference batch size."""

    def __init__(self, reference_batch_size):
        super().__init__()
        reference_batch_size = int(reference_batch_size)
        if reference_batch_size <= 0:
            raise ValueError("reference_batch_size must be positive.")
        self.reference_batch_size = reference_batch_size

    def forward(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError(
                f"Regression prediction shape {tuple(prediction.shape)} does not "
                f"match target shape {tuple(target.shape)}."
            )
        return 0.5 * self.reference_batch_size * functional.mse_loss(
            prediction,
            target,
            reduction="mean",
        )
