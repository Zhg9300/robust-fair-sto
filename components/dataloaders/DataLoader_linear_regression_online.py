"""Online independent Gaussian batches for the theory-matched experiment."""

import torch

from components.dataloaders.DataLoader_linear_regression import (
    DataLoader_linear_regression,
    LinearPopulationLoss,
)
from components.param_utils import parse_bool
from linear_regression_theory_conditions import validate_separation


class OnlineLinearBatchData:
    """One fresh batch per access; no finite training dataset is stored."""

    def __init__(self, inputs, true_parameter, noise_std, seed, client_id):
        self.inputs = inputs
        self.true_parameter = true_parameter.to(inputs.device).reshape(-1, 1)
        self.noise_std = noise_std
        self.batch_size = len(inputs)
        self.generator = torch.Generator(device="cpu")
        # Use a fixed stride, not the current participant count, to pair N=8/10.
        self.generator.manual_seed((10 * seed + client_id) % (2 ** 63))

    def __len__(self):
        return 1

    def __getitem__(self, position):
        if position not in (0, -1):
            raise IndexError(position)
        noise = torch.randn(
            self.batch_size, 1, generator=self.generator, dtype=self.inputs.dtype,
        ).to(self.inputs.device)
        targets = self.inputs @ self.true_parameter + self.noise_std * noise
        return self.inputs, targets

    def __iter__(self):
        yield self[0]


class DataLoader_linear_regression_online(DataLoader_linear_regression):
    """Reuse worker geometry and analytic evaluation, with online training data."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = (
            f"linear_regression_theory_online_N{self.pool_size}_d{self.DIMENSION}_"
            f"sigma{self.noise_std:g}_delta{self.delta:g}_seed{self.partition_seed}"
        )
        self.nickname = (
            f"linear regression theory_online sigma{self.noise_std:g} "
            f"delta{self.delta:g} K{self.REFERENCE_BATCH_SIZE}"
        )

    @classmethod
    def _validate_configuration(cls, params, pool_size, batch_size, noise_std,
                                delta, train_batches, data_device):
        super()._validate_configuration(
            params, pool_size, batch_size, noise_std, delta, train_batches, data_device,
        )
        if batch_size != cls.REFERENCE_BATCH_SIZE:
            raise ValueError("Online linear regression requires B=20.")
        if not parse_bool(params.get("sgd_step", False), "sgd_step"):
            raise ValueError("Online linear regression requires sgd_step=True.")
        if int(params.get("E", 1)) != 1:
            raise ValueError("Online linear regression requires E=1.")
        if (params.get("algorithm") == "qFedAvg"
                and params.get("qffl_loss_mode") != "population"):
            raise ValueError("Online qFFL requires qffl_loss_mode='population'.")
        if (params.get("lr_schedule") != "polynomial"
                or float(params.get("decay", 1)) != 1):
            raise ValueError("Online theory experiments require polynomial steps and decay=1.")
        radius = params.get("projection_radius")
        if radius is None or not delta / 2 <= float(radius) < float("inf"):
            raise ValueError("Online theory experiments require finite projection_radius >= delta/2.")
        attack = str(params.get("attack_mode", "None")).strip().lower()
        if attack not in {"", "none"}:
            validate_separation(
                (params.get("gradient_aggregator", "mean"),), noise_std, delta,
            )

    def _build_data_pool(self):
        residency = self.device if self.data_device == "model" else torch.device("cpu")
        inputs = torch.eye(self.DIMENSION, device=residency)
        return [
            {
                "local_training_data": OnlineLinearBatchData(
                    inputs, parameter, self.noise_std, self.partition_seed, client_id,
                ),
                # Equal nominal one-round counts preserve FedAvg's uniform weights.
                "local_training_number": self.REFERENCE_BATCH_SIZE,
                "local_test_data": None,
                "local_test_number": 1,
                "test_loss_evaluator": LinearPopulationLoss(
                    parameter, self.covariance, self.REFERENCE_BATCH_SIZE, self.noise_std,
                ),
            }
            for client_id, parameter in enumerate(self.true_parameters)
        ]
