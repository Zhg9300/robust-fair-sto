import math

import torch

from components.DataLoader import DataLoader
from components.dataloaders.indexed_data import IndexedBatchData
from components.loss import HalfSquaredBatchLoss
from components.param_utils import FULL_BATCH, parse_batch_size, parse_id_list


class LinearPopulationLoss:
    """Analytic expected loss from equation (27)."""

    def __init__(self, true_parameter, covariance, theorem_batch_size, noise_std):
        self.true_parameter = true_parameter.detach().cpu().clone()
        self.covariance = covariance.detach().cpu().clone()
        self.theorem_batch_size = int(theorem_batch_size)
        self.noise_std = float(noise_std)

    def __call__(self, module):
        parameter = module.span_model_params_to_vec().detach().cpu()
        error = parameter - self.true_parameter
        quadratic = 0.5 * (error @ self.covariance @ error)
        noise_floor = 0.5 * self.theorem_batch_size * self.noise_std ** 2
        return float(quadratic.item() + noise_floor)


class DataLoader_linear_regression(DataLoader):
    """Finite, reproducible realization of the Section V.A construction."""

    WORKER_COUNT = 10
    HONEST_COUNT = 8
    SUPPORTED_WORKER_COUNTS = (HONEST_COUNT, WORKER_COUNT)
    BYZANTINE_IDS = (8, 9)
    EXCEPTIONAL_WORKER_ID = 7
    DIMENSION = 20
    THEOREM_BATCH_SIZE = 20

    def __init__(self, pool_size=10, batch_size=20, input_require_shape=None,
                 params=None, device="cpu"):
        params = params if params is not None else {}
        pool_size = int(params.get("N", pool_size))
        batch_size = parse_batch_size(params.get("B", batch_size))
        noise_std = float(params.get("linear_noise_std", 1e-3))
        delta = float(params.get("linear_delta", 1.0))
        train_batches = int(params.get("linear_train_batches", 100))
        partition_seed = int(params.get("partition_seed", 1))
        data_device = params.get("data_device", "model")

        self._validate_configuration(
            params,
            pool_size,
            batch_size,
            noise_std,
            delta,
            train_batches,
            data_device,
        )
        self._apply_va_defaults(params, pool_size)

        nickname = (
            f"linear VA sigma{noise_std:g} delta{delta:g} "
            f"K{self.THEOREM_BATCH_SIZE} batches{train_batches}"
        )
        name = (
            f"linear_va_N{pool_size}_d{self.DIMENSION}_K{self.THEOREM_BATCH_SIZE}_"
            f"sigma{noise_std:g}_delta{delta:g}_batches{train_batches}_seed{partition_seed}"
        )
        super().__init__(
            name=name,
            nickname=nickname,
            pool_size=pool_size,
            batch_size=batch_size,
            input_require_shape=input_require_shape,
        )

        self.device = torch.device(device)
        self.data_device = data_device
        self.preserve_pool_order = True
        self.task_type = "linear_regression"
        self.input_data_shape = [self.DIMENSION]
        self.target_class_num = 1
        self.noise_std = noise_std
        self.delta = delta
        self.train_batches = train_batches
        self.partition_seed = partition_seed
        self.honest_client_ids = tuple(range(self.HONEST_COUNT))
        self.byzantine_client_ids = (
            self.BYZANTINE_IDS
            if pool_size == self.WORKER_COUNT
            else ()
        )
        self.v = torch.zeros(self.DIMENSION)
        self.theta = torch.zeros(self.DIMENSION)
        self.theta[0] = 1.0
        self.covariance = torch.eye(self.DIMENSION)
        self.true_parameters = self._build_true_parameters()
        self.honest_average_optimum = self.delta / self.HONEST_COUNT * self.theta
        self.initial_model_parameter = self.honest_average_optimum.clone()
        params["linear_initialization"] = "honest_average_optimum"
        self.data_pool = self._build_data_pool()

    def initialize_model(self, module):
        """Start every V.A run from the honest average-loss optimum w_H*."""
        current = module.span_model_params_to_vec()
        initial = self.initial_model_parameter.to(
            device=current.device,
            dtype=current.dtype,
        )
        if initial.shape != current.shape:
            raise ValueError(
                "Linear-regression initial parameter shape does not match the model: "
                f"{tuple(initial.shape)} != {tuple(current.shape)}."
            )
        module.reshape_vec_to_model_params(initial)

    @classmethod
    def _validate_configuration(cls, params, pool_size, batch_size, noise_std,
                                delta, train_batches, data_device):
        if pool_size not in cls.SUPPORTED_WORKER_COUNTS:
            raise ValueError(
                "Section V.A requires N=8 for the honest-only mean baseline "
                "or N=10 when the two Byzantine slots are present."
            )
        if not math.isclose(float(params.get("C", 1.0)), 1.0):
            raise ValueError("Section V.A requires full worker participation (C=1).")
        if batch_size not in {cls.THEOREM_BATCH_SIZE, FULL_BATCH}:
            raise ValueError(
                f"Section V.A requires B={cls.THEOREM_BATCH_SIZE} or B='full'."
            )
        if not math.isfinite(noise_std) or noise_std <= 0.0:
            raise ValueError("linear_noise_std must be positive and finite.")
        if not math.isfinite(delta) or delta <= 0.0:
            raise ValueError("linear_delta must be positive and finite.")
        if train_batches <= 0:
            raise ValueError("linear_train_batches must be positive.")
        if data_device not in {"model", "cpu"}:
            raise ValueError("data_device must be 'model' or 'cpu'.")

        if params.get("algorithm") == "qFedAvg":
            if not math.isclose(float(params.get("q", 0.1)), 1.0):
                raise ValueError("Section V.A qFFL experiments require q=1.")
            if params.get("qffl_update_rule", "normalized") != "objective_gradient":
                raise ValueError(
                    "Section V.A qFFL experiments require "
                    "qffl_update_rule='objective_gradient'."
                )
            if int(params.get("E", 1)) != 1:
                raise ValueError("Section V.A qFFL experiments require E=1.")

        attack_mode = str(params.get("attack_mode", "None")).strip().lower()
        if attack_mode not in {"", "none"}:
            if pool_size != cls.WORKER_COUNT:
                raise ValueError("Section V.A attacks require N=10.")
            if int(params.get("dishonest_num", 0)) != len(cls.BYZANTINE_IDS):
                raise ValueError("Section V.A attacks require dishonest_num=2.")
            explicit_ids = parse_id_list(params.get("byzantine_ids"))
            if explicit_ids and tuple(explicit_ids) != cls.BYZANTINE_IDS:
                raise ValueError("Section V.A Byzantine worker ids must be 8 and 9.")

        excluded_ids = parse_id_list(params.get("evaluation_excluded_ids"))
        if pool_size == cls.HONEST_COUNT and excluded_ids:
            raise ValueError(
                "The N=8 honest-only baseline evaluates all eight workers."
            )
        if (pool_size == cls.WORKER_COUNT
                and excluded_ids
                and tuple(excluded_ids) != cls.BYZANTINE_IDS):
            raise ValueError("Section V.A evaluation must exclude worker ids 8 and 9.")

    @classmethod
    def _apply_va_defaults(cls, params, pool_size):
        if params.get("gradient_aggregator_f") is None:
            params["gradient_aggregator_f"] = (
                len(cls.BYZANTINE_IDS)
                if pool_size == cls.WORKER_COUNT
                else 0
            )
        if pool_size == cls.HONEST_COUNT:
            params["evaluation_excluded_ids"] = []
        elif not parse_id_list(params.get("evaluation_excluded_ids")):
            params["evaluation_excluded_ids"] = list(cls.BYZANTINE_IDS)

    def _build_true_parameters(self):
        parameters = []
        for worker_id in range(self.pool_size):
            parameter = self.v.clone()
            if worker_id == self.EXCEPTIONAL_WORKER_ID:
                parameter = parameter + self.delta * self.theta
            parameters.append(parameter)
        return parameters

    def _build_data_pool(self):
        residency_device = self.device if self.data_device == "model" else torch.device("cpu")
        row_count = self.train_batches * self.THEOREM_BATCH_SIZE
        inputs = torch.eye(self.DIMENSION).repeat(self.train_batches, 1)
        inputs = inputs.to(residency_device)
        indices = torch.arange(row_count, device=residency_device)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.partition_seed)

        data_pool = []
        for true_parameter in self.true_parameters:
            noise = self.noise_std * torch.randn(
                row_count,
                1,
                generator=generator,
            )
            targets = inputs.cpu() @ true_parameter.reshape(-1, 1) + noise
            targets = targets.to(residency_device)
            data_pool.append({
                "local_training_data": IndexedBatchData(
                    inputs,
                    targets,
                    indices,
                    self.batch_size,
                ),
                "local_training_number": row_count,
                "local_test_data": None,
                "local_test_number": 1,
                "test_loss_evaluator": LinearPopulationLoss(
                    true_parameter,
                    self.covariance,
                    self.THEOREM_BATCH_SIZE,
                    self.noise_std,
                ),
            })
        return data_pool

    def build_criterion(self):
        return HalfSquaredBatchLoss(self.THEOREM_BATCH_SIZE)

    @staticmethod
    def build_metrics():
        return []

    def population_losses(self, parameter, worker_ids=None):
        parameter = torch.as_tensor(parameter).detach().cpu()
        if worker_ids is None:
            worker_ids = range(self.pool_size)
        noise_floor = 0.5 * self.THEOREM_BATCH_SIZE * self.noise_std ** 2
        losses = []
        for worker_id in worker_ids:
            error = parameter - self.true_parameters[worker_id]
            losses.append(
                0.5 * float(error @ self.covariance @ error) + noise_floor
            )
        return losses

    def theorem_metrics(self, module):
        parameter = module.span_model_params_to_vec().detach().cpu()
        optimum_losses = self.population_losses(
            self.honest_average_optimum,
            self.honest_client_ids,
        )
        current_losses = self.population_losses(parameter, self.honest_client_ids)
        return {
            "honest_average_loss_gap": sum(current_losses) / self.HONEST_COUNT
            - sum(optimum_losses) / self.HONEST_COUNT,
            "distance_to_v": float(torch.linalg.vector_norm(parameter - self.v).item()),
        }
