import math
from statistics import NormalDist

import torch

from components.param_utils import parse_id_list


EPS = 1e-12


class ByzantineAttackController:
    MODES = {
        "alie",
        "gaussian",
        "ipm",
        "label_random_flip",
        "label_cyclic_flip",
        "label_targeted_flip",
        "sign_flip",
        "large_norm",
        "sybil_direction",
        "loss_inflation",
        "loss_deflation",
        "loss_ranking",
        "high_loss_malicious_gradient",
        "low_loss_malicious_gradient",
        "adaptive_copying",
        "multi_decoy_minority",
    }
    LOSS_DEPENDENT_MODES = {
        "loss_inflation",
        "loss_deflation",
        "loss_ranking",
        "high_loss_malicious_gradient",
        "low_loss_malicious_gradient",
    }
    IMPERSONATION_MODES = {
        "adaptive_copying",
        "multi_decoy_minority",
    }
    LABEL_POISONING_MODES = {
        "label_random_flip",
        "label_cyclic_flip",
        "label_targeted_flip",
    }
    LOG_FIELDS = [
        "round",
        "client_id",
        "is_byzantine",
        "attack_mode",
        "alie_z",
        "true_loss",
        "reported_loss",
        "true_grad_norm",
        "reported_grad_norm",
        "true_update_norm",
        "reported_update_norm",
        "effective_update_norm",
        "effective_weight",
        "loss_delta",
        "gradient_cosine_true_reported",
        "target_client_id",
        "path",
        "event_id",
    ]

    def __init__(self, params=None, client_num=0, device=None):
        params = params or {}
        self.client_num = int(client_num or params.get("N", 0) or 0)
        self.device = device
        self.attack_mode = self._parse_mode(params.get("attack_mode", "None"))
        self.dishonest_num = int(params.get("dishonest_num", 0) or 0)
        self.attack_start_round = int(params.get("attack_start_round", 1) or 1)
        self.attack_end_round = self._parse_optional_int(params.get("attack_end_round", None))
        attack_seed = params.get("attack_seed", 1)
        self.attack_seed = 1 if attack_seed is None else int(attack_seed)
        self.attack_scale = float(params.get("attack_scale", 1.0))
        self.alie_z = self._parse_optional_float(params.get("alie_z", None))
        self.loss_bias = float(params.get("loss_bias", 0.0))
        self.copy_loss = self._parse_bool(params.get("copy_loss", False))
        self.copy_gradient = self._parse_bool(params.get("copy_gradient", False))
        self.attack_target_clients = self._parse_id_list(params.get("attack_target_clients", None))
        explicit_byzantine_ids = self._parse_id_list(params.get("byzantine_ids", None))
        self._has_explicit_byzantine_ids = bool(explicit_byzantine_ids)
        if explicit_byzantine_ids and self.dishonest_num <= 0:
            self.dishonest_num = len(explicit_byzantine_ids)
        self.byzantine_ids = self._resolve_byzantine_ids(explicit_byzantine_ids)
        self._byzantine_id_set = set(self.byzantine_ids)
        self.enabled = (
            self.attack_mode is not None
            and self.dishonest_num > 0
            and len(self.byzantine_ids) > 0
        )
        self._validate_config()
        self._label_mapping = self._build_label_mapping()

    @staticmethod
    def _parse_mode(value):
        if value is None:
            return None
        if value is False:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            lowered = stripped.lower()
            if lowered in {"", "none", "null", "false", "off"}:
                return None
            return lowered
        return str(value).lower()

    @staticmethod
    def _parse_optional_int(value):
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "" or stripped.lower() == "none":
                return None
            return int(stripped)
        return int(value)

    @staticmethod
    def _parse_optional_float(value):
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "" or stripped.lower() == "none":
                return None
            return float(stripped)
        return float(value)

    @staticmethod
    def _parse_bool(value):
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        lowered = str(value).strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n", "none", ""}:
            return False
        raise RuntimeError(f"Invalid boolean attack parameter: {value}")

    @classmethod
    def _parse_id_list(cls, value):
        return parse_id_list(value)

    def _resolve_byzantine_ids(self, explicit_ids):
        if explicit_ids:
            return explicit_ids
        if self.dishonest_num <= 0:
            return []
        start = max(0, self.client_num - self.dishonest_num)
        return list(range(start, self.client_num))

    def _validate_config(self):
        if self.attack_mode is not None and self.attack_mode not in self.MODES:
            raise RuntimeError(f"Unsupported attack_mode: {self.attack_mode}")
        if self.dishonest_num < 0:
            raise RuntimeError("dishonest_num must be non-negative.")
        if self.client_num > 0 and self.dishonest_num > self.client_num:
            raise RuntimeError(f"dishonest_num must be <= N ({self.client_num}).")
        if self._has_duplicates(self.byzantine_ids):
            raise RuntimeError("byzantine_ids must not contain duplicate client ids.")
        if self._has_duplicates(self.attack_target_clients):
            raise RuntimeError("attack_target_clients must not contain duplicate client ids.")
        if self._has_explicit_byzantine_ids and self.dishonest_num != len(self.byzantine_ids):
            raise RuntimeError(
                "dishonest_num must match the number of explicit byzantine_ids."
            )
        if self.attack_start_round < 0:
            raise RuntimeError("attack_start_round must be non-negative.")
        if self.attack_end_round is not None and self.attack_end_round < self.attack_start_round:
            raise RuntimeError("attack_end_round must be >= attack_start_round.")
        for client_id in self.byzantine_ids + self.attack_target_clients:
            if self.client_num > 0 and (client_id < 0 or client_id >= self.client_num):
                raise RuntimeError(f"Attack client id {client_id} is outside [0, {self.client_num - 1}].")
        if self.attack_mode is None:
            if self.dishonest_num > 0 or self.byzantine_ids:
                raise RuntimeError("dishonest_num/byzantine_ids require attack_mode.")
            return
        if self.dishonest_num <= 0 or not self.byzantine_ids:
            raise RuntimeError("attack_mode requires at least one Byzantine client.")
        if self.attack_mode == "large_norm" and self.attack_scale <= 1.0:
            raise RuntimeError("large_norm requires attack_scale > 1.")
        if self.attack_mode in {"gaussian", "ipm"} and (
            not math.isfinite(self.attack_scale) or self.attack_scale <= 0.0
        ):
            raise RuntimeError(
                f"{self.attack_mode} requires a finite attack_scale > 0."
            )
        if self.alie_z is not None and not math.isfinite(self.alie_z):
            raise RuntimeError("alie_z must be a finite number.")
        honest_client_num = self.client_num - len(self._byzantine_id_set)
        if (
            self.client_num > 0
            and honest_client_num <= 0
            and self.attack_mode in (
                self.LOSS_DEPENDENT_MODES
                | self.IMPERSONATION_MODES
                | {"alie", "gaussian", "ipm"}
            )
        ):
            raise RuntimeError(f"{self.attack_mode} requires at least one honest client.")
        if self.attack_mode in self.IMPERSONATION_MODES:
            if not self.attack_target_clients:
                raise RuntimeError(f"{self.attack_mode} requires attack_target_clients.")
            if not self.copy_loss and not self.copy_gradient:
                raise RuntimeError(f"{self.attack_mode} requires copy_loss or copy_gradient.")
            for target_id in self.attack_target_clients:
                if target_id in self._byzantine_id_set:
                    raise RuntimeError(f"Attack target client {target_id} must be honest.")

    def experiment_suffix(self):
        if not self.enabled:
            return ""
        byzantine_ids = self._id_suffix(self.byzantine_ids)
        end_round = "None" if self.attack_end_round is None else self.attack_end_round
        parts = [
            f"attack_{self.attack_mode}",
            f"dn{self.dishonest_num}",
            f"ids{byzantine_ids}",
            f"scale{self._suffix_token(self.attack_scale)}",
            f"lossbias{self._suffix_token(self.loss_bias)}",
            f"round{self.attack_start_round}-{end_round}",
        ]
        if self.attack_mode == "alie":
            alie_z = "auto" if self.alie_z is None else self._suffix_token(self.alie_z)
            parts.append(f"aliez{alie_z}")
        if self.attack_mode == "gaussian":
            parts.append(f"attackseed{self.attack_seed}")
        if self.attack_mode == "label_random_flip":
            parts.append(f"labelseed{self.attack_seed}")
        if self.attack_target_clients:
            parts.append(f"target{self._id_suffix(self.attack_target_clients)}")
        if self.copy_loss or self.copy_gradient:
            copied = ""
            if self.copy_loss:
                copied += "L"
            if self.copy_gradient:
                copied += "G"
            parts.append(f"copy{copied}")
        return " ".join(parts)

    @staticmethod
    def _has_duplicates(values):
        return len(values) != len(set(values))

    @classmethod
    def _id_suffix(cls, values):
        if not values:
            return "none"
        return "-".join(cls._suffix_token(value) for value in values)

    @staticmethod
    def _suffix_token(value):
        if value is None:
            return "None"
        if isinstance(value, float):
            text = f"{value:g}"
        else:
            text = str(value)
        return (
            text.strip()
            .replace(" ", "")
            .replace(",", "-")
            .replace("/", "-")
            .replace("\\", "-")
        )

    def is_byzantine(self, client_id):
        return int(client_id) in self._byzantine_id_set

    def is_active(self, round_id):
        if not self.enabled:
            return False
        if int(round_id) < self.attack_start_round:
            return False
        if self.attack_end_round is not None and int(round_id) > self.attack_end_round:
            return False
        return True

    def label_mapping_for(self, client_id, round_id):
        if self.attack_mode not in self.LABEL_POISONING_MODES:
            return None
        if not self.is_byzantine(client_id) or not self.is_active(round_id):
            return None
        return self._label_mapping

    def _build_label_mapping(self):
        if self.attack_mode == "label_cyclic_flip":
            return tuple((label + 1) % 10 for label in range(10))
        if self.attack_mode == "label_targeted_flip":
            return tuple(9 - label for label in range(10))
        if self.attack_mode != "label_random_flip":
            return None

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.attack_seed)
        labels = torch.arange(10)
        while True:
            mapping = torch.randperm(10, generator=generator)
            if torch.all(mapping != labels):
                return tuple(int(label) for label in mapping.tolist())

    def apply(self, round_id, reports, path, old_model_params=None, lr=None):
        if not self.enabled:
            return [self._prepare_report(report, clone_tensors=False) for report in reports], []

        prepared = [self._prepare_report(report) for report in reports]
        if self.is_active(round_id):
            self._apply_active_attack(
                prepared,
                old_model_params,
                lr,
                round_id=round_id,
            )
        rows = [
            self._make_log_row(round_id, report, path, old_model_params, lr)
            for report in prepared
        ]
        return prepared, rows

    def _prepare_report(self, report, clone_tensors=True):
        copy_tensor = self._clone_tensor if clone_tensors else lambda value: value
        true_gradient = copy_tensor(report.get("true_gradient"))
        true_model = copy_tensor(report.get("true_model"))
        true_loss = self._to_float_or_none(report.get("true_loss"))
        return {
            "client_id": int(report["client_id"]),
            "true_loss": true_loss,
            "true_gradient": true_gradient,
            "true_model": true_model,
            "reported_loss": true_loss,
            "reported_gradient": copy_tensor(true_gradient),
            "reported_model": copy_tensor(true_model),
            "target_client_id": "NA",
            "alie_z": "NA",
        }

    def _apply_active_attack(self, reports, old_model_params, lr, round_id):
        byzantine_reports = [
            report for report in reports
            if report["client_id"] in self._byzantine_id_set
        ]
        if not byzantine_reports:
            return

        honest_reports = [
            report for report in reports
            if report["client_id"] not in self._byzantine_id_set
        ]
        honest_losses = [
            report["true_loss"] for report in honest_reports
            if report["true_loss"] is not None
        ]
        shared_sybil_gradient = self._shared_sybil_gradient(byzantine_reports)
        ipm_gradient = self._ipm_gradient(honest_reports, byzantine_reports)
        alie_gradient, effective_alie_z = self._alie_gradient(
            honest_reports,
            byzantine_reports,
        )
        gaussian_gradients = self._gaussian_gradients(
            round_id,
            honest_reports,
            byzantine_reports,
        )
        if effective_alie_z is not None:
            for report in reports:
                report["alie_z"] = effective_alie_z
        ranking_losses = self._ranking_losses(byzantine_reports, honest_losses)
        target_map = self._build_target_map(reports, byzantine_reports)

        for report in byzantine_reports:
            mode = self.attack_mode
            if mode in {"loss_inflation", "high_loss_malicious_gradient"}:
                report["reported_loss"] = self._max_honest_loss(honest_losses) + self.loss_bias
            elif mode in {"loss_deflation", "low_loss_malicious_gradient"}:
                report["reported_loss"] = max(0.0, self._min_honest_loss(honest_losses) - self.loss_bias)
            elif mode == "loss_ranking":
                report["reported_loss"] = ranking_losses[report["client_id"]]

            if mode == "sign_flip":
                self._set_reported_gradient(report, self._sign_flip(report), old_model_params, lr)
            elif mode == "gaussian":
                self._set_reported_gradient(
                    report,
                    gaussian_gradients.get(report["client_id"]),
                    old_model_params,
                    lr,
                )
            elif mode == "large_norm":
                self._set_reported_gradient(report, self._large_norm(report), old_model_params, lr)
            elif mode == "sybil_direction":
                self._set_reported_gradient(report, shared_sybil_gradient, old_model_params, lr)
            elif mode == "ipm":
                self._set_reported_gradient(report, ipm_gradient, old_model_params, lr)
            elif mode == "alie":
                self._set_reported_gradient(report, alie_gradient, old_model_params, lr)
            elif mode in {"high_loss_malicious_gradient", "low_loss_malicious_gradient"}:
                self._set_reported_gradient(report, self._sign_flip(report), old_model_params, lr)
            elif mode in {"adaptive_copying", "multi_decoy_minority"}:
                self._apply_impersonation(report, target_map, old_model_params, lr)

    def _shared_sybil_gradient(self, byzantine_reports):
        gradients = [
            report["true_gradient"]
            for report in byzantine_reports
            if report["true_gradient"] is not None
        ]
        if not gradients:
            return None
        return -self.attack_scale * torch.stack(gradients).mean(dim=0)

    def _ipm_gradient(self, honest_reports, byzantine_reports):
        if self.attack_mode != "ipm":
            return None
        if not any(report["true_gradient"] is not None for report in byzantine_reports):
            return None

        honest_gradients = [
            report["true_gradient"]
            for report in honest_reports
            if report["true_gradient"] is not None
        ]
        if not honest_gradients:
            raise RuntimeError(
                "ipm requires at least one online honest client with a gradient."
            )
        return -self.attack_scale * torch.stack(honest_gradients).mean(dim=0)

    def _gaussian_gradients(self, round_id, honest_reports, byzantine_reports):
        """Return deterministic Gaussian directions with an honest-norm scale."""
        if self.attack_mode != "gaussian":
            return {}

        honest_gradients = [
            report["true_gradient"]
            for report in honest_reports
            if report["true_gradient"] is not None
        ]
        if not honest_gradients:
            raise RuntimeError(
                "gaussian requires at least one online honest client with a gradient."
            )

        mean_honest_norm = sum(
            float(torch.linalg.vector_norm(gradient).detach().cpu().item())
            for gradient in honest_gradients
        ) / len(honest_gradients)
        target_norm = self.attack_scale * mean_honest_norm
        maximum_seed = 2**63 - 1
        gradients = {}

        for report in byzantine_reports:
            reference = report["true_gradient"]
            if reference is None:
                continue
            if target_norm <= EPS:
                gradients[report["client_id"]] = torch.zeros_like(reference)
                continue

            derived_seed = (
                self.attack_seed
                + 1_000_003 * int(round_id)
                + 10_007 * int(report["client_id"])
            ) % maximum_seed
            generator = torch.Generator(device="cpu")
            generator.manual_seed(derived_seed)
            direction = torch.randn(
                reference.shape,
                generator=generator,
                dtype=torch.float64,
                device="cpu",
            )
            direction_norm = torch.linalg.vector_norm(direction)
            if float(direction_norm.item()) <= EPS:
                direction.zero_()
                direction.reshape(-1)[0] = 1.0
                direction_norm = torch.linalg.vector_norm(direction)
            direction = direction * (target_norm / float(direction_norm.item()))
            gradients[report["client_id"]] = direction.to(
                device=reference.device,
                dtype=reference.dtype,
            )
        return gradients

    def _alie_gradient(self, honest_reports, byzantine_reports):
        if self.attack_mode != "alie":
            return None, None

        byzantine_gradients = [
            report["true_gradient"]
            for report in byzantine_reports
            if report["true_gradient"] is not None
        ]
        if not byzantine_gradients:
            return None, None

        honest_gradients = [
            report["true_gradient"]
            for report in honest_reports
            if report["true_gradient"] is not None
        ]
        if not honest_gradients:
            raise RuntimeError(
                "alie requires at least one online honest client with a gradient."
            )

        honest_num = len(honest_gradients)
        byzantine_num = len(byzantine_gradients)
        effective_z = self.alie_z
        if effective_z is None:
            online_num = honest_num + byzantine_num
            required_honest_tail = math.floor(online_num / 2 + 1) - byzantine_num
            cdf_value = (online_num - required_honest_tail) / online_num
            if not 0.0 < cdf_value < 1.0:
                raise RuntimeError(
                    "alie cannot auto-compute z for "
                    f"{honest_num} online honest and {byzantine_num} online Byzantine "
                    f"clients because the normal CDF value is {cdf_value:g}; "
                    "set a finite alie_z explicitly or change online participation."
                )
            effective_z = NormalDist().inv_cdf(cdf_value)

        stacked = torch.stack(honest_gradients)
        honest_mean = stacked.mean(dim=0)
        honest_std = stacked.std(dim=0, unbiased=False)
        return honest_mean - float(effective_z) * honest_std, float(effective_z)

    def _ranking_losses(self, byzantine_reports, honest_losses):
        if self.attack_mode != "loss_ranking":
            return {}
        sorted_losses = sorted(float(loss) for loss in honest_losses)
        if not sorted_losses:
            raise RuntimeError("loss_ranking requires at least one online honest client.")
        byzantine_ids = sorted(report["client_id"] for report in byzantine_reports)
        if len(byzantine_ids) == 1:
            return {byzantine_ids[0]: self._quantile(sorted_losses, 1.0) + self.loss_bias}
        losses = {}
        for idx, client_id in enumerate(byzantine_ids):
            quantile = 0.6 + 0.4 * idx / (len(byzantine_ids) - 1)
            losses[client_id] = self._quantile(sorted_losses, quantile) + self.loss_bias
        return losses

    def _build_target_map(self, reports, byzantine_reports):
        if self.attack_mode not in {"adaptive_copying", "multi_decoy_minority"}:
            return {}
        reports_by_id = {report["client_id"]: report for report in reports}
        for target_id in self.attack_target_clients:
            if target_id not in reports_by_id:
                raise RuntimeError(f"Attack target client {target_id} is not online in this round.")
            if target_id in self._byzantine_id_set:
                raise RuntimeError(f"Attack target client {target_id} must be honest.")
        target_map = {}
        byzantine_ids = sorted(report["client_id"] for report in byzantine_reports)
        if self.attack_mode == "adaptive_copying":
            target_id = self.attack_target_clients[0]
            return {client_id: reports_by_id[target_id] for client_id in byzantine_ids}
        for idx, client_id in enumerate(byzantine_ids):
            target_id = self.attack_target_clients[idx % len(self.attack_target_clients)]
            target_map[client_id] = reports_by_id[target_id]
        return target_map

    def _apply_impersonation(self, report, target_map, old_model_params, lr):
        target = target_map[report["client_id"]]
        report["target_client_id"] = target["client_id"]
        if self.copy_loss:
            report["reported_loss"] = self._to_float_or_none(target["true_loss"]) + self.loss_bias
        if self.copy_gradient and target["true_gradient"] is not None:
            self._set_reported_gradient(report, target["true_gradient"], old_model_params, lr)

    def _sign_flip(self, report):
        if report["true_gradient"] is None:
            return None
        return -self.attack_scale * report["true_gradient"]

    def _large_norm(self, report):
        if report["true_gradient"] is None:
            return None
        return self.attack_scale * report["true_gradient"]

    def _set_reported_gradient(self, report, gradient, old_model_params, lr):
        if gradient is None:
            return
        gradient = self._clone_tensor(gradient)
        report["reported_gradient"] = gradient
        if old_model_params is not None and lr is not None:
            report["reported_model"] = old_model_params.detach().clone() - float(lr) * gradient

    @staticmethod
    def _max_honest_loss(honest_losses):
        if not honest_losses:
            raise RuntimeError("Loss attack requires at least one online honest client.")
        return max(float(loss) for loss in honest_losses)

    @staticmethod
    def _min_honest_loss(honest_losses):
        if not honest_losses:
            raise RuntimeError("Loss attack requires at least one online honest client.")
        return min(float(loss) for loss in honest_losses)

    @staticmethod
    def _quantile(sorted_values, quantile):
        if not sorted_values:
            raise RuntimeError("Cannot compute quantile on empty values.")
        if len(sorted_values) == 1:
            return float(sorted_values[0])
        position = min(max(float(quantile), 0.0), 1.0) * (len(sorted_values) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return float(sorted_values[lower])
        weight = position - lower
        return float((1.0 - weight) * sorted_values[lower] + weight * sorted_values[upper])

    def _make_log_row(self, round_id, report, path, old_model_params, lr):
        true_gradient = report.get("true_gradient")
        reported_gradient = report.get("reported_gradient")
        true_loss = report.get("true_loss")
        reported_loss = report.get("reported_loss")
        return {
            "round": int(round_id),
            "client_id": int(report["client_id"]),
            "is_byzantine": int(report["client_id"] in self._byzantine_id_set),
            "attack_mode": self.attack_mode or "None",
            "alie_z": report.get("alie_z", "NA"),
            "true_loss": self._metric(true_loss),
            "reported_loss": self._metric(reported_loss),
            "true_grad_norm": self._tensor_norm(true_gradient),
            "reported_grad_norm": self._tensor_norm(reported_gradient),
            "true_update_norm": self._update_norm(report.get("true_model"), true_gradient, old_model_params, lr),
            "reported_update_norm": self._update_norm(report.get("reported_model"), reported_gradient, old_model_params, lr),
            "effective_update_norm": "NA",
            "effective_weight": "NA",
            "loss_delta": self._loss_delta(true_loss, reported_loss),
            "gradient_cosine_true_reported": self._cosine(true_gradient, reported_gradient),
            "target_client_id": report.get("target_client_id", "NA"),
            "path": path,
            "event_id": "NA",
        }

    @classmethod
    def _update_norm(cls, model_params, gradient, old_model_params, lr):
        if model_params is not None and old_model_params is not None:
            return cls._tensor_norm(model_params - old_model_params)
        if gradient is not None and lr is not None:
            return cls._tensor_norm(float(lr) * gradient)
        return "NA"

    @staticmethod
    def _clone_tensor(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        return torch.as_tensor(value).detach().clone()

    @staticmethod
    def _to_float_or_none(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    @classmethod
    def _metric(cls, value):
        if value is None:
            return "NA"
        return float(value)

    @classmethod
    def _loss_delta(cls, true_loss, reported_loss):
        if true_loss is None or reported_loss is None:
            return "NA"
        return float(reported_loss) - float(true_loss)

    @classmethod
    def _tensor_norm(cls, value):
        if value is None:
            return "NA"
        return float(torch.norm(value.detach()).cpu().item())

    @staticmethod
    def _cosine(vec_a, vec_b):
        if vec_a is None or vec_b is None:
            return "NA"
        denom = torch.norm(vec_a) * torch.norm(vec_b)
        if float(denom.detach().cpu().item()) <= EPS:
            return 0.0
        cosine = float(((vec_a @ vec_b) / denom).detach().cpu().item())
        return min(1.0, max(-1.0, cosine))
