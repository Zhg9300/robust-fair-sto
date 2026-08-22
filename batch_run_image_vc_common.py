"""Experiment-group definitions shared by the eight V.C batch runners.

This module only builds the ``base``/``grid`` dictionaries consumed by
``batch_run.py``. GPU scheduling, command execution, logging, and result
writing deliberately remain in ``batch_run.py``.
"""

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional


AGGREGATORS = [
    "mean",
    "cwm",
    "cwtm",
    "median",
    "krum",
    "mda",
    "faba",
    "nbs",
]

ATTACK_CONFIGS = {
    "no_attack": {
        "attack_mode": "None",
        "dishonest_num": 0,
        "gradient_aggregator": "mean",
        "gradient_aggregator_f": 0,
    },
    "gaussian": {
        "attack_mode": "gaussian",
        "dishonest_num": 2,
        "byzantine_ids": "8,9",
        "attack_scale": 5.0,
        "gradient_aggregator_f": 2,
    },
    "sign_flip": {
        "attack_mode": "sign_flip",
        "dishonest_num": 2,
        "byzantine_ids": "8,9",
        "attack_scale": 1.0,
        "gradient_aggregator_f": 2,
    },
    "adaptive_copying": {
        "attack_mode": "adaptive_copying",
        "dishonest_num": 2,
        "byzantine_ids": "8,9",
        "attack_scale": 1.0,
        "attack_target_clients": "0",
        "copy_loss": True,
        "copy_gradient": True,
        "gradient_aggregator_f": 2,
    },
}


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    module: str
    rounds: int
    pathological_loader: str
    dirichlet_loader: str


CIFAR10 = DatasetSpec(
    slug="cifar10",
    module="CNN",
    rounds=3000,
    pathological_loader="DataLoader_cifar10_pat",
    dirichlet_loader="DataLoader_cifar10_dir",
)

FASHION = DatasetSpec(
    slug="fashion",
    module="MLP",
    rounds=2000,
    pathological_loader="DataLoader_fashion_pat",
    dirichlet_loader="DataLoader_fashion_dir",
)


def _partition_configs(spec: DatasetSpec) -> List[Dict[str, object]]:
    return [
        {
            "name": "pathological_nc1",
            "dataloader": spec.pathological_loader,
            "NC": 1,
        },
        {
            "name": "dirichlet_alpha0.5",
            "dataloader": spec.dirichlet_loader,
            "NC": 1,
            "Diralpha": 0.5,
        },
    ]


def _algorithm_grids() -> List[Dict[str, object]]:
    return [
        {
            "name": "qffl",
            "base": {
                "algorithm": "qFedAvg",
                "qffl_update_rule": "objective_gradient",
                "decay": 1.0,
            },
            "grid": {
                "q": [0.1, 0.5, 1.0],
                "lr": [0.1, 0.2, 0.4, 0.8],
            },
        },
        {
            "name": "drfl",
            "base": {
                "algorithm": "DRFL",
                "decay": 0.999,
            },
            "grid": {
                "lr": [0.3, 0.4, 0.5],
            },
        },
        {
            "name": "afl",
            "base": {
                "algorithm": "AFL",
                "decay": 0.999,
            },
            "grid": {
                "lam": [0.05, 0.1, 0.4, 0.8],
                "lr": [0.01, 0.05, 0.1],
            },
        },
    ]


def get_experiment_groups(
    spec: DatasetSpec,
    scenario: str,
) -> List[Dict[str, object]]:
    """Return ``batch_run.py`` experiment groups for one dataset/attack."""
    if scenario not in ATTACK_CONFIGS:
        choices = ", ".join(sorted(ATTACK_CONFIGS))
        raise ValueError(f"Unknown scenario {scenario!r}; expected one of: {choices}")

    attack = ATTACK_CONFIGS[scenario]
    attacked = scenario != "no_attack"
    common = {
        "seed": 1,
        "partition_seed": 1,
        "attack_seed": 1,
        "module": spec.module,
        "N": 10,
        "C": 1.0,
        "B": 64,
        "micro_batch_size": 0,
        "data_device": "model",
        "min_client_samples": "auto",
        "balance": True,
        "R": spec.rounds,
        "E": 1,
        "test_interval": 50,
        "sgd_step": True,
        "momentum": 0.0,
        "weight_decay": 5e-4,
        "attack_start_round": 1,
        "attack_end_round": "None",
        "loss_bias": 0.0,
        "evaluation_excluded_ids": "8,9",
        **attack,
    }

    groups = []
    for partition in _partition_configs(spec):
        partition_name = str(partition["name"])
        partition_base = {
            key: value for key, value in partition.items() if key != "name"
        }
        for algorithm in _algorithm_grids():
            grid = dict(algorithm["grid"])
            if attacked:
                grid["gradient_aggregator"] = list(AGGREGATORS)
            groups.append({
                "name": (
                    f"vc_{spec.slug}_{scenario}_{partition_name}_"
                    f"{algorithm['name']}"
                ),
                "base": {
                    **common,
                    **partition_base,
                    **algorithm["base"],
                },
                "grid": grid,
            })
    return groups


def use_default_gpus(gpus: Optional[str] = "2,3") -> None:
    """Apply runner defaults while preserving an explicit ``--gpus`` option."""
    if gpus is None:
        return
    has_gpu_option = any(
        arg == "--gpus" or arg.startswith("--gpus=")
        for arg in sys.argv[1:]
    )
    if not has_gpu_option:
        sys.argv.extend(["--gpus", gpus])
