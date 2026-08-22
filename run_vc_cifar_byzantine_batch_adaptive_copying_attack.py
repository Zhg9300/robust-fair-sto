"""Run the CIFAR-10 V.C adaptive-copying matrix through batch_run.py."""

import batch_run
from batch_run_image_vc_common import CIFAR10, get_experiment_groups as _build
from batch_run_image_vc_common import use_default_gpus


def get_experiment_groups():
    return _build(CIFAR10, "adaptive_copying")


if __name__ == "__main__":
    batch_run.get_experiment_groups = get_experiment_groups
    use_default_gpus()
    batch_run.main()
