"""Run the Fashion-MNIST V.C hyperparameter and robust-aggregation matrix."""

from batch_run_image_vc_common import DatasetSpec, build_experiments as _build, run_cli


DATASET_SPEC = DatasetSpec(
    slug="fashion",
    display_name="Fashion-MNIST",
    module="MLP",
    rounds=2000,
    pathological_loader="DataLoader_fashion_pat",
    dirichlet_loader="DataLoader_fashion_dir",
)


def build_experiments(seeds=(1,)):
    return _build(DATASET_SPEC, seeds)


def main(argv=None):
    return run_cli(DATASET_SPEC, argv)


if __name__ == "__main__":
    raise SystemExit(main())
