# On the Tension Between Fairness and Robustness

Code for federated learning experiments on client-level fairness and Byzantine
robustness, with image classification and structured linear regression.

## Installation

Install the dependencies in a Python environment:

```bash
pip install -r requirements.txt
```

For GPU experiments, use a PyTorch build compatible with your CUDA version.

## Quick Start

Run a FashionMNIST experiment on CPU:

```bash
python run.py --device -1 \
  --module MLP --dataloader DataLoader_fashion_pat \
  --algorithm FedAvg --N 10 --C 1 --B full --R 50 --E 1
```

See `python run.py --help` for available options.

## Supported Methods

- Algorithms: FedAvg, q-FFL (`qFedAvg`), AFL, FedFV, FedMGDA+, DRFL, and AdaFed.
- Models: CNN, MLP, and linear regression.
- Datasets: CIFAR-10, FashionMNIST, and synthetic linear-regression data.
- Image partitions: pathological and Dirichlet.

Select an aggregation rule with `--gradient_aggregator`:

| Value | Aggregator |
| --- | --- |
| `mean` | Mean |
| `cwm` | Coordinate-wise median |
| `cwtm` | Coordinate-wise trimmed mean |
| `median` | Geometric median (GM) |
| `krum` | Krum |
| `mda` | Minimum-diameter averaging |
| `faba` | FABA |
| `nbs` | Norm-based screening |
| `centered_clipping` | Centered clipping |

Attacks include `sign_flip`, `gaussian`, and `mimic_attack`, selected with
`--attack_mode`. Set the assumed Byzantine count with `--gradient_aggregator_f`.

## Linear Regression Experiments

Three entries share the same 16 configurations per paired seed: FedAvg and
q-FFL baselines, q-FFL with seven robust aggregators without attacks, and the
same robust combinations under mimic attack.

Attack-free runs use eight honest clients. Attacked runs add two Byzantine
clients that copy client 0's weighted message. Robust aggregators use `f=2`
in both scenarios. All entries default to 1000 rounds and dimension/batch size 20.

| Entry | Data / q-FFL loss | Step / projection | Noise std. | Seeds |
| --- | --- | --- | --- | --- |
| `batch_run_linear_regression.py` | Fixed / empirical | Constant / none | 0.01 | 1-30 |
| `batch_run_linear_regression_theory.py` | Fixed / empirical | Diminishing / ball | 0.01 | 1-30 |
| `batch_run_linear_regression_theory_online.py` | Online / population | Diminishing / ball | 0.001 | 1-5 |

Run the experiments:

```bash
python batch_run_linear_regression.py \
  --output results/linear_regression_summary.csv
python batch_run_linear_regression_theory.py \
  --output results/linear_regression_theory_summary.csv
python batch_run_linear_regression_theory_online.py \
  --output results/linear_regression_theory_online_summary.csv
```

Each entry writes summary and trajectory CSVs. Plot the trajectories:

```bash
python plot_linear_regression_panels.py \
  results/linear_regression_trajectory.csv
python plot_linear_regression_panels.py \
  results/linear_regression_theory_trajectory.csv
python plot_linear_regression_panels.py \
  results/linear_regression_theory_online_trajectory.csv
```

For a finer view of rounds 800-1000:

```bash
python plot_linear_regression_panels_zoom.py \
  results/linear_regression_theory_online_trajectory.csv
```

The zoomed insets show each method's difference from CWM.
