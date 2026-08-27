# On the Tension Between Fairness and Robustness

This project is a Python framework for federated learning experiments focused on client-level fairness and Byzantine robustness. It provides image classification and structured linear-regression tasks, multiple federated algorithms, configurable attacks, and robust gradient aggregation.

## Installation

Create an isolated Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

For GPU experiments, install a PyTorch build compatible with the CUDA version
on the training host.

## Quick Start

Run a FashionMNIST classification experiment on CPU:

```bash
python run.py --device -1 \
  --module MLP --dataloader DataLoader_fashion_pat \
  --algorithm FedAvg --N 10 --C 1 --B full --R 50 --E 1
```

Run structured linear regression with qFFL and Krum:

```bash
python run.py --device -1 \
  --module LinearRegression \
  --dataloader DataLoader_linear_regression \
  --algorithm qFedAvg \
  --N 10 --C 1 --B 20 --R 50 --E 1 \
  --q 1 --qffl_update_rule objective_gradient \
  --qffl_loss_mode full --sgd_step True \
  --gradient_aggregator krum --gradient_aggregator_f 2 \
  --weight_decay 0 --decay 1
```

Run a sign-flip attack:

```bash
python run.py --device -1 \
  --module MLP --dataloader DataLoader_fashion_pat \
  --algorithm FedAvg --N 10 --C 1 --B full --R 50 --E 1 \
  --attack_mode sign_flip --dishonest_num 2 --attack_scale 5
```

Use `python run.py --help` for the complete command-line interface.

## Components

- Algorithms: `FedAvg`, `qFedAvg`, `AFL`, `FedFV`, `FedMGDA_plus`, `DRFL`,
  and `AdaFed`.
- Models: `CNN`, `MLP`, and `LinearRegression`.
- Datasets: CIFAR-10, FashionMNIST, and generated linear-regression data.
- Partitions: pathological (`*_pat`) and Dirichlet (`*_dir`) image partitions.
- Metrics: classification accuracy and loss; linear regression additionally
  reports honest-client loss statistics and distance metrics.

## Key Configuration

| Option | Meaning |
| --- | --- |
| `--device` | GPU index; use `-1` for CPU |
| `--N` | Total number of clients |
| `--C` | Fraction of clients selected per round |
| `--B` | Positive batch size or `full` |
| `--R` | Communication rounds |
| `--E` | Local epochs per round |
| `--lr` | Learning rate |
| `--decay` | Round-wise learning-rate decay |
| `--partition_seed` | Seed used only for client data partitioning |
| `--evaluation_excluded_ids` | Client IDs omitted from evaluation |

### Full-client updates

Set `--B full` to compute one client update from all local samples. Use
`--micro_batch_size SIZE` to reduce peak activation memory while retaining one
logical full-client update. Strict micro-batch accumulation is rejected for
BatchNorm models because chunking changes batch statistics.

Federated tensors reside on the model device by default. Set
`--data_device cpu` to keep them in host memory and transfer only the active
batch. Client partitions are cached as compact indices and reproduced with
`--partition_seed`.

### Structured linear regression

The linear-regression task supports an eight-client honest baseline and a
ten-client configuration with two Byzantine slots. Its main options are:

| Option | Default |
| --- | ---: |
| `--linear_noise_std` | `1e-3` |
| `--linear_delta` | `1` |
| `--linear_train_batches` | `100` |

Use `--B 20` for stochastic mini-batches or `--B full` for full local data.
The output includes honest-client loss mean, variance, minimum, maximum,
average-loss gap, and distance to the aligned model.

## Byzantine Attacks

Select an attack with `--attack_mode` and configure Byzantine clients with
`--dishonest_num` or explicit `--byzantine_ids`.

Supported modes include:

- Gradient attacks: `sign_flip`, `large_norm`, `sybil_direction`, `alie`,
  `gaussian`, and `ipm`.
- Label attacks: `label_random_flip`, `label_cyclic_flip`, and
  `label_targeted_flip`.
- Loss attacks: `loss_inflation`, `loss_deflation`, and `loss_ranking`.
- Combined attacks: `high_loss_malicious_gradient` and
  `low_loss_malicious_gradient`.
- Impersonation attacks: `adaptive_copying` and `multi_decoy_minority`.

Common controls are `--attack_scale`, `--loss_bias`, `--attack_seed`,
`--attack_start_round`, and `--attack_end_round`. Copying attacks additionally
require `--attack_target_clients` and at least one of `--copy_gradient` or
`--copy_loss`.

## Robust Aggregation

Choose the server-side rule with `--gradient_aggregator`:

| Value | Aggregator |
| --- | --- |
| `mean` | Sample-count-weighted mean |
| `cwtm` | Coordinate-wise trimmed mean |
| `cwm` | Coordinate-wise median |
| `median` | Geometric median |
| `faba` | Iterative farthest-update removal |
| `krum` | Krum selection |
| `mda` | Minimum-diameter averaging |
| `centered_clipping` | Stateful centered clipping |
| `nbs` | Norm-based screening |

Set `--gradient_aggregator_f` to the assumed Byzantine-client count. When it
is omitted, the active Byzantine count is used. Some aggregators impose
additional minimum-client constraints and reject invalid configurations.

### qFFL with NBS

NBS can be combined with qFFL without adding a separate algorithm:

```bash
python run.py --algorithm qFedAvg \
  --qffl_update_rule objective_gradient \
  --gradient_aggregator nbs --gradient_aggregator_f 2 \
  --q 1 --C 1 --E 1 --sgd_step False
```

The direct-objective qFFL mode requires full client participation and one local
epoch. Replace `nbs` with `mean` to use the same update without screening.

## Generated Files

The following local paths are excluded from Git:

- `components/data/`: downloaded datasets
- `components/pool/`: cached client partitions
- `results/`: experiment logs and outputs
- `batch_results/`: batch experiment summaries

They are created again when experiments run on a new machine.
