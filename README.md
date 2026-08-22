# On the Tension Between Fairness and Robustness

## Full-client gradient mode

Use `--B full` when every client must compute one update from all of its local
samples. The client size is discovered automatically; values such as 8300 no
longer need to be estimated manually.

```bash
python run.py --module MLP --dataloader DataLoader_fashion_dir \
  --Diralpha 0.5 --N 10 --C 1 --B full --R 50 --E 1
```

`--micro_batch_size 1024` (or another positive size) can reduce peak activation
memory while retaining one optimizer step per client epoch. Its default is `0`,
which processes the whole client at once. Strict micro-batch accumulation is
rejected for BatchNorm models because chunking changes their batch statistics.

Federated tensors reside on the model device by default. Use
`--data_device cpu` to keep them in host memory and transfer only the current
batch. Client partitions use the independent `--partition_seed` (default `1`)
and are cached as compact indices, so changing `B`, `micro_batch_size`, or the
data device does not duplicate a partition cache. For unbalanced pathological
partitions, `--min_client_samples auto` uses 10% of the mean client sample count.

On a training host, compare the default whole-client path with 1024/2048
micro-batches using:

```bash
python benchmark_full_batch.py --device 0 --sizes 0,1024,2048
```

The benchmark reports mean gradient time and peak allocated CUDA memory for the
largest FashionMNIST Dirichlet-0.5 client.

FairAndRobust is a Python research codebase for federated learning experiments, with a focus on fairness and robustness across clients. It includes image-classification experiments and the structured linear-regression experiment from Section V.A of `journal-v8.pdf`.

## Setup

Create an isolated Python environment, then install the project dependencies:

```bash
pip install -r requirements.txt
```

For GPU experiments, install the PyTorch build that matches the CUDA version on your machine or server.

## Run Experiments

Current public components are:

- Algorithms: `FedAvg`, `qFedAvg`, `AFL`, `FedFV`, `FedMGDA_plus`, `DRFL`, and `AdaFed`.
- Models: `CNN`, `MLP`, and `LinearRegression`.
- Data loaders: `DataLoader_cifar10_pat`, `DataLoader_cifar10_dir`,
  `DataLoader_fashion_pat`, `DataLoader_fashion_dir`, and
  `DataLoader_linear_regression`.
- Classification metric: `Correct`. Linear regression reports analytic
  honest-worker losses and does not create accuracy fields.

Run a classification experiment:

```bash
python run.py --algorithm FedAvg --dataloader DataLoader_cifar10_pat
```

### Section V.A linear regression

The V.A loader is integrated into the normal
`run.py -> DataLoader -> Module -> Algorithm` path. It supports the paper's
`N=H=8` honest-only mean baseline and the `n=10,h=8,b=2` defended/attacked
setting. Both use `d=K=20`, `v=0`, `theta=e1`, and `X=I_20` in every worker
batch. Workers 0--6 and, when present, the benign distributions in slots 8--9
use `w_dagger=0`; exceptional honest worker 7 uses `w_dagger=delta*e1`.
The `N=8` baseline trains and evaluates all eight workers. The `N=10` setting
excludes slots 8--9 from evaluation. All evaluation losses are analytic
population losses from equation (27).

Every linear-regression run starts from the honest average-loss optimum
`w_H*=v+(delta/h)*theta` (distance `delta/h` from `v`). This shared nonzero
initialization makes the clean fairness-aware movement and the attacked
trajectory back toward the aligned model visible; image-classification model
initialization is unchanged.

The paper does not specify all numerical values needed to instantiate this
finite experiment. The project-supplied defaults are `linear_noise_std=1e-3`,
`linear_delta=1`, and `linear_train_batches=100`; they can be overridden with
the corresponding CLI options. Data and noise are reproduced by
`--partition_seed`. The training loss is `0.5*K*MSE(mean)`, matching
`0.5*||Xw-y||_2^2` for each theorem-sized batch. `--B 20` enables stochastic
mini-batch gradients, while `--B full --sgd_step False` enables a full local
gradient.

A one-round stochastic qFFL/Krum run is:

```bash
python run.py --device -1 \
  --module LinearRegression \
  --dataloader DataLoader_linear_regression \
  --algorithm qFedAvg \
  --N 10 --C 1 --B 20 --R 1 --E 1 \
  --q 1 --qffl_update_rule objective_gradient \
  --qffl_loss_mode full --sgd_step True \
  --gradient_aggregator krum --gradient_aggregator_f 2 \
  --weight_decay 0 --decay 1 --test_interval 1
```

For the adaptive copying attack, append:

```bash
--attack_mode adaptive_copying --dishonest_num 2 \
--attack_target_clients 0 --copy_gradient True --copy_loss False
```

Use `--gradient_aggregator mda` for MDA. The linear-regression output reports
the honest loss list, its mean and population variance `V_H(w)`,
`F_H(w)-F_H(w_H*)`, and `||w-v||_2`. Equation (28)'s step-size schedule, the
projection set, and theorem-result decisions are not part of this implementation
stage; use the existing `lr` and `decay` options.

The compact paper-oriented batch runner executes FedAvg and qFFL with mean
aggregation on the honest-only `N=8` set, plus qFFL under `adaptive_copying` on `N=10`
with CWM, CWTM, geometric median, Krum, MDA, FABA, and NBS:

```bash
python batch_run_linear_regression.py
```

The current defaults are built in: paired seeds 1--10, 400 communication rounds,
and a checkpoint every 10 rounds. The runner incrementally writes two files
under `results/`: a `*_summary.csv` with one row per completed run and a
long-format `*_trajectory.csv` with one row per method, seed, and checkpoint.
Both tables record the initialization and its distance from `v`; the trajectory
also contains the exact analytic round-0 metrics. Override the
defaults with options such as `--seeds 1,2,3`, `--test-interval 5`, or
`--aggregators krum,mda`. `--dry-run` prints the full experiment matrix without
starting training. Centered Clipping remains available through `run.py` but is
not included here because it is not in the V.A aggregator list.

Generate multi-seed trajectory panels and final-result panels from the long
table:

```bash
python plot_linear_regression_panels.py results/linear_regression_batch_TIMESTAMP_trajectory.csv
```

The plotting script separates the clean baselines from adaptive-copying runs.
Clean panels show the original metrics; attack panels show log-scale absolute
deviations from the values at `v`. Each attacked trajectory panel includes a
linear-scale inset over the last three checkpoints, where the aggregator
differences are largest; inset curves show method means without confidence-band
occlusion. Thin trajectories and final scatter points show individual paired
seeds, while the bold means use two-sided 95% Student-t confidence intervals.
PNG and vector PDF versions are written beside the CSV. Use `--no-pdf` when only
PNG output is needed.

`--decay` controls the round-wise learning-rate schedule, while
`--weight_decay` controls SGD L2 regularization. Classification runs default to
`--weight_decay 5e-4`; pass `--weight_decay 0` explicitly for
paper-aligned FedMGDA+ comparisons. The selected value is included in result
names and batch-result workbooks so runs with different regularization settings
are not mixed.

Run a FashionMNIST Byzantine attack experiment:

```bash
python run.py --device -1 --module MLP --dataloader DataLoader_fashion_pat --algorithm FedAvg --N 10 --NC 1 --B full --R 1 --E 1 --attack_mode sign_flip --dishonest_num 1 --attack_scale 5
```

### Byzantine Attack Options

Common attack modes:

- `sign_flip`: flip Byzantine gradients; a standard baseline.
- `large_norm`: scale Byzantine gradients; requires `--attack_scale > 1`.
- `sybil_direction`: make online Byzantine clients report the same malicious direction.
- `alie`: make Byzantine clients report the shared ALIE gradient estimated from online honest gradients.
- `gaussian`: report independent seeded Gaussian directions rescaled relative to
  the online honest-gradient norms.
- `ipm`: report the negative scaled mean of the online honest gradients.
- `label_random_flip`: poison Byzantine training labels with a seeded random derangement.
- `label_cyclic_flip`: poison Byzantine training labels with `b -> (b + 1) mod 10`.
- `label_targeted_flip`: poison Byzantine training labels with `b -> 9 - b`.
- `loss_inflation`: report high losses.
- `loss_deflation`: report low losses.
- `loss_ranking`: report losses distributed over the high-loss rank range.
- `high_loss_malicious_gradient`: high reported loss plus sign-flip malicious gradient.
- `low_loss_malicious_gradient`: low reported loss plus sign-flip malicious gradient.
- `adaptive_copying`: all Byzantine clients copy one honest target.
- `multi_decoy_minority`: Byzantine clients cycle through multiple honest targets.

Common Byzantine client settings:

```bash
--dishonest_num 1
--dishonest_num 2
--dishonest_num 3
--byzantine_ids 8,9
```

For `--N 10 --dishonest_num 2`, the default Byzantine client ids are `[8, 9]`. Explicit `--byzantine_ids` binds attacks to client ids rather than sampled list positions.

Common attack intensity settings:

```bash
--attack_scale 1
--attack_scale 0.5
--attack_scale 5
--attack_scale 10
--attack_scale 50
--alie_z 0.5
--loss_bias 0
--loss_bias 0.5
--loss_bias 1.0
--loss_bias 2.0
```

`sign_flip` with `--attack_scale 1` is plain gradient reversal. `large_norm` requires a scale greater than 1. Loss attacks use `--loss_bias` to shift reported losses.

For `gaussian`, each Byzantine client receives an independent deterministic
Gaussian direction whose L2 norm is `attack_scale` times the mean L2 norm of
the online honest gradients. The direction is reproduced from `attack_seed`,
the communication round, and the client id. Gaussian attacks leave reported
losses unchanged and require a finite positive `attack_scale`.

ALIE uses the omniscient-attack assumption: in each gradient-bearing call it
estimates the coordinate-wise mean `mu` and population standard deviation
`sigma` from the online honest gradients, then makes every online Byzantine
client report `mu - z * sigma`. By default, with `m` online Byzantine clients
and `n` total online clients, it computes `s = floor(n / 2 + 1) - m` and
`z = Phi^-1((n - s) / n)`. Use `--alie_z`
to override that round-dependent value. The effective value is recorded in the
`alie_z` column of `byzantine_attack_log.csv`.

IPM also assumes an omniscient attacker. With online honest-gradient mean
`g_h`, every online Byzantine client reports `-epsilon * g_h`, where
`epsilon` is configured through `--attack_scale` and must be positive and
finite. IPM changes gradients/models but leaves reported losses unchanged.

For all label-poisoning modes, every training label of an active Byzantine
client is remapped before its loss and gradient are computed. Original training
data and all test labels remain unchanged, and labels become clean again outside
the attack window. `label_random_flip` uses `attack_seed` to create one fixed
derangement of classes `0..9` shared by every Byzantine client and every round.
The cyclic and targeted mappings are deterministic.

Current experiments use deterministic Byzantine client ids: explicit
`--byzantine_ids` when provided, otherwise the last `dishonest_num` resolved
client ids. `attack_seed` controls randomized attacks such as
`gaussian` and `label_random_flip`.

Common attack timing settings:

```bash
# attack for the whole run
--attack_start_round 1 --attack_end_round None

# start after warm-up
--attack_start_round 5 --attack_end_round None

# attack only a bounded round window
--attack_start_round 5 --attack_end_round 20
```

Impersonation attacks require target clients and at least one copied channel:

```bash
--attack_mode adaptive_copying --dishonest_num 2 --attack_target_clients 0 --copy_loss True --copy_gradient True
--attack_mode multi_decoy_minority --dishonest_num 3 --attack_target_clients 0,1,2 --copy_loss True --copy_gradient True
```

Common command examples:

```bash
# gradient sign-flip attack
python run.py --device -1 --module MLP --dataloader DataLoader_fashion_pat --algorithm FedAvg --N 10 --NC 1 --B full --R 50 --E 1 --attack_mode sign_flip --dishonest_num 1 --attack_scale 5

# ALIE with z computed from the online honest/Byzantine counts
python run.py --device -1 --module MLP --dataloader DataLoader_fashion_pat --algorithm FedAvg --N 10 --NC 1 --B full --R 50 --E 1 --attack_mode alie --dishonest_num 2

# ALIE with an explicitly selected z
python run.py --device -1 --module MLP --dataloader DataLoader_fashion_pat --algorithm FedAvg --N 10 --NC 1 --B full --R 50 --E 1 --attack_mode alie --dishonest_num 2 --alie_z 0.5

# IPM with epsilon=0.5
python run.py --device -1 --module MLP --dataloader DataLoader_fashion_pat --algorithm FedAvg --N 10 --NC 1 --B full --R 50 --E 1 --attack_mode ipm --dishonest_num 2 --attack_scale 0.5

# shared, reproducible random label derangement on Byzantine clients
python run.py --device -1 --module MLP --dataloader DataLoader_fashion_pat --algorithm FedAvg --N 10 --NC 1 --B full --R 50 --E 1 --attack_mode label_random_flip --dishonest_num 2 --attack_seed 1

# loss inflation attack
python run.py --device -1 --module MLP --dataloader DataLoader_fashion_pat --algorithm qFedAvg --N 10 --NC 1 --B full --R 50 --E 1 --attack_mode loss_inflation --dishonest_num 1 --loss_bias 1.0

# high-loss malicious-gradient attack
python run.py --device -1 --module MLP --dataloader DataLoader_fashion_pat --algorithm qFedAvg --N 10 --NC 1 --B full --R 50 --E 1 --attack_mode high_loss_malicious_gradient --dishonest_num 1 --attack_scale 5 --loss_bias 1.0
```

### Robust Aggregators

Server-side gradient aggregation is configured through `--gradient_aggregator`:

- `mean`: original sample-count-weighted FedAvg aggregation.
- `cwtm`: coordinate-wise trimmed mean.
- `cwm`: coordinate-wise median.
- `median`: Euclidean geometric median.
- `faba`: iterative farthest-update removal.
- `krum`: select the message with the smallest squared-distance score over its
  nearest `n-f-2` other messages.
- `mda`: exactly search size-`n-f` subsets, choose the one with minimum maximum
  pairwise distance, and average it.
- `centered_clipping`: stateful centered clipping.
- `nbs`: norm-based screening; remove the `f` largest L2 norms and average the rest.

For CWTM, FABA, Krum, MDA, and NBS, `--gradient_aggregator_f` is the assumed
Byzantine count. When it is omitted, the number of currently online Byzantine
clients is used. CWTM, FABA, and MDA require strictly more than twice that
count in online clients; Krum requires `n >= 2f+3`. NBS only requires
`0 <= f < online_client_count`. MDA is an exact combinatorial implementation
intended for the small worker counts in the V.A experiment. Krum breaks score
ties by worker-message order, and MDA breaks diameter ties lexicographically.

DRFL and AFL embed their fairness weights in the messages sent to this
aggregation interface. With `m` online clients, DRFL aggregates
`m * alpha_i * g_i`, where `alpha_i` is the normalized reported-loss weight;
AFL aggregates `m * lambda_i * g_i`, where `lambda_i` is its current dual
weight. The factor `m` makes `--gradient_aggregator mean` reproduce the original
weighted update, while CWTM, CWM, geometric median, FABA, Krum, MDA, and NBS
operate on the complete fairness-weighted vectors. Centered Clipping remains
available but is outside the DRFL/AFL validation matrix described here.

For the Section V.C adaptive-copying condition with DRFL or AFL, start the
attack in round 1 and copy both channels so the copied gradient and its
loss/dual-weight evolution remain synchronized:

```bash
--attack_mode adaptive_copying --attack_start_round 1 \
--dishonest_num 2 --byzantine_ids 8,9 --attack_target_clients 0 \
--copy_gradient True --copy_loss True
```

Use `--evaluation_excluded_ids` to keep attack and no-attack runs on the same
evaluation clients.

Example sign-flip runs:

```bash
python run.py --device -1 --module MLP --dataloader DataLoader_fashion_pat --algorithm FedAvg --gradient_aggregator cwtm --N 10 --C 1 --B full --R 50 --E 1 --attack_mode sign_flip --dishonest_num 2 --attack_scale 5
python run.py --device -1 --module MLP --dataloader DataLoader_fashion_pat --algorithm FedAvg --gradient_aggregator cwm --N 10 --C 1 --B full --R 50 --E 1 --attack_mode sign_flip --dishonest_num 2 --attack_scale 5
python run.py --device -1 --module MLP --dataloader DataLoader_fashion_pat --algorithm FedAvg --gradient_aggregator median --N 10 --C 1 --B full --R 50 --E 1 --attack_mode sign_flip --dishonest_num 2 --attack_scale 5
python run.py --device -1 --module MLP --dataloader DataLoader_fashion_pat --algorithm FedAvg --gradient_aggregator faba --N 10 --C 1 --B full --R 50 --E 1 --attack_mode sign_flip --dishonest_num 2 --attack_scale 5
python run.py --device -1 --module MLP --dataloader DataLoader_fashion_pat --algorithm FedAvg --gradient_aggregator centered_clipping --cc_tau 10 --cc_iterations 1 --N 10 --C 1 --B full --R 50 --E 1 --attack_mode sign_flip --dishonest_num 2 --attack_scale 5
```

### H-nobs as q-FFL plus NBS

H-nobs is implemented compositionally rather than as a separate algorithm.
Use qFedAvg's `objective_gradient` rule to form each honest client contribution
as `F_i^q * grad(F_i)`, then select NBS as the gradient aggregator:

```bash
python run.py --algorithm qFedAvg \
  --qffl_update_rule objective_gradient \
  --gradient_aggregator nbs \
  --gradient_aggregator_f 4 \
  --q 1 --C 1 --E 1 --sgd_step False
```

Here `gradient_aggregator_f` is the paper's screening count `f = beta * m`.
Set it explicitly when screening is desired in a no-attack run; otherwise the
automatic count is zero. Replacing `nbs` with `mean` gives the vanilla q-FFL
baseline under the same direct-objective update. The direct-objective mode
requires full client participation and `E=1`. With `sgd_step=False`, both the
loss and gradient use the complete local training set. With `sgd_step=True`,
the default `qffl_loss_mode=same_batch` computes the outer loss and stochastic
gradient from the same randomly selected mini-batch. Pass
`--qffl_loss_mode full` to instead compute the outer loss over the complete
local training set while retaining a random mini-batch gradient. The default
`qffl_update_rule=normalized` retains the original qFedAvg update with its
`sum(h_i)` denominator and rejects non-mean gradient aggregators explicitly.

For `objective_gradient`, attacks are injected after forming the complete
loss-weighted qFFL message; therefore copying attacks duplicate the final
transmitted message, and IPM estimates and reverses the mean q-FFL gradient in
H-nobs runs.
For label poisoning, the ordering is instead: remap Byzantine training labels,
compute `F_i` and `grad(F_i)`, form `F_i^q * grad(F_i)`, then apply NBS.

Suggested first attack sweep. Build valid combinations rather than a full Cartesian product: use `dishonest_num=0` only for `attack_mode=None`, and do not pair `large_norm` with `attack_scale=1`.

| Dimension | Values |
| --- | --- |
| `attack_mode` | `None`, `sign_flip`, `large_norm`, `alie`, `gaussian`, `ipm`, `label_random_flip`, `label_cyclic_flip`, `label_targeted_flip`, `loss_inflation`, `loss_deflation`, `high_loss_malicious_gradient` |
| `dishonest_num` | `1`, `2` |
| `attack_scale` | `0.5`, `1`, `5`, `10`, `50` |
| `alie_z` | omitted (automatic), `0.5` |
| `loss_bias` | `0`, `1` |
| `algorithm` | `FedAvg`, `qFedAvg`, `AFL`, `FedFV`, `FedMGDA_plus`, `DRFL`, `AdaFed` |

Run batch experiments with GPU scheduling:

```bash
python batch_run.py --gpus 0,1 --max-per-gpu 2
```

The Section V.C image-classification tuning matrices are split into one
runner per dataset and attack condition:

```bash
python run_vc_cifar_byzantine_batch_no_attack.py --gpus 0,1 --max-per-gpu 2
python run_vc_cifar_byzantine_batch_gaussian_attack.py --gpus 0,1 --max-per-gpu 2
python run_vc_cifar_byzantine_batch_sign_flip_attack.py --gpus 0,1 --max-per-gpu 2
python run_vc_cifar_byzantine_batch_adaptive_copying_attack.py --gpus 0,1 --max-per-gpu 2

python run_vc_fashion_byzantine_batch_no_attack.py --gpus 0,1 --max-per-gpu 2
python run_vc_fashion_byzantine_batch_gaussian_attack.py --gpus 0,1 --max-per-gpu 2
python run_vc_fashion_byzantine_batch_sign_flip_attack.py --gpus 0,1 --max-per-gpu 2
python run_vc_fashion_byzantine_batch_adaptive_copying_attack.py --gpus 0,1 --max-per-gpu 2
```

Each runner uses `batch_run.py` directly for grid expansion, GPU scheduling,
logging, and incremental result writing. Both partition schemes and the
configured qFedAvg/DRFL/AFL grids are retained. Attack runners sweep Mean plus
every validated non-Centered-Clipping robust aggregator, while no-attack
runners use Mean only. GPU defaults are `2,3` unless `--gpus` is supplied; use
`--dry-run` to inspect all generated commands without starting training.

## Generated Files

The following paths are generated locally and are intentionally not tracked by Git:

- `components/data/`: downloaded datasets
- `components/pool/`: cached client data pools
- `results/`: experiment logs and results
- `batch_results/`: batch runner summaries

These files will be regenerated on each machine as needed.
