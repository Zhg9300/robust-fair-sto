"""Run the Section V.A linear-regression matrix and collect plotting tables.

The attack-free FedAvg/qFFL mean baselines train on N=H=8 workers.  The
adaptive-copying experiments use n=10, h=8, b=2 and apply every robust aggregator
listed in Section V.A of journal-v8.pdf.
"""

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SEEDS = tuple(range(1, 11))
PAPER_ROBUST_AGGREGATORS = (
    "cwm",
    "cwtm",
    "median",
    "krum",
    "mda",
    "faba",
    "nbs",
)
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
INITIALIZATION = "honest_average_optimum"
SUMMARY_FIELDS = (
    "experiment",
    "method",
    "scenario",
    "seed",
    "algorithm",
    "gradient_aggregator",
    "gradient_aggregator_f",
    "worker_count",
    "honest_worker_count",
    "byzantine_worker_count",
    "attack_mode",
    "rounds",
    "batch_size",
    "local_samples",
    "learning_rate",
    "decay",
    "q",
    "noise_std",
    "delta",
    "initialization",
    "initial_distance_to_v",
    "honest_loss_mean",
    "honest_loss_variance",
    "honest_loss_min",
    "honest_loss_max",
    "honest_average_loss_gap",
    "distance_to_v",
    "honest_loss_list",
    "status",
    "return_code",
    "duration_seconds",
    "finished_at",
    "command",
    "error",
)
TRAJECTORY_FIELDS = (
    "experiment",
    "method",
    "scenario",
    "seed",
    "algorithm",
    "gradient_aggregator",
    "gradient_aggregator_f",
    "worker_count",
    "honest_worker_count",
    "byzantine_worker_count",
    "attack_mode",
    "round",
    "checkpoint_source",
    "batch_size",
    "local_samples",
    "learning_rate",
    "decay",
    "q",
    "noise_std",
    "delta",
    "initialization",
    "initial_distance_to_v",
    "honest_loss_mean",
    "honest_loss_variance",
    "honest_loss_min",
    "honest_loss_max",
    "honest_average_loss_gap",
    "distance_to_v",
    "honest_loss_list",
    "status",
)

AGGREGATOR_LABELS = {
    "cwm": "CWM",
    "cwtm": "CWTM",
    "median": "GM",
    "krum": "Krum",
    "mda": "MDA",
    "faba": "FABA",
    "nbs": "NBS",
}


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def parse_seeds(value):
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    if len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must not contain duplicates")
    return seeds


def parse_aggregators(value):
    aggregators = tuple(
        item.strip().lower() for item in value.split(",") if item.strip()
    )
    if not aggregators:
        raise argparse.ArgumentTypeError("at least one aggregator is required")
    unknown = sorted(set(aggregators).difference(PAPER_ROBUST_AGGREGATORS))
    if unknown:
        supported = ", ".join(PAPER_ROBUST_AGGREGATORS)
        raise argparse.ArgumentTypeError(
            f"unsupported V.A aggregator(s): {', '.join(unknown)}; choose from {supported}"
        )
    if len(aggregators) != len(set(aggregators)):
        raise argparse.ArgumentTypeError("aggregators must not contain duplicates")
    return aggregators


def read_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run and summarize the paper V.A linear-regression experiments."
    )
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=DEFAULT_SEEDS,
        help="comma-separated paired seeds; defaults to 1,...,10",
    )
    parser.add_argument("--rounds", type=positive_int, default=400)
    parser.add_argument(
        "--test-interval",
        type=positive_int,
        default=10,
        help="checkpoint interval written to the trajectory CSV",
    )
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--decay", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=1e-3)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--train-batches", type=positive_int, default=100)
    parser.add_argument(
        "--aggregators",
        type=parse_aggregators,
        default=PAPER_ROBUST_AGGREGATORS,
        help="comma-separated subset of the V.A robust aggregators",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="final-summary CSV path; defaults to a timestamped file under results/",
    )
    parser.add_argument(
        "--trajectory-output",
        type=Path,
        default=None,
        help="long-format trajectory CSV path; derived from --output by default",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print all commands without executing them",
    )
    args = parser.parse_args(argv)
    for name in ("lr", "decay", "noise_std", "delta"):
        value = getattr(args, name)
        if not math.isfinite(value):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    if args.lr <= 0.0:
        parser.error("--lr must be positive")
    if args.decay <= 0.0:
        parser.error("--decay must be positive")
    if args.noise_std <= 0.0:
        parser.error("--noise-std must be positive")
    if args.delta <= 0.0:
        parser.error("--delta must be positive")
    if args.test_interval > args.rounds:
        parser.error("--test-interval must not exceed --rounds")
    return args


def build_experiments(seeds, aggregators=PAPER_ROBUST_AGGREGATORS):
    experiments = []
    for seed in seeds:
        for algorithm in ("FedAvg", "qFedAvg"):
            method = "FedAvg (clean)" if algorithm == "FedAvg" else "qFFL (clean)"
            experiments.append({
                "experiment": f"clean_{algorithm}_mean_seed{seed}",
                "method": method,
                "scenario": "honest_only_no_attack",
                "seed": seed,
                "algorithm": algorithm,
                "gradient_aggregator": "mean",
                "gradient_aggregator_f": 0,
                "worker_count": 8,
                "honest_worker_count": 8,
                "byzantine_worker_count": 0,
                "attack_mode": "None",
            })
        for aggregator in aggregators:
            experiments.append({
                "experiment": f"adaptive_copying_qFedAvg_{aggregator}_seed{seed}",
                "method": f"qFFL+{AGGREGATOR_LABELS[aggregator]} (adaptive copying)",
                "scenario": "adaptive_copying_attack",
                "seed": seed,
                "algorithm": "qFedAvg",
                "gradient_aggregator": aggregator,
                "gradient_aggregator_f": 2,
                "worker_count": 10,
                "honest_worker_count": 8,
                "byzantine_worker_count": 2,
                "attack_mode": "adaptive_copying",
            })
    return experiments


def build_command(experiment, args):
    command = [
        sys.executable,
        "run.py",
        "--seed", str(experiment["seed"]),
        "--partition_seed", str(experiment["seed"]),
        "--device", str(args.device),
        "--module", "LinearRegression",
        "--dataloader", "DataLoader_linear_regression",
        "--algorithm", experiment["algorithm"],
        "--N", str(experiment["worker_count"]),
        "--C", "1",
        "--B", "20",
        "--R", str(args.rounds),
        "--E", "1",
        "--test_interval", str(args.test_interval),
        "--sgd_step", "True",
        "--data_device", "cpu",
        "--lr", str(args.lr),
        "--decay", str(args.decay),
        "--momentum", "0",
        "--weight_decay", "0",
        "--linear_noise_std", str(args.noise_std),
        "--linear_delta", str(args.delta),
        "--linear_train_batches", str(args.train_batches),
        "--gradient_aggregator", experiment["gradient_aggregator"],
        "--gradient_aggregator_f", str(experiment["gradient_aggregator_f"]),
        "--attack_mode", experiment["attack_mode"],
        "--dishonest_num", str(experiment["byzantine_worker_count"]),
    ]
    if experiment["algorithm"] == "qFedAvg":
        command.extend([
            "--q", "1",
            "--qffl_update_rule", "objective_gradient",
        ])
    if experiment["attack_mode"] == "adaptive_copying":
        command.extend([
            "--byzantine_ids", "8,9",
            "--attack_target_clients", "0",
            "--copy_gradient", "True",
            "--copy_loss", "False",
            "--evaluation_excluded_ids", "8,9",
        ])
    return command


def _last_float(text, label):
    matches = re.findall(
        re.escape(label) + r"\s*(" + FLOAT_PATTERN + r")",
        text,
    )
    return float(matches[-1]) if matches else None


def parse_output(text):
    result = {
        "honest_loss_mean": None,
        "honest_loss_variance": None,
        "honest_loss_min": None,
        "honest_loss_max": None,
        "honest_average_loss_gap": None,
        "distance_to_v": None,
        "honest_loss_list": None,
    }
    stats_pattern = re.compile(
        r"Loss Average:\s*(" + FLOAT_PATTERN + r")\.\s*"
        r"Loss Variance:\s*(" + FLOAT_PATTERN + r")\.\s*"
        r"Loss Min:\s*(" + FLOAT_PATTERN + r")\.\s*"
        r"Loss Max:\s*(" + FLOAT_PATTERN + r")"
    )
    stats = list(stats_pattern.finditer(text))
    if stats:
        values = tuple(float(value) for value in stats[-1].groups())
        result.update({
            "honest_loss_mean": values[0],
            "honest_loss_variance": values[1],
            "honest_loss_min": values[2],
            "honest_loss_max": values[3],
        })

    loss_lists = re.findall(r"Test Loss List:\s*\[(.*?)\]", text)
    if loss_lists:
        losses = [float(value) for value in re.findall(FLOAT_PATTERN, loss_lists[-1])]
        result["honest_loss_list"] = json.dumps(losses)

    exact_variance = _last_float(text, "Honest Loss Variance V_H(w):")
    if exact_variance is not None:
        result["honest_loss_variance"] = exact_variance
    result["honest_average_loss_gap"] = _last_float(
        text,
        "Honest Average-Loss Gap F_H(w)-F_H(w_H*):",
    )
    result["distance_to_v"] = _last_float(text, "Distance ||w-v||_2:")
    return result


def parse_trajectory_output(text):
    """Return reported checkpoint metrics in chronological round order."""
    markers = list(re.finditer(r"(?m)^round\s+(\d+)\s*$", text))
    checkpoints = {}
    for index, marker in enumerate(markers):
        block_end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        metrics = parse_output(text[marker.end():block_end])
        if metrics["honest_loss_variance"] is None:
            continue
        round_id = int(marker.group(1))
        checkpoints[round_id] = {
            "round": round_id,
            "checkpoint_source": "reported",
            **metrics,
        }
    return [checkpoints[round_id] for round_id in sorted(checkpoints)]


def analytic_initial_metrics(noise_std, delta):
    """Metrics at the shared w_H* initialization, evaluated on H."""
    honest_count = 8
    noise_floor = 0.5 * 20 * float(noise_std) ** 2
    initial_coordinate = float(delta) / honest_count
    aligned_loss = noise_floor + 0.5 * initial_coordinate ** 2
    exceptional_loss = (
        noise_floor + 0.5 * (initial_coordinate - float(delta)) ** 2
    )
    losses = [aligned_loss] * (honest_count - 1) + [exceptional_loss]
    loss_mean = sum(losses) / honest_count
    return {
        "honest_loss_mean": loss_mean,
        "honest_loss_variance": sum(
            (loss - loss_mean) ** 2 for loss in losses
        ) / honest_count,
        "honest_loss_min": aligned_loss,
        "honest_loss_max": exceptional_loss,
        "honest_average_loss_gap": 0.0,
        "distance_to_v": initial_coordinate,
        "honest_loss_list": json.dumps(losses),
    }


def _run_configuration(experiment, args):
    return {
        **experiment,
        "batch_size": 20,
        "local_samples": 20 * args.train_batches,
        "learning_rate": args.lr,
        "decay": args.decay,
        "q": 1 if experiment["algorithm"] == "qFedAvg" else "",
        "noise_std": args.noise_std,
        "delta": args.delta,
        "initialization": INITIALIZATION,
        "initial_distance_to_v": args.delta / 8.0,
    }


def build_trajectory_rows(experiment, args, stdout):
    configuration = _run_configuration(experiment, args)
    initial = {
        **configuration,
        "round": 0,
        "checkpoint_source": "analytic_initial",
        **analytic_initial_metrics(args.noise_std, args.delta),
        "status": "OK",
    }
    rows = [initial]
    for checkpoint in parse_trajectory_output(stdout):
        rows.append({
            **configuration,
            **checkpoint,
            "status": "OK",
        })
    return rows


def write_table(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def subprocess_environment():
    environment = os.environ.copy()
    conda_prefix = environment.get("CONDA_PREFIX")
    if conda_prefix:
        conda_library = str(Path(conda_prefix) / "lib")
        current = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = (
            conda_library if not current else conda_library + os.pathsep + current
        )
    return environment


def run_experiment(experiment, args):
    command = build_command(experiment, args)
    started = time.perf_counter()
    stdout = ""
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=subprocess_environment(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        stdout = completed.stdout
        metrics = parse_output(stdout)
        parse_ok = metrics["honest_loss_variance"] is not None
        status = "OK" if completed.returncode == 0 and parse_ok else "FAILED"
        error = completed.stderr.strip()
        if completed.returncode == 0 and not parse_ok:
            error = "run succeeded but final V.A metrics were not found in stdout"
        return_code = completed.returncode
    except OSError as exc:
        metrics = parse_output("")
        status = "FAILED"
        error = str(exc)
        return_code = -1

    row = _run_configuration(experiment, args)
    row.update({
        "rounds": args.rounds,
        **metrics,
        "status": status,
        "return_code": return_code,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": shlex.join(command),
        "error": error,
    })
    trajectory_rows = (
        build_trajectory_rows(experiment, args, stdout)
        if status == "OK"
        else []
    )
    return row, trajectory_rows


def _workspace_path(path):
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_output_paths(args):
    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = PROJECT_ROOT / "results" / f"linear_regression_batch_{stamp}"
        summary = base.with_name(base.name + "_summary.csv")
    else:
        summary = _workspace_path(args.output)
        base_name = summary.stem
        if base_name.endswith("_summary"):
            base_name = base_name[:-len("_summary")]
        base = summary.with_name(base_name)

    trajectory = (
        _workspace_path(args.trajectory_output)
        if args.trajectory_output is not None
        else base.with_name(base.name + "_trajectory.csv")
    )
    if summary == trajectory:
        raise ValueError("summary and trajectory output paths must be different")
    return summary, trajectory


def main(argv=None):
    args = read_args(argv)
    experiments = build_experiments(args.seeds, args.aggregators)
    commands = [build_command(experiment, args) for experiment in experiments]
    if args.dry_run:
        for index, command in enumerate(commands, start=1):
            print(f"[{index}/{len(commands)}] {shlex.join(command)}")
        return 0

    summary_output, trajectory_output = resolve_output_paths(args)

    summary_rows = []
    trajectory_rows = []
    write_table(summary_output, summary_rows, SUMMARY_FIELDS)
    write_table(trajectory_output, trajectory_rows, TRAJECTORY_FIELDS)
    print(f"Running {len(experiments)} experiments")
    print(f"Summary table: {summary_output}")
    print(f"Trajectory table: {trajectory_output}")
    for index, experiment in enumerate(experiments, start=1):
        print(f"[{index}/{len(experiments)}] {experiment['experiment']}")
        row, checkpoints = run_experiment(experiment, args)
        summary_rows.append(row)
        trajectory_rows.extend(checkpoints)
        write_table(summary_output, summary_rows, SUMMARY_FIELDS)
        write_table(trajectory_output, trajectory_rows, TRAJECTORY_FIELDS)
        print(
            f"  {row['status']} variance={row['honest_loss_variance']} "
            f"gap={row['honest_average_loss_gap']} distance={row['distance_to_v']} "
            f"checkpoints={len(checkpoints)}"
        )
    failures = sum(row["status"] != "OK" for row in summary_rows)
    print(
        f"Finished: {len(summary_rows) - failures} succeeded, "
        f"{failures} failed, {len(trajectory_rows)} trajectory rows"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
