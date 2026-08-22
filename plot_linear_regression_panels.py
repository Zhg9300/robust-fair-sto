"""Plot multi-seed Section V.A trajectory and final-result panels."""

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

# Matplotlib otherwise tries to use the user's read-only configuration directory
# in some Conda/container environments.  Keep its cache with the ignored results.
_MPL_CONFIG_DIR = Path(__file__).resolve().parent / "results" / ".matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


TITLE_FONT_SIZE = 18
TITLE_FONT_FAMILY = "Tinos"
AXIS_LABEL_FONT_SIZE = 18
TICK_FONT_SIZE = 13
LEGEND_FONT_SIZE = 13
plt.rcParams.update({
    "font.size": 13,
    "mathtext.fontset": "stix",
    "axes.titlesize": TITLE_FONT_SIZE,
    "axes.labelsize": AXIS_LABEL_FONT_SIZE,
    "xtick.labelsize": TICK_FONT_SIZE,
    "ytick.labelsize": TICK_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
})


METHOD_ORDER = (
    "FedAvg (clean)",
    "qFFL (clean)",
    "qFFL+CWM (adaptive copying)",
    "qFFL+CWTM (adaptive copying)",
    "qFFL+GM (adaptive copying)",
    "qFFL+Krum (adaptive copying)",
    "qFFL+MDA (adaptive copying)",
    "qFFL+FABA (adaptive copying)",
    "qFFL+NBS (adaptive copying)",
)
METHOD_COLORS = {
    "FedAvg (clean)": "#4D4D4D",
    "qFFL (clean)": "#0072B2",
    "qFFL+CWM (adaptive copying)": "#D55E00",
    "qFFL+CWTM (adaptive copying)": "#E69F00",
    "qFFL+GM (adaptive copying)": "#CC79A7",
    "qFFL+Krum (adaptive copying)": "#009E73",
    "qFFL+MDA (adaptive copying)": "#56B4E9",
    "qFFL+FABA (adaptive copying)": "#8C564B",
    "qFFL+NBS (adaptive copying)": "#9467BD",
}
METHOD_SHORT_LABELS = {
    "FedAvg (clean)": "FedAvg",
    "qFFL (clean)": "qFFL",
    "qFFL+CWM (adaptive copying)": "CWM",
    "qFFL+CWTM (adaptive copying)": "CWTM",
    "qFFL+GM (adaptive copying)": "GM",
    "qFFL+Krum (adaptive copying)": "Krum",
    "qFFL+MDA (adaptive copying)": "MDA",
    "qFFL+FABA (adaptive copying)": "FABA",
    "qFFL+NBS (adaptive copying)": "NBS",
}
METHOD_INSET_MARKERS = {
    "qFFL+CWM (adaptive copying)": "o",
    "qFFL+CWTM (adaptive copying)": "s",
    "qFFL+GM (adaptive copying)": "D",
    "qFFL+Krum (adaptive copying)": "^",
    "qFFL+MDA (adaptive copying)": "v",
    "qFFL+FABA (adaptive copying)": "P",
    "qFFL+NBS (adaptive copying)": "X",
}
# Two-sided 95% Student-t critical values for 1--30 degrees of freedom.
T_CRITICAL_95 = (
    math.nan,
    12.706204736, 4.302652730, 3.182446305, 2.776445105,
    2.570581836, 2.446911851, 2.364624252, 2.306004135,
    2.262157163, 2.228138852, 2.200985160, 2.178812830,
    2.160368656, 2.144786688, 2.131449546, 2.119905299,
    2.109815578, 2.100922040, 2.093024054, 2.085963447,
    2.079613845, 2.073873068, 2.068657610, 2.063898562,
    2.059538553, 2.055529439, 2.051830516, 2.048407142,
    2.045229642, 2.042272456,
)
RESIDUAL_FLOOR = 1e-10
REFERENCE_STYLES = {
    "honest_optimum": {
        "label": r"Honest optimum $w_{\mathcal{H}}^*$",
        "color": "#6A3D9A",
        "linestyle": ":",
        "marker": "D",
        "markerfacecolor": "white",
    },
    "adaptive_copying_limit": {
        "label": r"Adaptive-copying limit $v$",
        "color": "#E66101",
        "linestyle": "--",
        "marker": "s",
        "markerfacecolor": "white",
    },
}
REFERENCE_MARKER_POSITIONS = np.linspace(0.15, 0.85, 3)
METRICS = (
    (
        "honest_loss_variance",
        r"Honest-loss variance $V_{\mathcal{H}}(w_t)$",
        r"$V_{\mathcal{H}}(w_t)$",
    ),
    (
        "honest_average_loss_gap",
        r"Average-loss gap",
        r"$F_{\mathcal{H}}(w_t)-F_{\mathcal{H}}(w_{\mathcal{H}}^*)$",
    ),
    (
        "distance_to_v",
        r"Distance to aligned model",
        r"$\|w_t-v\|_2$",
    ),
)
ATTACK_RESIDUAL_LABELS = {
    "honest_loss_variance": (
        r"$|V_{\mathcal{H}}(w_t)-V_{\mathcal{H}}(v)|"
        r"\;(\times10^{-3})$"
    ),
    "honest_average_loss_gap": (
        r"$|\Delta F_{\mathcal{H}}(w_t)-\Delta F_{\mathcal{H}}(v)|"
        r"\;(\times10^{-3})$"
    ),
    "distance_to_v": r"$\|w_t-v\|_2\;(\times10^{-1})$",
}
ATTACK_RESIDUAL_SCALES = {
    "honest_loss_variance": 1e3,
    "honest_average_loss_gap": 1e3,
    "distance_to_v": 1e1,
}
NUMERIC_FIELDS = {
    "seed": int,
    "round": int,
    "honest_worker_count": int,
    "delta": float,
    "initial_distance_to_v": float,
    "honest_loss_mean": float,
    "honest_loss_variance": float,
    "honest_loss_min": float,
    "honest_loss_max": float,
    "honest_average_loss_gap": float,
    "distance_to_v": float,
}


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def read_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Draw paper-ready multi-seed panels from V.A CSV tables."
    )
    parser.add_argument("trajectory_csv", type=Path)
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="final-summary CSV; inferred from the trajectory filename when omitted",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="figure prefix; defaults beside the trajectory CSV",
    )
    parser.add_argument("--dpi", type=positive_int, default=300)
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="write PNG only instead of PNG and PDF",
    )
    return parser.parse_args(argv)


def _fallback_method(row):
    if row.get("method"):
        return row["method"]
    if row.get("scenario") == "honest_only_no_attack":
        return "FedAvg (clean)" if row.get("algorithm") == "FedAvg" else "qFFL (clean)"
    labels = {
        "cwm": "CWM",
        "cwtm": "CWTM",
        "median": "GM",
        "krum": "Krum",
        "mda": "MDA",
        "faba": "FABA",
        "nbs": "NBS",
    }
    aggregator = row.get("gradient_aggregator", "unknown")
    return f"qFFL+{labels.get(aggregator, aggregator)} (adaptive copying)"


def load_rows(path, require_round=False):
    with path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = []
    for source in source_rows:
        if source.get("status", "OK") != "OK":
            continue
        row = dict(source)
        row["method"] = _fallback_method(row)
        try:
            for field, converter in NUMERIC_FIELDS.items():
                if field in row and row[field] != "":
                    row[field] = converter(row[field])
            if require_round and not isinstance(row.get("round"), int):
                continue
            if any(not math.isfinite(float(row[key])) for key, _, _ in METRICS):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        rows.append(row)
    if not rows:
        raise ValueError(f"No usable OK rows found in {path}")
    return rows


def ordered_methods(rows):
    present = {row["method"] for row in rows}
    ordered = [method for method in METHOD_ORDER if method in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def mean_and_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if values.size < 2:
        return mean, 0.0
    standard_error = float(values.std(ddof=1) / np.sqrt(values.size))
    degrees_of_freedom = values.size - 1
    critical_value = (
        T_CRITICAL_95[degrees_of_freedom]
        if degrees_of_freedom < len(T_CRITICAL_95)
        else 1.96
    )
    return mean, critical_value * standard_error


def aggregate_trajectories(rows, metric, transform=None):
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        identity = (row["method"], row["seed"], row["round"])
        if identity in seen:
            raise ValueError(
                "Duplicate trajectory key "
                f"(method={identity[0]!r}, seed={identity[1]}, round={identity[2]})."
            )
        seen.add(identity)
        value = row[metric]
        grouped[(row["method"], row["round"])].append(
            transform(value) if transform is not None else value
        )

    result = defaultdict(list)
    for (method, round_id), values in grouped.items():
        mean, ci = mean_and_ci(values)
        result[method].append((round_id, mean, ci, len(values)))
    for method in result:
        result[method].sort(key=lambda item: item[0])
    return result


def final_rows_from_trajectory(rows):
    final = {}
    for row in rows:
        key = (row["method"], row["seed"])
        if key not in final or row["round"] > final[key]["round"]:
            final[key] = row
    return list(final.values())


def _reference_values(rows):
    delta = float(rows[0].get("delta", 1.0))
    honest_count = int(rows[0].get("honest_worker_count", 8))
    difference_at_v = 0.5 * delta ** 2
    difference_at_optimum = delta ** 2 * (0.5 - 1.0 / honest_count)
    return {
        "honest_loss_variance": {
            "honest_optimum": (
                (honest_count - 1)
                / honest_count ** 2
                * difference_at_optimum ** 2
            ),
            "adaptive_copying_limit": (
                (honest_count - 1)
                / honest_count ** 2
                * difference_at_v ** 2
            ),
        },
        "honest_average_loss_gap": {
            "honest_optimum": 0.0,
            "adaptive_copying_limit": 0.5 * (delta / honest_count) ** 2,
        },
        "distance_to_v": {
            "honest_optimum": delta / honest_count,
            "adaptive_copying_limit": 0.0,
        },
    }


def _attack_reference_values(rows):
    references = _reference_values(rows)
    return {
        metric: values["adaptive_copying_limit"]
        for metric, values in references.items()
    }


def attack_residual(value, reference, floor=RESIDUAL_FLOOR):
    """Absolute deviation from the adaptive-copying limit on a log-safe scale."""
    return max(abs(float(value) - float(reference)), float(floor))


def scaled_attack_residual(value, reference, metric):
    """Scale an attack residual so its power of ten can live in the axis label."""
    return ATTACK_RESIDUAL_SCALES[metric] * attack_residual(value, reference)


def _is_clean_method(method):
    return method.endswith("(clean)")


def _method_groups(rows):
    methods = ordered_methods(rows)
    return (
        [method for method in methods if _is_clean_method(method)],
        [method for method in methods if not _is_clean_method(method)],
    )


def _seed_trajectories(rows, method, metric, transform=None):
    grouped = defaultdict(list)
    for row in rows:
        if row["method"] != method:
            continue
        value = row[metric]
        grouped[row["seed"]].append(
            (row["round"], transform(value) if transform is not None else value)
        )
    for seed in grouped:
        grouped[seed].sort(key=lambda item: item[0])
    return grouped


def _add_reference_lines(axis, metric, references):
    plotted_values = []
    for reference_name, style in REFERENCE_STYLES.items():
        value = references[metric][reference_name]
        if any(math.isclose(value, plotted) for plotted in plotted_values):
            continue
        axis.axhline(
            value,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.3,
            alpha=0.65,
            zorder=0,
        )
        axis.plot(
            REFERENCE_MARKER_POSITIONS,
            np.full_like(REFERENCE_MARKER_POSITIONS, value),
            transform=axis.get_yaxis_transform(),
            color=style["color"],
            linestyle="none",
            marker=style["marker"],
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor=style["color"],
            markeredgewidth=1.2,
            markersize=5.5,
            alpha=0.8,
            zorder=1,
        )
        plotted_values.append(value)


def _reference_legend_handles():
    return [
        Line2D(
            [0],
            [0],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.5,
            marker=style["marker"],
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor=style["color"],
            markeredgewidth=1.2,
            markersize=6.5,
            label=style["label"],
        )
        for style in REFERENCE_STYLES.values()
    ]


def _add_reference_legend(figure):
    figure.legend(
        handles=_reference_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(REFERENCE_STYLES),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=3.0,
        columnspacing=2.5,
    )


def _plot_trajectory_group(
    axis,
    rows,
    methods,
    metric,
    transform=None,
    allow_negative=False,
):
    aggregated = aggregate_trajectories(rows, metric, transform=transform)
    for method in methods:
        color = METHOD_COLORS.get(method, "#333333")
        for points in _seed_trajectories(
            rows,
            method,
            metric,
            transform=transform,
        ).values():
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color=color,
                linewidth=0.65,
                alpha=0.12,
                zorder=1,
            )

        points = aggregated.get(method, [])
        if not points:
            continue
        rounds = np.asarray([point[0] for point in points])
        means = np.asarray([point[1] for point in points])
        cis = np.asarray([point[2] for point in points])
        axis.plot(
            rounds,
            means,
            label=METHOD_SHORT_LABELS.get(method, method),
            color=color,
            linewidth=2.2,
            alpha=0.98,
            zorder=3,
        )
        lower = means - cis
        if not allow_negative:
            lower = np.maximum(
                lower,
                RESIDUAL_FLOOR if transform else 0.0,
            )
        axis.fill_between(
            rounds,
            lower,
            means + cis,
            color=color,
            alpha=0.16,
            linewidth=0,
            zorder=2,
        )


def _add_late_round_inset(axis, rows, methods, metric, transform=None):
    """Add a linear-scale mean-trajectory zoom over the last three checkpoints."""
    aggregated = aggregate_trajectories(rows, metric, transform=transform)
    available_rounds = sorted({
        point[0]
        for method in methods
        for point in aggregated.get(method, [])
    })
    if len(available_rounds) < 2:
        return
    zoom_rounds = available_rounds[-3:]
    zoom_start, zoom_end = zoom_rounds[0], zoom_rounds[-1]
    zoom_axis = inset_axes(
        axis,
        width="49%",
        height="45%",
        loc="upper right",
        borderpad=1.0,
    )
    plotted_values = []
    for method in methods:
        points = [
            point for point in aggregated.get(method, [])
            if point[0] >= zoom_start
        ]
        if not points:
            continue
        rounds = [point[0] for point in points]
        means = [point[1] for point in points]
        plotted_values.extend(means)
        color = METHOD_COLORS.get(method, "#333333")
        zoom_axis.plot(
            rounds,
            means,
            color=color,
            linewidth=1.5,
            marker=METHOD_INSET_MARKERS.get(method, "o"),
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.9,
            markersize=3.5,
            zorder=2,
        )

    if not plotted_values:
        zoom_axis.remove()
        return
    y_min, y_max = min(plotted_values), max(plotted_values)
    padding = max((y_max - y_min) * 0.08, abs(y_max) * 1e-7, 1e-12)
    zoom_axis.set_xlim(zoom_start, zoom_end)
    zoom_axis.set_ylim(y_min - padding, y_max + padding)
    zoom_axis.set_xticks([zoom_start, zoom_end])
    _format_attack_ticks(zoom_axis, max_ticks=4)
    zoom_axis.tick_params(axis="both", labelsize=9, pad=1.5)
    zoom_axis.grid(True, alpha=0.2, linewidth=0.55)
    zoom_axis.set_title(
        f"Late-round mean ({zoom_start}–{zoom_end})",
        fontsize=10,
        fontfamily=TITLE_FONT_FAMILY,
        fontweight="normal",
        pad=3,
    )
    for spine in zoom_axis.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#666666")
    mark_inset(
        axis,
        zoom_axis,
        loc1=2,
        loc2=4,
        facecolor="none",
        edgecolor="#666666",
        linewidth=0.8,
        alpha=0.55,
    )


def _style_axis(axis):
    axis.grid(True, alpha=0.22, linewidth=0.7, which="both")
    axis.spines[["top", "right"]].set_visible(False)


def _format_attack_ticks(axis, max_ticks=5):
    axis.yaxis.set_major_locator(ticker.MaxNLocator(nbins=max_ticks))
    axis.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda value, _: f"{value:.4g}")
    )
    axis.yaxis.set_minor_formatter(ticker.NullFormatter())
    axis.yaxis.offsetText.set_visible(False)


def plot_trajectory_panels(rows, output_base, dpi=300, write_pdf=True):
    clean_methods, attacked_methods = _method_groups(rows)
    references = _reference_values(rows)
    attack_references = _attack_reference_values(rows)
    figure, axes = plt.subplots(2, 3, figsize=(18.0, 10.5), sharex="col")
    for panel_index, (metric, title, ylabel) in enumerate(METRICS):
        clean_axis = axes[0, panel_index]
        attacked_axis = axes[1, panel_index]

        _plot_trajectory_group(clean_axis, rows, clean_methods, metric)
        _add_reference_lines(clean_axis, metric, references)
        clean_axis.set_title(
            f"({chr(97 + panel_index)}) No attack\n{title}",
            fontsize=TITLE_FONT_SIZE,
            fontfamily=TITLE_FONT_FAMILY,
            fontweight="normal",
            pad=12,
        )
        clean_axis.set_ylabel(
            ylabel,
            fontsize=AXIS_LABEL_FONT_SIZE,
            labelpad=10,
        )
        _style_axis(clean_axis)

        reference = attack_references[metric]
        transform = lambda value, reference=reference, metric=metric: scaled_attack_residual(
            value,
            reference,
            metric,
        )
        _plot_trajectory_group(
            attacked_axis,
            rows,
            attacked_methods,
            metric,
            transform=transform,
        )
        attacked_axis.set_yscale("log")
        _format_attack_ticks(attacked_axis)
        attacked_axis.set_title(
            "Adaptive copying\nAttack-limit residual",
            fontsize=TITLE_FONT_SIZE,
            fontfamily=TITLE_FONT_FAMILY,
            fontweight="normal",
            pad=12,
        )
        attacked_axis.set_xlabel(
            "Communication round",
            fontsize=AXIS_LABEL_FONT_SIZE,
            labelpad=8,
        )
        attacked_axis.set_ylabel(
            ATTACK_RESIDUAL_LABELS[metric],
            fontsize=AXIS_LABEL_FONT_SIZE,
            labelpad=10,
        )
        _style_axis(attacked_axis)

    clean_handles, clean_labels = axes[0, 0].get_legend_handles_labels()
    attack_handles, attack_labels = axes[1, 0].get_legend_handles_labels()
    if clean_handles:
        axes[0, 0].legend(
            clean_handles,
            clean_labels,
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(1.0, 0.96),
            fontsize=LEGEND_FONT_SIZE,
        )
    if attack_handles:
        figure.legend(
            attack_handles,
            attack_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=max(1, len(attack_handles)),
            frameon=False,
            fontsize=LEGEND_FONT_SIZE,
        )
    _add_reference_legend(figure)
    figure.tight_layout(rect=(0, 0.06, 1, 0.94), h_pad=3.0, w_pad=2.0)
    for panel_index, (metric, _, _) in enumerate(METRICS):
        reference = attack_references[metric]
        transform = lambda value, reference=reference, metric=metric: scaled_attack_residual(
            value,
            reference,
            metric,
        )
        _add_late_round_inset(
            axes[1, panel_index],
            rows,
            attacked_methods,
            metric,
            transform=transform,
        )
    png_path = output_base.with_name(output_base.name + "_trajectory_panels.png")
    pdf_path = output_base.with_name(output_base.name + "_trajectory_panels.pdf")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    if write_pdf:
        figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path if write_pdf else None


def _plot_final_group(axis, rows, methods, metric, transform=None):
    positions = np.arange(len(methods), dtype=float)
    for position, method in zip(positions, methods):
        method_rows = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: row["seed"],
        )
        values = np.asarray([
            transform(row[metric]) if transform is not None else row[metric]
            for row in method_rows
        ], dtype=float)
        if values.size == 0:
            continue
        jitter = (
            np.linspace(-0.17, 0.17, values.size)
            if values.size > 1
            else np.zeros(1)
        )
        color = METHOD_COLORS.get(method, "#333333")
        axis.scatter(
            position + jitter,
            values,
            color=color,
            s=25,
            alpha=0.48,
            linewidths=0,
            zorder=2,
        )
        mean, ci = mean_and_ci(values)
        if transform is not None:
            lower_error = min(ci, max(0.0, mean - RESIDUAL_FLOOR))
            yerr = np.asarray([[lower_error], [ci]])
        else:
            yerr = ci
        axis.errorbar(
            position,
            mean,
            yerr=yerr,
            fmt="D",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=6.5,
            capsize=3.5,
            elinewidth=1.5,
            zorder=4,
        )
    axis.set_xticks(
        positions,
        [METHOD_SHORT_LABELS.get(method, method) for method in methods],
        rotation=20,
        ha="right",
        rotation_mode="anchor",
    )
    axis.margins(x=0.12)


def plot_final_panels(rows, output_base, dpi=300, write_pdf=True):
    clean_methods, attacked_methods = _method_groups(rows)
    references = _reference_values(rows)
    attack_references = _attack_reference_values(rows)
    figure, axes = plt.subplots(2, 3, figsize=(18.0, 10.0))
    for panel_index, (metric, title, ylabel) in enumerate(METRICS):
        clean_axis = axes[0, panel_index]
        attacked_axis = axes[1, panel_index]

        _plot_final_group(clean_axis, rows, clean_methods, metric)
        _add_reference_lines(clean_axis, metric, references)
        clean_axis.set_title(
            f"({chr(97 + panel_index)}) No attack\nFinal {title.lower()}",
            fontsize=TITLE_FONT_SIZE,
            fontfamily=TITLE_FONT_FAMILY,
            fontweight="normal",
            pad=12,
        )
        clean_axis.set_ylabel(
            ylabel,
            fontsize=AXIS_LABEL_FONT_SIZE,
            labelpad=10,
        )
        _style_axis(clean_axis)

        reference = attack_references[metric]
        transform = lambda value, reference=reference, metric=metric: scaled_attack_residual(
            value,
            reference,
            metric,
        )
        _plot_final_group(
            attacked_axis,
            rows,
            attacked_methods,
            metric,
            transform=transform,
        )
        attacked_axis.set_yscale("log")
        _format_attack_ticks(attacked_axis)
        attacked_axis.set_title(
            "Adaptive copying\nFinal attack-limit residual",
            fontsize=TITLE_FONT_SIZE,
            fontfamily=TITLE_FONT_FAMILY,
            fontweight="normal",
            pad=12,
        )
        attacked_axis.set_ylabel(
            ATTACK_RESIDUAL_LABELS[metric],
            fontsize=AXIS_LABEL_FONT_SIZE,
            labelpad=10,
        )
        _style_axis(attacked_axis)

    _add_reference_legend(figure)
    figure.tight_layout(rect=(0, 0.01, 1, 0.94), h_pad=3.0, w_pad=2.0)
    png_path = output_base.with_name(output_base.name + "_final_panels.png")
    pdf_path = output_base.with_name(output_base.name + "_final_panels.pdf")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    if write_pdf:
        figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path if write_pdf else None


def _default_paths(args):
    trajectory = args.trajectory_csv.resolve()
    stem = trajectory.stem
    base_stem = stem[:-len("_trajectory")] if stem.endswith("_trajectory") else stem
    summary = args.summary
    if summary is None:
        candidate = trajectory.with_name(base_stem + "_summary.csv")
        summary = candidate if candidate.exists() else None
    elif not summary.is_absolute():
        summary = summary.resolve()

    output_base = args.output_prefix
    if output_base is None:
        output_base = trajectory.with_name(base_stem)
    elif not output_base.is_absolute():
        output_base = output_base.resolve()
    if output_base.suffix:
        output_base = output_base.with_suffix("")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    return trajectory, summary, output_base


def main(argv=None):
    args = read_args(argv)
    trajectory_path, summary_path, output_base = _default_paths(args)
    trajectory_rows = load_rows(trajectory_path, require_round=True)
    if summary_path is not None and summary_path.exists():
        final_rows = load_rows(summary_path, require_round=False)
    else:
        final_rows = final_rows_from_trajectory(trajectory_rows)

    trajectory_outputs = plot_trajectory_panels(
        trajectory_rows,
        output_base,
        dpi=args.dpi,
        write_pdf=not args.no_pdf,
    )
    final_outputs = plot_final_panels(
        final_rows,
        output_base,
        dpi=args.dpi,
        write_pdf=not args.no_pdf,
    )
    seeds = sorted({row["seed"] for row in trajectory_rows})
    print(f"Loaded {len(trajectory_rows)} trajectory rows from {trajectory_path}")
    print(f"Paired seeds: {len(seeds)} ({seeds[0]}..{seeds[-1]})")
    for output in (*trajectory_outputs, *final_outputs):
        if output is not None:
            print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
