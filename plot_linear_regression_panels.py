"""Plot multi-seed linear-regression trajectory and final-result panels."""

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


FIGURE_WIDTH = 8
TRAJECTORY_FIGURE_SIZE = (FIGURE_WIDTH, 5)
FINAL_FIGURE_SIZE = (FIGURE_WIDTH, 4.8)
TITLE_FONT_SIZE = 9
TITLE_FONT_FAMILY = "Tinos"
AXIS_LABEL_FONT_SIZE = 9
TICK_FONT_SIZE = 8
LEGEND_FONT_SIZE = 8
plt.rcParams.update({
    "font.size": 8,
    "font.family": "serif",
    "font.serif": [TITLE_FONT_FAMILY],
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
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
    "qFFL+GM (adaptive copying)": "#8A9A00",
    "qFFL+Krum (adaptive copying)": "#009E73",
    "qFFL+MDA (adaptive copying)": "#56B4E9",
    "qFFL+FABA (adaptive copying)": "#8C564B",
    "qFFL+NBS (adaptive copying)": "#332288",
}
METHOD_SHORT_LABELS = {
    "FedAvg (clean)": "FedAvg",
    "qFFL (clean)": r"$q$-FFL",
    "qFFL+CWM (adaptive copying)": "CWM",
    "qFFL+CWTM (adaptive copying)": "CWTM",
    "qFFL+GM (adaptive copying)": "GM",
    "qFFL+Krum (adaptive copying)": "Krum",
    "qFFL+MDA (adaptive copying)": "MDA",
    "qFFL+FABA (adaptive copying)": "FABA",
    "qFFL+NBS (adaptive copying)": "NBS",
}
METHOD_MARKERS = {
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
REFERENCE_STYLES = {
    "fairness_optimum": {
        "label": r"Fairness-aware optimum $w_g$",
        "color": "#1A1A1A",
        "linestyle": "-.",
        "marker": "o",
        "markerfacecolor": "white",
    },
    "honest_optimum": {
        "label": r"Honest optimum $w_{\mathcal{H}}^*$",
        "color": "#7A0177",
        "linestyle": ":",
        "marker": "D",
        "markerfacecolor": "white",
    },
    "aligned_model": {
        "label": r"Aligned model $v$",
        "color": "#B2182B",
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
COLUMN_HEADERS = (
    "Honest-loss variance\n" r"$V_{\mathcal{H}}(w_t)$",
    (
        "Average-loss gap\n"
        r"$F_{\mathcal{H}}(w_t)-F_{\mathcal{H}}(w_{\mathcal{H}}^*)$"
    ),
    "Distance to aligned model\n" r"$\|w_t-v\|_2$",
)
FINAL_COLUMN_HEADERS = (
    "Final honest-loss variance\n" r"$V_{\mathcal{H}}(w_T)$",
    (
        "Final average-loss gap\n"
        r"$F_{\mathcal{H}}(w_T)-F_{\mathcal{H}}(w_{\mathcal{H}}^*)$"
    ),
    "Final distance to aligned model\n" r"$\|w_T-v\|_2$",
)
FINAL_YLABELS = (
    r"$V_{\mathcal{H}}(w_T)$",
    r"$F_{\mathcal{H}}(w_T)-F_{\mathcal{H}}(w_{\mathcal{H}}^*)$",
    r"$\|w_T-v\|_2$",
)
NUMERIC_FIELDS = {
    "seed": int,
    "round": int,
    "batch_size": int,
    "honest_worker_count": int,
    "noise_std": float,
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
        description="Draw multi-seed panels from linear-regression CSV tables."
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
    parser.add_argument("--dpi", type=positive_int, default=600)
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


def _fairness_aware_optimum(delta, honest_count, noise_std, batch_size):
    """Return the scalar coordinate of w_g for q=1 and Sigma=I."""
    noise_floor = 0.5 * batch_size * noise_std ** 2

    def objective_gradient(coordinate):
        aligned_loss = 0.5 * coordinate ** 2 + noise_floor
        exceptional_offset = coordinate - delta
        exceptional_loss = 0.5 * exceptional_offset ** 2 + noise_floor
        return (
            (honest_count - 1) * aligned_loss * coordinate
            + exceptional_loss * exceptional_offset
        )

    lower = delta / honest_count
    upper = delta / 2.0
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if objective_gradient(midpoint) < 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def _reference_values(rows):
    delta = float(rows[0].get("delta", 1.0))
    honest_count = int(rows[0].get("honest_worker_count", 8))
    noise_std = float(rows[0].get("noise_std", 1e-3))
    batch_size = int(rows[0].get("batch_size", 20))
    honest_optimum = delta / honest_count
    fairness_optimum = _fairness_aware_optimum(
        delta,
        honest_count,
        noise_std,
        batch_size,
    )

    def variance_at(coordinate):
        loss_difference = 0.5 * delta ** 2 - delta * coordinate
        return (
            (honest_count - 1)
            / honest_count ** 2
            * loss_difference ** 2
        )

    def average_loss_gap_at(coordinate):
        return 0.5 * (coordinate - honest_optimum) ** 2

    return {
        "honest_loss_variance": {
            "fairness_optimum": variance_at(fairness_optimum),
            "honest_optimum": variance_at(honest_optimum),
            "aligned_model": variance_at(0.0),
        },
        "honest_average_loss_gap": {
            "fairness_optimum": average_loss_gap_at(fairness_optimum),
            "honest_optimum": 0.0,
            "aligned_model": average_loss_gap_at(0.0),
        },
        "distance_to_v": {
            "fairness_optimum": fairness_optimum,
            "honest_optimum": honest_optimum,
            "aligned_model": 0.0,
        },
    }


def _is_clean_method(method):
    return method.endswith("(clean)")


def _method_groups(rows):
    methods = ordered_methods(rows)
    return (
        [method for method in methods if _is_clean_method(method)],
        [method for method in methods if not _is_clean_method(method)],
    )


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
            linewidth=0.9,
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
            markeredgewidth=0.8,
            markersize=3.5,
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
            linewidth=1.0,
            marker=style["marker"],
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor=style["color"],
            markeredgewidth=0.8,
            markersize=4.0,
            label=style["label"],
        )
        for style in REFERENCE_STYLES.values()
    ]


def _add_reference_legend(figure, y=0.005):
    figure.legend(
        handles=_reference_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=len(REFERENCE_STYLES),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=2.3,
        handletextpad=0.6,
        columnspacing=1.3,
    )


def _plot_trajectory_group(
    axis,
    rows,
    methods,
    metric,
):
    aggregated = aggregate_trajectories(rows, metric)
    for method in methods:
        color = METHOD_COLORS.get(method, "#333333")
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
            linewidth=1.4,
            alpha=0.98,
            zorder=3,
        )
        lower = np.maximum(means - cis, 0.0)
        axis.fill_between(
            rounds,
            lower,
            means + cis,
            color=color,
            alpha=0.24,
            linewidth=0,
            zorder=2,
        )


def _late_round_window(aggregated, methods, fraction=0.2):
    rounds = sorted({
        point[0]
        for method in methods
        for point in aggregated.get(method, [])
    })
    if len(rounds) < 2:
        return None
    start = rounds[0] + (rounds[-1] - rounds[0]) * (1.0 - fraction)
    late_rounds = [round_id for round_id in rounds if round_id >= start]
    if len(late_rounds) < 2:
        late_rounds = rounds[-2:]
    return late_rounds[0], late_rounds[-1]


def _add_late_round_inset(axis, rows, methods, metric):
    """Zoom the final 20% of raw attacked trajectories using method means."""
    aggregated = aggregate_trajectories(rows, metric)
    window = _late_round_window(aggregated, methods)
    if window is None:
        return
    zoom_start, zoom_end = window
    zoom_axis = inset_axes(
        axis,
        width="48%",
        height="43%",
        loc="center right",
        borderpad=0.65,
    )
    plotted_values = []
    for method in methods:
        points = [
            point for point in aggregated.get(method, [])
            if zoom_start <= point[0] <= zoom_end
        ]
        if not points:
            continue
        rounds = np.asarray([point[0] for point in points])
        means = np.asarray([point[1] for point in points])
        plotted_values.extend(means.tolist())
        color = METHOD_COLORS.get(method, "#333333")
        zoom_axis.plot(
            rounds,
            means,
            color=color,
            linewidth=0.85,
            marker=METHOD_MARKERS.get(method, "o"),
            markevery=max(1, len(rounds) // 3),
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.5,
            markersize=2.0,
            zorder=2,
        )
    if not plotted_values:
        zoom_axis.remove()
        return

    y_min, y_max = min(plotted_values), max(plotted_values)
    padding = max((y_max - y_min) * 0.12, abs(y_max) * 2e-4, 1e-9)
    zoom_axis.set_xlim(zoom_start, zoom_end)
    zoom_axis.set_ylim(y_min - padding, y_max + padding)
    zoom_axis.set_xticks([zoom_start, zoom_end])
    zoom_axis.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    zoom_axis.tick_params(axis="both", labelsize=5, pad=1.0)
    zoom_axis.grid(True, alpha=0.2, linewidth=0.3)
    zoom_axis.set_facecolor((1.0, 1.0, 1.0, 0.94))
    zoom_axis.set_title(
        f"Rounds {zoom_start:g}–{zoom_end:g}",
        fontsize=5.5,
        fontfamily=TITLE_FONT_FAMILY,
        fontweight="normal",
        pad=1.5,
    )
    for spine in zoom_axis.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.5)
    mark_inset(
        axis,
        zoom_axis,
        loc1=1,
        loc2=4,
        facecolor="none",
        edgecolor="#777777",
        linewidth=0.45,
        alpha=0.5,
    )


def _style_axis(axis):
    axis.grid(True, alpha=0.22, linewidth=0.4, which="both")
    axis.spines[["top", "right"]].set_visible(False)


def _add_column_headers(figure, axes, titles, y=0.9):
    """Place one shared metric heading above each two-panel column."""
    for axis, title in zip(axes, titles):
        position = axis.get_position()
        figure.text(
            0.5 * (position.x0 + position.x1),
            y,
            title,
            ha="center",
            va="center",
            fontsize=TITLE_FONT_SIZE,
            fontfamily=TITLE_FONT_FAMILY,
            fontweight="normal",
        )


def _add_row_header(figure, axes, label, y):
    """Draw a centered row label with rules spanning the three panels."""
    positions = [axis.get_position() for axis in axes]
    left = min(position.x0 for position in positions)
    right = max(position.x1 for position in positions)
    center = 0.5 * (left + right)
    half_gap = 0.055 if label == "No attack" else 0.09
    for start, end in ((left, center - half_gap), (center + half_gap, right)):
        figure.add_artist(Line2D(
            [start, end],
            [y, y],
            transform=figure.transFigure,
            color="#777777",
            linewidth=0.5,
            alpha=0.55,
        ))
    figure.text(
        center,
        y,
        label,
        ha="center",
        va="center",
        fontsize=TITLE_FONT_SIZE,
        fontfamily=TITLE_FONT_FAMILY,
        fontweight="normal",
    )


def _add_facet_headers(figure, axes, column_headers, middle_y=None):
    """Add column headings and one centered heading for each facet row."""
    _add_column_headers(figure, axes[0], column_headers)
    top_positions = [axis.get_position() for axis in axes[0]]
    bottom_positions = [axis.get_position() for axis in axes[1]]
    _add_row_header(
        figure,
        axes[0],
        "No attack",
        max(position.y1 for position in top_positions) + 0.027,
    )
    if middle_y is None:
        middle_y = 0.5 * (
            min(position.y0 for position in top_positions)
            + max(position.y1 for position in bottom_positions) - 0.04
        )
    _add_row_header(figure, axes[1], "Adaptive copying", middle_y)


def _add_panel_labels(axes):
    for panel_index, axis in enumerate(axes.flat):
        axis.text(
            0.0,
            1.015,
            f"({chr(97 + panel_index)})",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=LEGEND_FONT_SIZE,
            fontfamily=TITLE_FONT_FAMILY,
            zorder=10,
            clip_on=False,
        )


def plot_trajectory_panels(rows, output_base, dpi=600, write_pdf=True):
    clean_methods, attacked_methods = _method_groups(rows)
    references = _reference_values(rows)
    figure, axes = plt.subplots(
        2,
        3,
        figsize=TRAJECTORY_FIGURE_SIZE,
        sharex="col",
        sharey="col",
    )
    for panel_index, (metric, title, ylabel) in enumerate(METRICS):
        clean_axis = axes[0, panel_index]
        attacked_axis = axes[1, panel_index]

        _plot_trajectory_group(clean_axis, rows, clean_methods, metric)
        _add_reference_lines(clean_axis, metric, references)
        clean_axis.set_ylabel(ylabel, labelpad=4)
        _style_axis(clean_axis)

        _plot_trajectory_group(
            attacked_axis,
            rows,
            attacked_methods,
            metric,
        )
        _add_reference_lines(attacked_axis, metric, references)
        attacked_axis.set_ylabel(ylabel, labelpad=4)
        _style_axis(attacked_axis)

    for axis in axes.flat:
        axis.tick_params(labelleft=True)

    clean_handles, clean_labels = axes[0, 0].get_legend_handles_labels()
    attack_handles, attack_labels = axes[1, 0].get_legend_handles_labels()
    if clean_handles:
        axes[0, 0].legend(
            clean_handles,
            clean_labels,
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(1.0, 0.90),
            fontsize=LEGEND_FONT_SIZE,
        )
    if attack_handles:
        figure.legend(
            attack_handles,
            attack_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.002),
            ncol=max(1, len(attack_handles)),
            frameon=False,
            fontsize=LEGEND_FONT_SIZE,
            handlelength=1.8,
            handletextpad=0.5,
            columnspacing=1.1,
        )
    _add_reference_legend(figure, y=0.047)
    figure.tight_layout(rect=(0, 0.15, 1, 0.83), h_pad=3.8, w_pad=1.6)
    _add_facet_headers(figure, axes, COLUMN_HEADERS)
    _add_panel_labels(axes)
    for panel_index, (metric, _, _) in enumerate(METRICS):
        _add_late_round_inset(
            axes[1, panel_index],
            rows,
            attacked_methods,
            metric,
        )
    figure.supxlabel(
        "Communication round",
        x=0.5,
        y=0.112,
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontfamily=TITLE_FONT_FAMILY,
    )
    png_path = output_base.with_name(output_base.name + "_trajectory_panels.png")
    pdf_path = output_base.with_name(output_base.name + "_trajectory_panels.pdf")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    if write_pdf:
        figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path if write_pdf else None


def _plot_final_group(
    axis,
    rows,
    methods,
    metric,
    transform=None,
    tick_rotation=40,
):
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
            s=7,
            alpha=0.28,
            linewidths=0,
            zorder=2,
        )
        mean, ci = mean_and_ci(values)
        axis.errorbar(
            position,
            mean,
            yerr=ci,
            fmt="D",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.55,
            markersize=4.8,
            capsize=2.5,
            elinewidth=0.9,
            zorder=4,
        )
    axis.set_xticks(
        positions,
        [METHOD_SHORT_LABELS.get(method, method) for method in methods],
        rotation=tick_rotation,
        ha="center" if tick_rotation == 0 else "right",
        rotation_mode="anchor",
    )
    axis.margins(x=0.12)


def plot_final_panels(rows, output_base, dpi=600, write_pdf=True):
    clean_methods, attacked_methods = _method_groups(rows)
    references = _reference_values(rows)
    figure, axes = plt.subplots(
        2,
        3,
        figsize=FINAL_FIGURE_SIZE,
        sharey="col",
    )
    for panel_index, (metric, title, ylabel) in enumerate(METRICS):
        clean_axis = axes[0, panel_index]
        attacked_axis = axes[1, panel_index]

        _plot_final_group(
            clean_axis,
            rows,
            clean_methods,
            metric,
            tick_rotation=0,
        )
        _add_reference_lines(clean_axis, metric, references)
        clean_axis.set_ylabel(FINAL_YLABELS[panel_index], labelpad=4)
        _style_axis(clean_axis)

        _plot_final_group(
            attacked_axis,
            rows,
            attacked_methods,
            metric,
        )
        _add_reference_lines(attacked_axis, metric, references)
        attacked_axis.set_ylabel(FINAL_YLABELS[panel_index], labelpad=4)
        _style_axis(attacked_axis)

    for axis in axes.flat:
        axis.tick_params(labelleft=True)

    _add_reference_legend(figure, y=0.065)
    figure.tight_layout(rect=(0, 0.13, 1, 0.83), h_pad=2.2, w_pad=1.3)
    _add_facet_headers(
        figure,
        axes,
        FINAL_COLUMN_HEADERS,
        # The top row has categorical labels, so anchor the second row heading
        # just above its axes instead of centering it in the inter-row gap.
        middle_y=max(axis.get_position().y1 for axis in axes[1]) + 0.022,
    )
    _add_panel_labels(axes)
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
