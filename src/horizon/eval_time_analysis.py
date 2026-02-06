"""Extract and visualize the distribution of human completion times for eval tasks.

Supports any task_source (SWAA, HCAST, RE-Bench). Automatically adapts time
units, histogram bins, and bucket boundaries to the data range.

Uses only Python standard library so no external dependencies are required.
"""

import argparse
import json
import logging
import math
import pathlib
import statistics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_human_times(input_file: pathlib.Path, task_source: str) -> dict[str, dict]:
    """Load runs.jsonl and return dict of task_id -> {human_minutes, task_family}."""
    tasks: dict[str, dict] = {}
    with open(input_file) as f:
        for line in f:
            row = json.loads(line)
            if row.get("task_source") != task_source:
                continue
            tid = row["task_id"]
            if tid not in tasks:
                tasks[tid] = {
                    "task_id": tid,
                    "human_minutes": row["human_minutes"],
                    "task_family": row.get("task_family", ""),
                }
    if not tasks:
        raise ValueError(f"No {task_source} tasks found in {input_file}")
    return tasks


# ---------------------------------------------------------------------------
# Adaptive time formatting
# ---------------------------------------------------------------------------

def _choose_unit(minutes_values: list[float]) -> tuple[str, float, str]:
    """Choose display unit based on the data range.

    Returns (unit_label, divisor, short_suffix) where display_value = minutes / divisor.
    """
    max_val = max(minutes_values)
    if max_val < 1:
        return "seconds", 1 / 60, "s"
    if max_val < 120:
        return "minutes", 1, "min"
    return "hours", 60, "hr"


def fmt_time(minutes: float, unit: str, divisor: float) -> str:
    """Format a minutes value into the chosen display unit."""
    v = minutes / divisor
    if unit == "seconds":
        return f"{v:.1f}s"
    if unit == "minutes":
        if v < 1:
            return f"{v:.2f} min"
        return f"{v:.1f} min"
    # hours
    if v < 0.1:
        return f"{minutes:.1f} min"
    return f"{v:.1f} hr"


def _choose_buckets(minutes_values: list[float]) -> list[tuple[str, float, float]]:
    """Choose time buckets appropriate for the data range (in minutes)."""
    max_val = max(minutes_values)
    if max_val < 1:
        # Sub-minute (seconds)
        return [
            ("0-3s", 0, 3 / 60),
            ("3-5s", 3 / 60, 5 / 60),
            ("5-10s", 5 / 60, 10 / 60),
            ("10-15s", 10 / 60, 15 / 60),
            ("15-30s", 15 / 60, 30 / 60),
            ("30s+", 30 / 60, float("inf")),
        ]
    if max_val < 120:
        # Minutes range
        return [
            ("< 5 min", 0, 5),
            ("5-15 min", 5, 15),
            ("15-30 min", 15, 30),
            ("30-60 min", 30, 60),
            ("1-2 hr", 60, 120),
        ]
    # Hours range
    return [
        ("< 15 min", 0, 15),
        ("15-60 min", 15, 60),
        ("1-2 hr", 60, 120),
        ("2-4 hr", 120, 240),
        ("4-8 hr", 240, 480),
        ("8-16 hr", 480, 960),
        ("16+ hr", 960, float("inf")),
    ]


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def percentile(data: list[float], p: float) -> float:
    s = sorted(data)
    n = len(s)
    k = (p / 100) * (n - 1)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] + (k - lo) * (s[hi] - s[lo])


def print_summary_table(
    task_source: str,
    minutes_values: list[float],
    task_families: dict[str, list[float]],
) -> None:
    unit, div, _ = _choose_unit(minutes_values)
    buckets = _choose_buckets(minutes_values)

    print(f"\n=== {task_source} Eval Human Completion Time Summary ===")
    print(f"  Unique tasks : {len(minutes_values)}")
    print(f"  Min          : {fmt_time(min(minutes_values), unit, div)}")
    print(f"  Max          : {fmt_time(max(minutes_values), unit, div)}")
    print(f"  Mean         : {fmt_time(statistics.mean(minutes_values), unit, div)}")
    if len(minutes_values) > 1:
        print(f"  Std          : {fmt_time(statistics.stdev(minutes_values), unit, div)}")
    for p in [10, 25, 50, 75, 90]:
        print(f"  P{p:<3}         : {fmt_time(percentile(minutes_values, p), unit, div)}")
    print()

    print("  Bucket breakdown:")
    for label, lo, hi in buckets:
        c = sum(1 for m in minutes_values if lo <= m < hi)
        pct = c / len(minutes_values) * 100
        print(f"    {label:>12s} : {c:4d} tasks ({pct:5.1f}%)")
    print()

    max_fam_len = max(len(f) for f in task_families) if task_families else 10
    print("  Per task family:")
    for fam in sorted(task_families.keys()):
        vals = task_families[fam]
        med = percentile(vals, 50) if len(vals) > 1 else vals[0]
        print(
            f"    {fam:>{max_fam_len}s} : {len(vals):3d} tasks, "
            f"median={fmt_time(med, unit, div)}, "
            f"range=[{fmt_time(min(vals), unit, div)}, {fmt_time(max(vals), unit, div)}]"
        )
    print()


# ---------------------------------------------------------------------------
# Histogram helpers
# ---------------------------------------------------------------------------

def _linear_bins(lo: float, hi: float, n_bins: int) -> list[float]:
    step = (hi - lo) / n_bins
    return [lo + i * step for i in range(n_bins + 1)]


def _log_bins(data_min: float, data_max: float, n_bins: int) -> list[float]:
    lo = math.log10(max(data_min, 0.01))
    hi = math.log10(data_max)
    step = (hi - lo) / n_bins
    return [10 ** (lo + i * step) for i in range(n_bins + 1)]


def _histogram(data: list[float], bin_edges: list[float]) -> list[int]:
    counts = [0] * (len(bin_edges) - 1)
    for v in data:
        placed = False
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= v < bin_edges[i + 1]:
                counts[i] += 1
                placed = True
                break
        if not placed and v >= bin_edges[-1]:
            counts[-1] += 1
    return counts


# ---------------------------------------------------------------------------
# SVG generation
# ---------------------------------------------------------------------------

def generate_svg(
    task_source: str,
    minutes_values: list[float],
    task_families: dict[str, list[float]],
    output_file: pathlib.Path,
) -> None:
    """Generate an SVG distribution chart, adapting layout to data range and family count."""
    unit, div, suffix = _choose_unit(minutes_values)
    display_values = [m / div for m in minutes_values]
    display_families = {
        fam: [m / div for m in vals] for fam, vals in task_families.items()
    }

    n_fam = len(task_families)
    # Adaptive SVG height based on number of families
    fam_bar_h = max(min(25, 300 / max(n_fam, 1)), 8)
    bottom_panel_h = int(n_fam * fam_bar_h) + 60
    max_fam_label = max((len(f) for f in task_families), default=10)
    fam_label_width = min(max_fam_label * 7 + 20, 280)

    svg_w, svg_h = 1200, 55 + 320 + 100 + bottom_panel_h + 50
    margin_top = 55
    margin_left, margin_right = 70, 30
    top_gap = 100
    top_panel_h = 320
    gap = 120
    color = "#4C72B0"
    color2 = "#55A868"

    # ===== Panel 1 (top-left): Histogram =====
    p1_x = margin_left
    p1_y = margin_top
    p1_w = (svg_w - margin_left - margin_right - gap) // 2
    p1_h = top_panel_h

    use_log = (max(display_values) / max(min(display_values), 0.001)) > 20
    n_bins = min(25, max(10, len(display_values) // 3))

    if use_log:
        bin_edges = _log_bins(min(display_values), max(display_values), n_bins)
    else:
        nice_max = math.ceil(max(display_values) * 1.05)
        bin_edges = _linear_bins(0 if min(display_values) > 0 else min(display_values), nice_max, n_bins)

    counts = _histogram(display_values, bin_edges)
    max_count = max(counts) if counts else 1

    def x_pos_hist(val: float) -> float:
        if use_log:
            log_min = math.log10(max(bin_edges[0], 0.01))
            log_max = math.log10(bin_edges[-1])
            log_range = log_max - log_min
            return p1_x + (math.log10(max(val, bin_edges[0])) - log_min) / log_range * p1_w
        return p1_x + (val - bin_edges[0]) / (bin_edges[-1] - bin_edges[0]) * p1_w

    bars_svg = []
    for i, c in enumerate(counts):
        if c == 0:
            continue
        bx = x_pos_hist(bin_edges[i])
        bw = x_pos_hist(bin_edges[i + 1]) - bx
        bh = (c / max_count) * p1_h
        by = p1_y + p1_h - bh
        bars_svg.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{max(bw - 1, 1):.1f}" '
            f'height="{bh:.1f}" fill="{color}" opacity="0.85" />'
        )

    # X-axis ticks
    xtick_svg = []
    if use_log:
        # Log-scale ticks
        tick_candidates = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 240, 480, 960, 1800]
        tick_vals = [t for t in tick_candidates if bin_edges[0] * 0.9 <= t <= bin_edges[-1] * 1.1]
        if len(tick_vals) > 12:
            tick_vals = tick_vals[::2]
    else:
        span = bin_edges[-1] - bin_edges[0]
        step = max(1, round(span / 10))
        if step > 5:
            step = round(step / 5) * 5
        tick_vals = list(range(int(bin_edges[0]), int(bin_edges[-1]) + 1, step))

    for t in tick_vals:
        tx = x_pos_hist(t)
        if tx < p1_x - 5 or tx > p1_x + p1_w + 5:
            continue
        xtick_svg.append(
            f'<line x1="{tx:.1f}" y1="{p1_y + p1_h}" '
            f'x2="{tx:.1f}" y2="{p1_y + p1_h + 6}" stroke="#333" />'
        )
        if unit == "seconds":
            label = f"{t:.0f}s" if t >= 1 else f"{t:.1f}s"
        elif unit == "minutes":
            label = f"{t:.0f}" if t == int(t) else f"{t:.1f}"
        else:
            label = f"{t:.0f}" if t == int(t) else f"{t:.1f}"
        xtick_svg.append(
            f'<text x="{tx:.1f}" y="{p1_y + p1_h + 20}" '
            f'text-anchor="middle" font-size="10" fill="#333">{label}</text>'
        )

    # Y-axis ticks
    ytick_step = max(1, max_count // 5)
    ytick_svg = []
    for yv in range(0, max_count + 1, ytick_step):
        yy = p1_y + p1_h - (yv / max_count) * p1_h
        ytick_svg.append(
            f'<line x1="{p1_x - 6}" y1="{yy:.1f}" x2="{p1_x}" y2="{yy:.1f}" stroke="#333" />'
        )
        ytick_svg.append(
            f'<line x1="{p1_x}" y1="{yy:.1f}" x2="{p1_x + p1_w}" y2="{yy:.1f}" stroke="#ddd" />'
        )
        ytick_svg.append(
            f'<text x="{p1_x - 10}" y="{yy + 4:.1f}" text-anchor="end" font-size="11" fill="#333">{yv}</text>'
        )

    # Stats box
    med = percentile(minutes_values, 50)
    mean_val = statistics.mean(minutes_values)
    stats_lines = [
        f"n = {len(minutes_values)} tasks",
        f"min = {fmt_time(min(minutes_values), unit, div)}",
        f"median = {fmt_time(med, unit, div)}",
        f"mean = {fmt_time(mean_val, unit, div)}",
        f"max = {fmt_time(max(minutes_values), unit, div)}",
    ]
    stats_x = p1_x + p1_w - 10
    stats_y = p1_y + 20
    stats_svg = [
        f'<rect x="{stats_x - 170}" y="{stats_y - 14}" width="175" '
        f'height="{len(stats_lines) * 16 + 8}" rx="4" fill="white" fill-opacity="0.85" stroke="#ccc" />'
    ]
    for idx, line in enumerate(stats_lines):
        stats_svg.append(
            f'<text x="{stats_x - 5}" y="{stats_y + idx * 16}" '
            f'text-anchor="end" font-size="10" fill="#333" font-family="monospace">{line}</text>'
        )

    # ===== Panel 2 (top-right): Time bucket bar chart =====
    p2_x = margin_left + p1_w + gap
    p2_y = margin_top
    p2_w = p1_w
    p2_h = top_panel_h

    buckets = _choose_buckets(minutes_values)
    bucket_counts = []
    bucket_labels = []
    for label, lo, hi in buckets:
        c = sum(1 for m in minutes_values if lo <= m < hi)
        bucket_counts.append(c)
        bucket_labels.append(label)

    max_bc = max(bucket_counts) if bucket_counts else 1
    n_bars = len(bucket_counts)
    bar_w = p2_w / (n_bars + 1)
    bar_gap = bar_w * 0.12

    bars2_svg = []
    for i, (lbl, cnt) in enumerate(zip(bucket_labels, bucket_counts)):
        bx = p2_x + (i + 0.5) * bar_w
        bh = (cnt / max_bc) * p2_h if max_bc > 0 else 0
        by = p2_y + p2_h - bh
        bars2_svg.append(
            f'<rect x="{bx + bar_gap:.1f}" y="{by:.1f}" '
            f'width="{bar_w - 2 * bar_gap:.1f}" height="{bh:.1f}" fill="{color}" opacity="0.85" />'
        )
        if cnt > 0:
            bars2_svg.append(
                f'<text x="{bx + bar_w / 2:.1f}" y="{by - 5:.1f}" '
                f'text-anchor="middle" font-size="11" font-weight="bold" fill="#333">{cnt}</text>'
            )
        bars2_svg.append(
            f'<text x="{bx + bar_w / 2:.1f}" y="{p2_y + p2_h + 20}" '
            f'text-anchor="middle" font-size="10" fill="#333">{lbl}</text>'
        )

    ytick2_step = max(1, max_bc // 5)
    ytick2_svg = []
    for yv in range(0, max_bc + 1, ytick2_step):
        yy = p2_y + p2_h - (yv / max_bc) * p2_h
        ytick2_svg.append(
            f'<line x1="{p2_x - 6}" y1="{yy:.1f}" x2="{p2_x}" y2="{yy:.1f}" stroke="#333" />'
        )
        ytick2_svg.append(
            f'<line x1="{p2_x}" y1="{yy:.1f}" x2="{p2_x + p2_w}" y2="{yy:.1f}" stroke="#ddd" />'
        )
        ytick2_svg.append(
            f'<text x="{p2_x - 10}" y="{yy + 4:.1f}" text-anchor="end" font-size="11" fill="#333">{yv}</text>'
        )

    # ===== Panel 3 (bottom): Per-family horizontal strip chart =====
    p3_x = margin_left + fam_label_width
    p3_y = margin_top + top_panel_h + top_gap
    p3_w = svg_w - p3_x - margin_right - 20

    families = sorted(display_families.keys())
    fam_gap_px = fam_bar_h * 0.15
    max_disp = max(max(vals) for vals in display_families.values())
    x_scale = p3_w / (max_disp * 1.1)

    fam_svg = []
    for i, fam in enumerate(families):
        vals = sorted(display_families[fam])
        med_val = percentile(vals, 50) if len(vals) > 1 else vals[0]
        by = p3_y + i * fam_bar_h

        min_v, max_v = min(vals), max(vals)
        rx = p3_x + min_v * x_scale
        rw = max((max_v - min_v) * x_scale, 2)
        fam_svg.append(
            f'<rect x="{rx:.1f}" y="{by + fam_gap_px:.1f}" '
            f'width="{rw:.1f}" height="{fam_bar_h - 2 * fam_gap_px:.1f}" '
            f'fill="{color}" opacity="0.3" rx="2" />'
        )

        mx = p3_x + med_val * x_scale
        fam_svg.append(
            f'<line x1="{mx:.1f}" y1="{by + fam_gap_px:.1f}" '
            f'x2="{mx:.1f}" y2="{by + fam_bar_h - fam_gap_px:.1f}" '
            f'stroke="{color}" stroke-width="2" />'
        )

        for v in vals:
            vx = p3_x + v * x_scale
            vy = by + fam_bar_h / 2
            fam_svg.append(
                f'<circle cx="{vx:.1f}" cy="{vy:.1f}" r="3" fill="{color2}" opacity="0.7" />'
            )

        fam_svg.append(
            f'<text x="{p3_x - 8}" y="{by + fam_bar_h / 2 + 4:.1f}" '
            f'text-anchor="end" font-size="9" fill="#333">{fam}</text>'
        )

    # X-axis ticks for panel 3
    fam_xtick_svg = []
    x_bottom = p3_y + n_fam * fam_bar_h
    tick_step = max_disp / 8
    # Round tick_step to a nice value
    mag = 10 ** math.floor(math.log10(max(tick_step, 0.01)))
    tick_step = math.ceil(tick_step / mag) * mag

    t = 0.0
    while t <= max_disp * 1.05:
        tx = p3_x + t * x_scale
        if tx > p3_x + p3_w + 5:
            break
        fam_xtick_svg.append(
            f'<line x1="{tx:.1f}" y1="{x_bottom}" x2="{tx:.1f}" y2="{x_bottom + 6}" stroke="#333" />'
        )
        fam_xtick_svg.append(
            f'<line x1="{tx:.1f}" y1="{p3_y}" x2="{tx:.1f}" y2="{x_bottom}" stroke="#eee" />'
        )
        if unit == "seconds":
            lbl = f"{t:.0f}s"
        elif unit == "minutes":
            lbl = f"{t:.0f} min" if t == int(t) else f"{t:.1f} min"
        else:
            lbl = f"{t:.0f} hr" if t == int(t) else f"{t:.1f} hr"
        fam_xtick_svg.append(
            f'<text x="{tx:.1f}" y="{x_bottom + 18}" '
            f'text-anchor="middle" font-size="10" fill="#333">{lbl}</text>'
        )
        t += tick_step

    hist_type = "Log-scale histogram" if use_log else "Histogram"
    unit_label = f"Human completion time ({unit})"

    # ===== Assemble SVG =====
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" '
        f'viewBox="0 0 {svg_w} {svg_h}" style="background:white;">',
        '<style>text { font-family: Arial, Helvetica, sans-serif; }</style>',
        f'<text x="{svg_w / 2}" y="30" text-anchor="middle" font-size="16" '
        f'font-weight="bold" fill="#222">Distribution of Human Completion Times '
        f'for {task_source} Eval Tasks</text>',
        # Panel 1
        f'<text x="{p1_x + p1_w / 2}" y="{p1_y - 8}" text-anchor="middle" font-size="13" fill="#333">{hist_type}</text>',
        f'<line x1="{p1_x}" y1="{p1_y + p1_h}" x2="{p1_x + p1_w}" y2="{p1_y + p1_h}" stroke="#333" />',
        f'<line x1="{p1_x}" y1="{p1_y}" x2="{p1_x}" y2="{p1_y + p1_h}" stroke="#333" />',
        f'<text x="{p1_x + p1_w / 2}" y="{p1_y + p1_h + 42}" text-anchor="middle" font-size="12" fill="#333">{unit_label}</text>',
        f'<text x="{p1_x - 50}" y="{p1_y + p1_h / 2}" text-anchor="middle" font-size="12" fill="#333" '
        f'transform="rotate(-90,{p1_x - 50},{p1_y + p1_h / 2})">Number of tasks</text>',
        # Panel 2
        f'<text x="{p2_x + p2_w / 2}" y="{p2_y - 8}" text-anchor="middle" font-size="13" fill="#333">Tasks by time bucket</text>',
        f'<line x1="{p2_x}" y1="{p2_y + p2_h}" x2="{p2_x + p2_w}" y2="{p2_y + p2_h}" stroke="#333" />',
        f'<line x1="{p2_x}" y1="{p2_y}" x2="{p2_x}" y2="{p2_y + p2_h}" stroke="#333" />',
        f'<text x="{p2_x + p2_w / 2}" y="{p2_y + p2_h + 42}" text-anchor="middle" font-size="12" fill="#333">Time bucket</text>',
        f'<text x="{p2_x - 50}" y="{p2_y + p2_h / 2}" text-anchor="middle" font-size="12" fill="#333" '
        f'transform="rotate(-90,{p2_x - 50},{p2_y + p2_h / 2})">Number of tasks</text>',
        # Panel 3
        f'<text x="{svg_w / 2}" y="{p3_y - 15}" text-anchor="middle" font-size="13" fill="#333">'
        f'Human time by task family (dots = tasks, line = median)</text>',
        f'<line x1="{p3_x}" y1="{x_bottom}" x2="{p3_x + p3_w}" y2="{x_bottom}" stroke="#333" />',
        f'<line x1="{p3_x}" y1="{p3_y}" x2="{p3_x}" y2="{x_bottom}" stroke="#333" />',
        f'<text x="{p3_x + p3_w / 2}" y="{x_bottom + 38}" text-anchor="middle" font-size="12" fill="#333">Time ({unit})</text>',
    ]
    svg_parts.extend(bars_svg)
    svg_parts.extend(xtick_svg)
    svg_parts.extend(ytick_svg)
    svg_parts.extend(stats_svg)
    svg_parts.extend(bars2_svg)
    svg_parts.extend(ytick2_svg)
    svg_parts.extend(fam_svg)
    svg_parts.extend(fam_xtick_svg)
    svg_parts.append("</svg>")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(svg_parts))
    logger.info(f"Saved SVG plot to {output_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze eval human completion time distribution"
    )
    parser.add_argument(
        "--input-files", nargs="+", type=pathlib.Path, required=True,
        help="Path(s) to runs.jsonl file(s)",
    )
    parser.add_argument(
        "--task-source", type=str, required=True,
        help="Task source to filter (SWAA, HCAST, RE-Bench)",
    )
    parser.add_argument(
        "--output-file", type=pathlib.Path, default=None,
        help="Output path for the distribution plot SVG",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper())

    all_tasks: dict[str, dict] = {}
    for f in args.input_files:
        logger.info(f"Loading {args.task_source} tasks from {f}")
        try:
            tasks = load_human_times(f, args.task_source)
        except ValueError:
            logger.warning(f"No {args.task_source} tasks in {f}, skipping")
            continue
        for tid, info in tasks.items():
            if tid not in all_tasks:
                all_tasks[tid] = info

    if not all_tasks:
        raise SystemExit(f"No {args.task_source} tasks found in any input file")

    minutes_values = [t["human_minutes"] for t in all_tasks.values()]
    task_families: dict[str, list[float]] = {}
    for t in all_tasks.values():
        task_families.setdefault(t["task_family"], []).append(t["human_minutes"])

    logger.info(f"Total unique {args.task_source} tasks: {len(minutes_values)}")
    logger.info(f"Task families: {len(task_families)}")

    print_summary_table(args.task_source, minutes_values, task_families)

    if args.output_file:
        generate_svg(args.task_source, minutes_values, task_families, args.output_file)
        print(f"Distribution plot saved to: {args.output_file}")


if __name__ == "__main__":
    main()
