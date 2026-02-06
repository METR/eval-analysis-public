"""Extract and visualize the distribution of human completion times for SWAA eval tasks.

Reads runs.jsonl data, filters for SWAA tasks, deduplicates to unique task-level
human_minutes, and produces distribution plots (SVG histogram + summary statistics).

Uses only Python standard library so no external dependencies are required.
"""

import argparse
import json
import logging
import math
import pathlib
import statistics

logger = logging.getLogger(__name__)


def load_swaa_human_times(input_file: pathlib.Path) -> dict[str, dict]:
    """Load runs.jsonl and return dict of task_id -> {human_minutes, task_family}."""
    tasks: dict[str, dict] = {}
    with open(input_file) as f:
        for line in f:
            row = json.loads(line)
            if row.get("task_source") != "SWAA":
                continue
            tid = row["task_id"]
            if tid not in tasks:
                tasks[tid] = {
                    "task_id": tid,
                    "human_minutes": row["human_minutes"],
                    "task_family": row.get("task_family", ""),
                }
    if not tasks:
        raise ValueError(f"No SWAA tasks found in {input_file}")
    return tasks


def fmt_seconds(s: float) -> str:
    """Format seconds into a human-readable string."""
    if s < 60:
        return f"{s:.1f}s"
    minutes = s / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes / 60:.1f}h"


def percentile(data: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) using linear interpolation."""
    s = sorted(data)
    n = len(s)
    k = (p / 100) * (n - 1)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] + (k - lo) * (s[hi] - s[lo])


def print_summary_table(human_secs: list[float], task_families: dict[str, list[float]]) -> None:
    """Print a text summary of SWAA human times to stdout."""
    pcts = [10, 25, 50, 75, 90]
    print("\n=== SWAA Eval Human Completion Time Summary ===")
    print(f"  Unique tasks : {len(human_secs)}")
    print(f"  Min          : {fmt_seconds(min(human_secs))}")
    print(f"  Max          : {fmt_seconds(max(human_secs))}")
    print(f"  Mean         : {fmt_seconds(statistics.mean(human_secs))}")
    print(f"  Std          : {fmt_seconds(statistics.stdev(human_secs))}")
    for p in pcts:
        print(f"  P{p:<3}         : {fmt_seconds(percentile(human_secs, p))}")
    print()

    # Buckets in seconds - adapted for the SWAA range (1s - 45s)
    buckets = [
        ("0-3s", 0, 3),
        ("3-5s", 3, 5),
        ("5-10s", 5, 10),
        ("10-15s", 10, 15),
        ("15-30s", 15, 30),
        ("30s+", 30, float("inf")),
    ]
    print("  Bucket breakdown:")
    for label, lo, hi in buckets:
        c = sum(1 for s in human_secs if lo <= s < hi)
        pct = c / len(human_secs) * 100
        print(f"    {label:>10s} : {c:4d} tasks ({pct:5.1f}%)")
    print()

    # Per-family breakdown
    print("  Per task family:")
    for fam in sorted(task_families.keys()):
        vals = task_families[fam]
        med = percentile(vals, 50) if len(vals) > 1 else vals[0]
        print(
            f"    {fam:>25s} : {len(vals):3d} tasks, "
            f"median={fmt_seconds(med)}, "
            f"range=[{fmt_seconds(min(vals))}, {fmt_seconds(max(vals))}]"
        )
    print()


def _linear_bins(lo: float, hi: float, n_bins: int) -> list[float]:
    """Generate evenly spaced bin edges."""
    step = (hi - lo) / n_bins
    return [lo + i * step for i in range(n_bins + 1)]


def _histogram(data: list[float], bin_edges: list[float]) -> list[int]:
    """Bin data into histogram counts."""
    counts = [0] * (len(bin_edges) - 1)
    for v in data:
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= v < bin_edges[i + 1]:
                counts[i] += 1
                break
        else:
            if v >= bin_edges[-1]:
                counts[-1] += 1
    return counts


def generate_svg(
    human_secs: list[float],
    task_families: dict[str, list[float]],
    output_file: pathlib.Path,
) -> None:
    """Generate an SVG with three panels: histogram, bucket bar chart, family breakdown."""
    # ---- Layout ----
    svg_w, svg_h = 1200, 850
    margin_top = 55
    margin_left, margin_right = 70, 30
    top_gap = 100
    top_panel_h = 320
    bottom_panel_h = 300
    gap = 120

    color = "#4C72B0"
    color2 = "#55A868"

    # ===== Panel 1 (top-left): Histogram =====
    p1_x = margin_left
    p1_y = margin_top
    p1_w = (svg_w - margin_left - margin_right - gap) // 2
    p1_h = top_panel_h

    n_bins = 20
    bin_edges = _linear_bins(0, math.ceil(max(human_secs) / 5) * 5, n_bins)
    counts = _histogram(human_secs, bin_edges)
    max_count = max(counts) if counts else 1

    def x_pos_hist(val: float) -> float:
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
    for t in range(0, int(bin_edges[-1]) + 1, 5):
        tx = x_pos_hist(t)
        xtick_svg.append(
            f'<line x1="{tx:.1f}" y1="{p1_y + p1_h}" '
            f'x2="{tx:.1f}" y2="{p1_y + p1_h + 6}" stroke="#333" />'
        )
        xtick_svg.append(
            f'<text x="{tx:.1f}" y="{p1_y + p1_h + 20}" '
            f'text-anchor="middle" font-size="11" fill="#333">{t}s</text>'
        )

    # Y-axis ticks
    ytick_step = max(1, max_count // 5)
    ytick_svg = []
    for yv in range(0, max_count + 1, ytick_step):
        yy = p1_y + p1_h - (yv / max_count) * p1_h
        ytick_svg.append(
            f'<line x1="{p1_x - 6}" y1="{yy:.1f}" '
            f'x2="{p1_x}" y2="{yy:.1f}" stroke="#333" />'
        )
        ytick_svg.append(
            f'<line x1="{p1_x}" y1="{yy:.1f}" '
            f'x2="{p1_x + p1_w}" y2="{yy:.1f}" stroke="#ddd" />'
        )
        ytick_svg.append(
            f'<text x="{p1_x - 10}" y="{yy + 4:.1f}" '
            f'text-anchor="end" font-size="11" fill="#333">{yv}</text>'
        )

    # Stats box
    med = percentile(human_secs, 50)
    mean_val = statistics.mean(human_secs)
    stats_lines = [
        f"n = {len(human_secs)} tasks",
        f"min = {fmt_seconds(min(human_secs))}",
        f"median = {fmt_seconds(med)}",
        f"mean = {fmt_seconds(mean_val)}",
        f"max = {fmt_seconds(max(human_secs))}",
    ]
    stats_x = p1_x + p1_w - 10
    stats_y = p1_y + 20
    stats_svg = [
        f'<rect x="{stats_x - 155}" y="{stats_y - 14}" width="160" '
        f'height="{len(stats_lines) * 16 + 8}" '
        f'rx="4" fill="white" fill-opacity="0.85" stroke="#ccc" />'
    ]
    for idx, line in enumerate(stats_lines):
        stats_svg.append(
            f'<text x="{stats_x - 5}" y="{stats_y + idx * 16}" '
            f'text-anchor="end" font-size="10" fill="#333" '
            f'font-family="monospace">{line}</text>'
        )

    # ===== Panel 2 (top-right): Time bucket bar chart =====
    p2_x = margin_left + p1_w + gap
    p2_y = margin_top
    p2_w = p1_w
    p2_h = top_panel_h

    buckets = [
        ("0-3s", 0, 3),
        ("3-5s", 3, 5),
        ("5-10s", 5, 10),
        ("10-15s", 10, 15),
        ("15-30s", 15, 30),
        ("30s+", 30, float("inf")),
    ]
    bucket_counts = []
    bucket_labels = []
    for label, lo, hi in buckets:
        c = sum(1 for s in human_secs if lo <= s < hi)
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
            f'width="{bar_w - 2 * bar_gap:.1f}" height="{bh:.1f}" '
            f'fill="{color}" opacity="0.85" />'
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

    # Y-axis for panel 2
    ytick2_step = max(1, max_bc // 5)
    ytick2_svg = []
    for yv in range(0, max_bc + 1, ytick2_step):
        yy = p2_y + p2_h - (yv / max_bc) * p2_h
        ytick2_svg.append(
            f'<line x1="{p2_x - 6}" y1="{yy:.1f}" '
            f'x2="{p2_x}" y2="{yy:.1f}" stroke="#333" />'
        )
        ytick2_svg.append(
            f'<line x1="{p2_x}" y1="{yy:.1f}" '
            f'x2="{p2_x + p2_w}" y2="{yy:.1f}" stroke="#ddd" />'
        )
        ytick2_svg.append(
            f'<text x="{p2_x - 10}" y="{yy + 4:.1f}" '
            f'text-anchor="end" font-size="11" fill="#333">{yv}</text>'
        )

    # ===== Panel 3 (bottom): Per-family horizontal bar chart =====
    p3_x = margin_left + 160  # leave room for family labels
    p3_y = margin_top + top_panel_h + top_gap
    p3_w = svg_w - p3_x - margin_right - 20
    p3_h = bottom_panel_h

    families = sorted(task_families.keys())
    n_fam = len(families)
    fam_bar_h = min(p3_h / (n_fam + 1), 25)
    fam_gap = fam_bar_h * 0.15

    max_sec = max(max(vals) for vals in task_families.values())
    x_scale = p3_w / (max_sec * 1.1)

    fam_svg = []
    for i, fam in enumerate(families):
        vals = sorted(task_families[fam])
        med_val = percentile(vals, 50) if len(vals) > 1 else vals[0]
        by = p3_y + i * fam_bar_h

        # Range bar (min to max)
        min_v, max_v = min(vals), max(vals)
        rx = p3_x + min_v * x_scale
        rw = max((max_v - min_v) * x_scale, 2)
        fam_svg.append(
            f'<rect x="{rx:.1f}" y="{by + fam_gap:.1f}" '
            f'width="{rw:.1f}" height="{fam_bar_h - 2 * fam_gap:.1f}" '
            f'fill="{color}" opacity="0.3" rx="2" />'
        )

        # Median marker
        mx = p3_x + med_val * x_scale
        fam_svg.append(
            f'<line x1="{mx:.1f}" y1="{by + fam_gap:.1f}" '
            f'x2="{mx:.1f}" y2="{by + fam_bar_h - fam_gap:.1f}" '
            f'stroke="{color}" stroke-width="2.5" />'
        )

        # Individual task dots
        for v in vals:
            vx = p3_x + v * x_scale
            vy = by + fam_bar_h / 2
            fam_svg.append(
                f'<circle cx="{vx:.1f}" cy="{vy:.1f}" r="3" '
                f'fill="{color2}" opacity="0.7" />'
            )

        # Family label
        fam_svg.append(
            f'<text x="{p3_x - 8}" y="{by + fam_bar_h / 2 + 4:.1f}" '
            f'text-anchor="end" font-size="11" fill="#333">{fam}</text>'
        )

    # X-axis ticks for panel 3
    fam_xtick_svg = []
    x_bottom = p3_y + n_fam * fam_bar_h
    for t in range(0, int(max_sec) + 5, 5):
        tx = p3_x + t * x_scale
        if tx > p3_x + p3_w:
            break
        fam_xtick_svg.append(
            f'<line x1="{tx:.1f}" y1="{x_bottom}" '
            f'x2="{tx:.1f}" y2="{x_bottom + 6}" stroke="#333" />'
        )
        fam_xtick_svg.append(
            f'<line x1="{tx:.1f}" y1="{p3_y}" '
            f'x2="{tx:.1f}" y2="{x_bottom}" stroke="#eee" />'
        )
        fam_xtick_svg.append(
            f'<text x="{tx:.1f}" y="{x_bottom + 20}" '
            f'text-anchor="middle" font-size="11" fill="#333">{t}s</text>'
        )

    # ===== Assemble SVG =====
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" '
        f'viewBox="0 0 {svg_w} {svg_h}" style="background:white;">',
        '<style>text { font-family: Arial, Helvetica, sans-serif; }</style>',
        # Main title
        f'<text x="{svg_w / 2}" y="30" text-anchor="middle" font-size="16" '
        f'font-weight="bold" fill="#222">Distribution of Human Completion Times '
        f'for SWAA Eval Tasks</text>',
        # Panel 1 title + axes
        f'<text x="{p1_x + p1_w / 2}" y="{p1_y - 8}" text-anchor="middle" '
        f'font-size="13" fill="#333">Histogram (linear scale)</text>',
        f'<line x1="{p1_x}" y1="{p1_y + p1_h}" '
        f'x2="{p1_x + p1_w}" y2="{p1_y + p1_h}" stroke="#333" />',
        f'<line x1="{p1_x}" y1="{p1_y}" '
        f'x2="{p1_x}" y2="{p1_y + p1_h}" stroke="#333" />',
        f'<text x="{p1_x + p1_w / 2}" y="{p1_y + p1_h + 42}" '
        f'text-anchor="middle" font-size="12" fill="#333">Human completion time (seconds)</text>',
        f'<text x="{p1_x - 50}" y="{p1_y + p1_h / 2}" '
        f'text-anchor="middle" font-size="12" fill="#333" '
        f'transform="rotate(-90,{p1_x - 50},{p1_y + p1_h / 2})">Number of tasks</text>',
        # Panel 2 title + axes
        f'<text x="{p2_x + p2_w / 2}" y="{p2_y - 8}" text-anchor="middle" '
        f'font-size="13" fill="#333">Tasks by time bucket</text>',
        f'<line x1="{p2_x}" y1="{p2_y + p2_h}" '
        f'x2="{p2_x + p2_w}" y2="{p2_y + p2_h}" stroke="#333" />',
        f'<line x1="{p2_x}" y1="{p2_y}" '
        f'x2="{p2_x}" y2="{p2_y + p2_h}" stroke="#333" />',
        f'<text x="{p2_x + p2_w / 2}" y="{p2_y + p2_h + 42}" '
        f'text-anchor="middle" font-size="12" fill="#333">Time bucket</text>',
        f'<text x="{p2_x - 50}" y="{p2_y + p2_h / 2}" '
        f'text-anchor="middle" font-size="12" fill="#333" '
        f'transform="rotate(-90,{p2_x - 50},{p2_y + p2_h / 2})">Number of tasks</text>',
        # Panel 3 title + axes
        f'<text x="{svg_w / 2}" y="{p3_y - 15}" text-anchor="middle" '
        f'font-size="13" fill="#333">Human time by task family '
        f'(dots = tasks, line = median)</text>',
        f'<line x1="{p3_x}" y1="{p3_y + n_fam * fam_bar_h}" '
        f'x2="{p3_x + p3_w}" y2="{p3_y + n_fam * fam_bar_h}" stroke="#333" />',
        f'<line x1="{p3_x}" y1="{p3_y}" '
        f'x2="{p3_x}" y2="{p3_y + n_fam * fam_bar_h}" stroke="#333" />',
        f'<text x="{p3_x + p3_w / 2}" y="{p3_y + n_fam * fam_bar_h + 42}" '
        f'text-anchor="middle" font-size="12" fill="#333">Time (seconds)</text>',
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze SWAA eval human completion time distribution"
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        type=pathlib.Path,
        required=True,
        help="Path(s) to runs.jsonl file(s)",
    )
    parser.add_argument(
        "--output-file",
        type=pathlib.Path,
        default=None,
        help="Output path for the distribution plot SVG",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
    )
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper())

    all_tasks: dict[str, dict] = {}
    for f in args.input_files:
        logger.info(f"Loading SWAA tasks from {f}")
        tasks = load_swaa_human_times(f)
        for tid, info in tasks.items():
            if tid not in all_tasks:
                all_tasks[tid] = info

    # Convert to seconds for display (SWAA tasks are all sub-minute)
    human_secs = [t["human_minutes"] * 60 for t in all_tasks.values()]

    # Group by task family
    task_families: dict[str, list[float]] = {}
    for t in all_tasks.values():
        fam = t["task_family"]
        task_families.setdefault(fam, []).append(t["human_minutes"] * 60)

    logger.info(f"Total unique SWAA tasks: {len(human_secs)}")
    logger.info(f"Task families: {len(task_families)}")

    print_summary_table(human_secs, task_families)

    if args.output_file:
        generate_svg(human_secs, task_families, args.output_file)
        print(f"Distribution plot saved to: {args.output_file}")


if __name__ == "__main__":
    main()
