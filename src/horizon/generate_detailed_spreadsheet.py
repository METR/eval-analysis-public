"""Generate detailed Excel spreadsheet with per-task, per-model scores.

Sheet 1 ("All Runs"): One row per run with task_id, task_family, human_minutes,
    human_hours, alias (model), score_binarized, score_cont.
Sheet 2 ("Task Summary"): Aggregated per-task stats (human time, success rate per model).
Sheet 3 ("Distribution"): Bucket breakdown, percentiles, per-family stats.

Uses only Python standard library.
"""

import argparse
import json
import logging
import math
import pathlib
import statistics
import zipfile
from xml.sax.saxutils import escape

logger = logging.getLogger(__name__)


def load_all_runs(input_file: pathlib.Path, task_source: str) -> list[dict]:
    """Load all runs for a given task_source."""
    runs = []
    with open(input_file) as f:
        for line in f:
            row = json.loads(line)
            if row.get("task_source") != task_source:
                continue
            runs.append({
                "task_id": row["task_id"],
                "task_family": row.get("task_family", ""),
                "human_minutes": row["human_minutes"],
                "alias": row.get("alias", ""),
                "model": row.get("model", ""),
                "score_binarized": row.get("score_binarized", 0),
                "score_cont": row.get("score_cont", 0.0),
                "run_id": row.get("run_id", ""),
            })
    return runs


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    n = len(s)
    k = (p / 100) * (n - 1)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] + (k - lo) * (s[hi] - s[lo])


def _choose_buckets_minutes(minutes_values: list[float]) -> list[tuple[str, float, float]]:
    max_val = max(minutes_values) if minutes_values else 1
    if max_val < 1:
        return [
            ("0-3s", 0, 3 / 60), ("3-5s", 3 / 60, 5 / 60),
            ("5-10s", 5 / 60, 10 / 60), ("10-15s", 10 / 60, 15 / 60),
            ("15-30s", 15 / 60, 30 / 60), ("30s+", 30 / 60, float("inf")),
        ]
    if max_val < 120:
        return [
            ("< 5 min", 0, 5), ("5-15 min", 5, 15), ("15-30 min", 15, 30),
            ("30-60 min", 30, 60), ("1-2 hr", 60, 120),
        ]
    return [
        ("< 15 min", 0, 15), ("15-60 min", 15, 60), ("1-2 hr", 60, 120),
        ("2-4 hr", 120, 240), ("4-8 hr", 240, 480), ("8-16 hr", 480, 960),
        ("16+ hr", 960, float("inf")),
    ]


# ---------------------------------------------------------------------------
# Minimal .xlsx writer
# ---------------------------------------------------------------------------

def _cell_ref(col: int, row: int) -> str:
    letters = ""
    c = col
    while True:
        letters = chr(ord("A") + c % 26) + letters
        c = c // 26 - 1
        if c < 0:
            break
    return f"{letters}{row + 1}"


class SimpleXlsx:
    def __init__(self) -> None:
        self._shared_strings: list[str] = []
        self._ss_index: dict[str, int] = {}
        self._sheets: list[tuple[str, list[list[tuple]]]] = []

    def _get_ssi(self, text: str) -> int:
        if text not in self._ss_index:
            self._ss_index[text] = len(self._shared_strings)
            self._shared_strings.append(text)
        return self._ss_index[text]

    def add_sheet(self, name: str, rows: list[list[tuple]]) -> None:
        self._sheets.append((name, rows))

    def save(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", self._content_types())
            zf.writestr("_rels/.rels", self._rels())
            zf.writestr("xl/workbook.xml", self._workbook())
            zf.writestr("xl/_rels/workbook.xml.rels", self._workbook_rels())
            zf.writestr("xl/styles.xml", self._styles())
            for i, (_, rows) in enumerate(self._sheets, 1):
                zf.writestr(f"xl/worksheets/sheet{i}.xml", self._sheet_xml(rows))
            zf.writestr("xl/sharedStrings.xml", self._shared_strings_xml())

    def _content_types(self) -> str:
        sheets = "".join(
            f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
            f'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            for i in range(1, len(self._sheets) + 1)
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            f'{sheets}'
            '<Override PartName="/xl/sharedStrings.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
            '<Override PartName="/xl/styles.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            '</Types>'
        )

    def _rels(self) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '</Relationships>'
        )

    def _workbook(self) -> str:
        sheets_xml = "".join(
            f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
            for i, (name, _) in enumerate(self._sheets, 1)
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f'<sheets>{sheets_xml}</sheets></workbook>'
        )

    def _workbook_rels(self) -> str:
        rels = "".join(
            f'<Relationship Id="rId{i}" '
            f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{i}.xml"/>'
            for i in range(1, len(self._sheets) + 1)
        )
        ss_id = len(self._sheets) + 1
        st_id = len(self._sheets) + 2
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'{rels}'
            f'<Relationship Id="rId{ss_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>'
            f'<Relationship Id="rId{st_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            '</Relationships>'
        )

    def _styles(self) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<numFmts count="3">'
            '<numFmt numFmtId="164" formatCode="0.0000"/>'
            '<numFmt numFmtId="165" formatCode="0.00"/>'
            '<numFmt numFmtId="166" formatCode="0.0%"/>'
            '</numFmts>'
            '<fonts count="2">'
            '<font><sz val="11"/><name val="Calibri"/></font>'
            '<font><b/><sz val="11"/><name val="Calibri"/></font>'
            '</fonts>'
            '<fills count="2"><fill><patternFill patternType="none"/></fill>'
            '<fill><patternFill patternType="gray125"/></fill></fills>'
            '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="5">'
            '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
            '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>'
            '<xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
            '<xf numFmtId="165" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
            '<xf numFmtId="166" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
            '</cellXfs></styleSheet>'
        )

    def _sheet_xml(self, rows: list[list[tuple]]) -> str:
        rows_xml = []
        for r, row in enumerate(rows):
            cells_xml = []
            for c, cell in enumerate(row):
                if cell[0] == "s":
                    ssi = self._get_ssi(str(cell[1]))
                    style = cell[2] if len(cell) > 2 else 0
                    ref = _cell_ref(c, r)
                    cells_xml.append(f'<c r="{ref}" t="s" s="{style}"><v>{ssi}</v></c>')
                elif cell[0] == "n":
                    style = cell[2] if len(cell) > 2 else 0
                    ref = _cell_ref(c, r)
                    cells_xml.append(f'<c r="{ref}" s="{style}"><v>{cell[1]}</v></c>')
                elif cell[0] == "e":
                    pass
            rows_xml.append(f'<row r="{r + 1}">{"".join(cells_xml)}</row>')
        col_xml = '<cols>'
        max_cols = max((len(row) for row in rows), default=1)
        for i in range(1, max_cols + 1):
            w = 25 if i <= 2 else 15
            col_xml += f'<col min="{i}" max="{i}" width="{w}" customWidth="1"/>'
        col_xml += '</cols>'
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'{col_xml}<sheetData>{"".join(rows_xml)}</sheetData></worksheet>'
        )

    def _shared_strings_xml(self) -> str:
        count = len(self._shared_strings)
        items = "".join(f"<si><t>{escape(s)}</t></si>" for s in self._shared_strings)
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            f'count="{count}" uniqueCount="{count}">{items}</sst>'
        )


STYLE_BOLD = 1
STYLE_4DP = 2
STYLE_2DP = 3
STYLE_PCT = 4


# ---------------------------------------------------------------------------
# Sheet builders
# ---------------------------------------------------------------------------

def build_all_runs_sheet(runs: list[dict]) -> list[list[tuple]]:
    """Sheet 1: All individual runs."""
    header = [
        ("s", "task_id", STYLE_BOLD),
        ("s", "task_family", STYLE_BOLD),
        ("s", "alias", STYLE_BOLD),
        ("s", "model", STYLE_BOLD),
        ("s", "human_minutes", STYLE_BOLD),
        ("s", "human_hours", STYLE_BOLD),
        ("s", "score_binarized", STYLE_BOLD),
        ("s", "score_cont", STYLE_BOLD),
        ("s", "run_id", STYLE_BOLD),
    ]
    rows: list[list[tuple]] = [header]
    for r in sorted(runs, key=lambda x: (x["task_id"], x["alias"])):
        rows.append([
            ("s", r["task_id"]),
            ("s", r["task_family"]),
            ("s", r["alias"]),
            ("s", r["model"]),
            ("n", round(r["human_minutes"], 4), STYLE_4DP),
            ("n", round(r["human_minutes"] / 60, 4), STYLE_4DP),
            ("n", r["score_binarized"]),
            ("n", round(r["score_cont"], 4), STYLE_4DP),
            ("s", r["run_id"]),
        ])
    return rows


def build_task_summary_sheet(runs: list[dict]) -> list[list[tuple]]:
    """Sheet 2: Per-task summary with per-model success rates."""
    # Group by task
    tasks: dict[str, dict] = {}
    for r in runs:
        tid = r["task_id"]
        if tid not in tasks:
            tasks[tid] = {
                "task_family": r["task_family"],
                "human_minutes": r["human_minutes"],
                "models": {},
            }
        alias = r["alias"]
        if alias not in tasks[tid]["models"]:
            tasks[tid]["models"][alias] = {"successes": 0, "total": 0}
        tasks[tid]["models"][alias]["total"] += 1
        tasks[tid]["models"][alias]["successes"] += r["score_binarized"]

    # Get all unique models
    all_models = sorted(set(r["alias"] for r in runs))

    # Build header
    header = [
        ("s", "task_id", STYLE_BOLD),
        ("s", "task_family", STYLE_BOLD),
        ("s", "human_minutes", STYLE_BOLD),
        ("s", "human_hours", STYLE_BOLD),
        ("s", "n_runs", STYLE_BOLD),
        ("s", "overall_success_rate", STYLE_BOLD),
    ]
    for m in all_models:
        header.append(("s", f"{m} (success_rate)", STYLE_BOLD))

    rows: list[list[tuple]] = [header]

    for tid in sorted(tasks.keys()):
        t = tasks[tid]
        total_runs = sum(m["total"] for m in t["models"].values())
        total_successes = sum(m["successes"] for m in t["models"].values())
        overall_rate = total_successes / total_runs if total_runs > 0 else 0

        row = [
            ("s", tid),
            ("s", t["task_family"]),
            ("n", round(t["human_minutes"], 2), STYLE_2DP),
            ("n", round(t["human_minutes"] / 60, 4), STYLE_4DP),
            ("n", total_runs),
            ("n", round(overall_rate, 3), STYLE_PCT),
        ]
        for m in all_models:
            if m in t["models"]:
                md = t["models"][m]
                rate = md["successes"] / md["total"] if md["total"] > 0 else 0
                row.append(("n", round(rate, 3), STYLE_PCT))
            else:
                row.append(("s", "-"))
        rows.append(row)

    return rows


def build_distribution_sheet(
    task_source: str,
    runs: list[dict],
) -> list[list[tuple]]:
    """Sheet 3: Distribution summary."""
    # Get unique task minutes
    tasks: dict[str, float] = {}
    task_families: dict[str, list[float]] = {}
    for r in runs:
        tid = r["task_id"]
        if tid not in tasks:
            tasks[tid] = r["human_minutes"]
            fam = r["task_family"]
            task_families.setdefault(fam, []).append(r["human_minutes"])

    minutes_values = list(tasks.values())

    rows: list[list[tuple]] = []

    # Section 1: Summary Statistics
    rows.append([("s", f"{task_source} Summary Statistics", STYLE_BOLD)])
    rows.append([("s", "Metric", STYLE_BOLD), ("s", "Value", STYLE_BOLD)])
    rows.append([("s", "Unique tasks"), ("n", len(minutes_values))])
    rows.append([("s", "Total runs"), ("n", len(runs))])
    rows.append([("s", "Min (min)"), ("n", round(min(minutes_values), 2), STYLE_2DP)])
    rows.append([("s", "Max (min)"), ("n", round(max(minutes_values), 2), STYLE_2DP)])
    rows.append([("s", "Mean (min)"), ("n", round(statistics.mean(minutes_values), 2), STYLE_2DP)])
    if len(minutes_values) > 1:
        rows.append([("s", "Std dev (min)"), ("n", round(statistics.stdev(minutes_values), 2), STYLE_2DP)])
    for p in [10, 25, 50, 75, 90]:
        rows.append([("s", f"P{p} (min)"), ("n", round(percentile(minutes_values, p), 2), STYLE_2DP)])
    rows.append([("e",)])

    # Section 2: Bucket Breakdown
    buckets = _choose_buckets_minutes(minutes_values)
    rows.append([("s", "Bucket Breakdown (by unique tasks)", STYLE_BOLD)])
    rows.append([("s", "Bucket", STYLE_BOLD), ("s", "Count", STYLE_BOLD), ("s", "Percentage", STYLE_BOLD)])
    n = len(minutes_values)
    for label, lo, hi in buckets:
        c = sum(1 for m in minutes_values if lo <= m < hi)
        rows.append([("s", label), ("n", c), ("n", round(c / n, 3), STYLE_PCT)])
    rows.append([("e",)])

    # Section 3: Per-Model Stats
    model_stats: dict[str, dict] = {}
    for r in runs:
        alias = r["alias"]
        if alias not in model_stats:
            model_stats[alias] = {"runs": 0, "successes": 0}
        model_stats[alias]["runs"] += 1
        model_stats[alias]["successes"] += r["score_binarized"]

    rows.append([("s", "Per-Model Statistics", STYLE_BOLD)])
    rows.append([
        ("s", "Model (alias)", STYLE_BOLD),
        ("s", "Runs", STYLE_BOLD),
        ("s", "Successes", STYLE_BOLD),
        ("s", "Success Rate", STYLE_BOLD),
    ])
    for m in sorted(model_stats.keys()):
        ms = model_stats[m]
        rate = ms["successes"] / ms["runs"] if ms["runs"] > 0 else 0
        rows.append([
            ("s", m),
            ("n", ms["runs"]),
            ("n", ms["successes"]),
            ("n", round(rate, 3), STYLE_PCT),
        ])
    rows.append([("e",)])

    # Section 4: Per-Family Statistics
    rows.append([("s", "Per Task Family Statistics", STYLE_BOLD)])
    rows.append([
        ("s", "Task Family", STYLE_BOLD),
        ("s", "Tasks", STYLE_BOLD),
        ("s", "Min (min)", STYLE_BOLD),
        ("s", "Median (min)", STYLE_BOLD),
        ("s", "Max (min)", STYLE_BOLD),
    ])
    for fam in sorted(task_families.keys()):
        vals = task_families[fam]
        med = percentile(vals, 50) if len(vals) > 1 else vals[0]
        rows.append([
            ("s", fam),
            ("n", len(vals)),
            ("n", round(min(vals), 2), STYLE_2DP),
            ("n", round(med, 2), STYLE_2DP),
            ("n", round(max(vals), 2), STYLE_2DP),
        ])

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate detailed Excel spreadsheet with per-task, per-model scores"
    )
    parser.add_argument("--input-files", nargs="+", type=pathlib.Path, required=True)
    parser.add_argument("--task-source", type=str, required=True,
                        help="Task source (SWAA, HCAST, RE-Bench)")
    parser.add_argument("--output-file", type=pathlib.Path, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper())

    all_runs: list[dict] = []
    seen_run_ids: set[str] = set()
    for f in args.input_files:
        logger.info(f"Loading {args.task_source} runs from {f}")
        runs = load_all_runs(f, args.task_source)
        for r in runs:
            rid = r["run_id"]
            if rid and rid not in seen_run_ids:
                seen_run_ids.add(rid)
                all_runs.append(r)
            elif not rid:
                all_runs.append(r)

    if not all_runs:
        raise SystemExit(f"No {args.task_source} runs found")

    unique_tasks = len(set(r["task_id"] for r in all_runs))
    unique_models = len(set(r["alias"] for r in all_runs))
    logger.info(f"Total runs: {len(all_runs)}, unique tasks: {unique_tasks}, models: {unique_models}")

    wb = SimpleXlsx()
    wb.add_sheet("All Runs", build_all_runs_sheet(all_runs))
    wb.add_sheet("Task Summary", build_task_summary_sheet(all_runs))
    wb.add_sheet("Distribution", build_distribution_sheet(args.task_source, all_runs))
    wb.save(args.output_file)

    print(f"Spreadsheet saved to: {args.output_file}")
    print(f"  Sheet 1 'All Runs': {len(all_runs)} runs")
    print(f"  Sheet 2 'Task Summary': {unique_tasks} tasks x {unique_models} models")
    print(f"  Sheet 3 'Distribution': summary stats, buckets, per-model & per-family breakdown")


if __name__ == "__main__":
    main()
