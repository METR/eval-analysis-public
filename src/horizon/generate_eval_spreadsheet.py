"""Generate an Excel spreadsheet with eval human completion times.

Supports any task_source (SWAA, HCAST, RE-Bench).

Sheet 1 ("Raw Timings"): One row per unique task with task_id,
    task_family, human_minutes, human_hours.
Sheet 2 ("Distribution"): Bucket breakdown, percentiles, and per-family
    summary statistics.

Uses only Python standard library (zipfile + XML) to produce a valid .xlsx
file without any external dependencies.
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_human_times(input_file: pathlib.Path, task_source: str) -> dict[str, dict]:
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
    return tasks


def percentile(data: list[float], p: float) -> float:
    s = sorted(data)
    n = len(s)
    k = (p / 100) * (n - 1)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] + (k - lo) * (s[hi] - s[lo])


def _choose_buckets_minutes(minutes_values: list[float]) -> list[tuple[str, float, float]]:
    max_val = max(minutes_values)
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
            '<numFmt numFmtId="165" formatCode="0.0"/>'
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
            col_xml += f'<col min="{i}" max="{i}" width="20" customWidth="1"/>'
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
STYLE_1DP = 3
STYLE_PCT = 4


# ---------------------------------------------------------------------------
# Sheet builders
# ---------------------------------------------------------------------------

def build_raw_timings_sheet(task_source: str, tasks: list[dict]) -> list[list[tuple]]:
    header = [
        ("s", "task_id", STYLE_BOLD),
        ("s", "task_family", STYLE_BOLD),
        ("s", "human_minutes", STYLE_BOLD),
        ("s", "human_hours", STYLE_BOLD),
    ]
    rows: list[list[tuple]] = [header]
    for t in sorted(tasks, key=lambda x: x["human_minutes"]):
        rows.append([
            ("s", t["task_id"]),
            ("s", t["task_family"]),
            ("n", round(t["human_minutes"], 4), STYLE_4DP),
            ("n", round(t["human_minutes"] / 60, 4), STYLE_4DP),
        ])
    return rows


def build_distribution_sheet(
    task_source: str,
    minutes_values: list[float],
    task_families: dict[str, list[float]],
) -> list[list[tuple]]:
    rows: list[list[tuple]] = []

    # Section 1: Summary Statistics
    rows.append([("s", f"{task_source} Summary Statistics", STYLE_BOLD)])
    rows.append([("s", "Metric", STYLE_BOLD), ("s", "Value (minutes)", STYLE_BOLD), ("s", "Value (hours)", STYLE_BOLD)])

    def stat_row(label: str, val_min: float) -> list[tuple]:
        return [("s", label), ("n", round(val_min, 2), STYLE_1DP), ("n", round(val_min / 60, 4), STYLE_4DP)]

    rows.append(stat_row("Unique tasks", len(minutes_values)))
    rows.append(stat_row("Min", min(minutes_values)))
    rows.append(stat_row("Max", max(minutes_values)))
    rows.append(stat_row("Mean", statistics.mean(minutes_values)))
    if len(minutes_values) > 1:
        rows.append(stat_row("Std dev", statistics.stdev(minutes_values)))
    for p in [10, 25, 50, 75, 90]:
        rows.append(stat_row(f"P{p}", percentile(minutes_values, p)))
    rows.append([("e",)])

    # Section 2: Bucket Breakdown
    buckets = _choose_buckets_minutes(minutes_values)
    rows.append([("s", "Bucket Breakdown", STYLE_BOLD)])
    rows.append([("s", "Bucket", STYLE_BOLD), ("s", "Count", STYLE_BOLD), ("s", "Percentage", STYLE_BOLD)])
    n = len(minutes_values)
    for label, lo, hi in buckets:
        c = sum(1 for m in minutes_values if lo <= m < hi)
        rows.append([("s", label), ("n", c), ("n", round(c / n, 3), STYLE_PCT)])
    rows.append([("e",)])

    # Section 3: Per-Family Statistics
    rows.append([("s", "Per Task Family Statistics", STYLE_BOLD)])
    rows.append([
        ("s", "Task Family", STYLE_BOLD),
        ("s", "Count", STYLE_BOLD),
        ("s", "Min (min)", STYLE_BOLD),
        ("s", "Median (min)", STYLE_BOLD),
        ("s", "Mean (min)", STYLE_BOLD),
        ("s", "Max (min)", STYLE_BOLD),
        ("s", "Std dev (min)", STYLE_BOLD),
    ])
    for fam in sorted(task_families.keys()):
        vals = task_families[fam]
        med = percentile(vals, 50) if len(vals) > 1 else vals[0]
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        rows.append([
            ("s", fam),
            ("n", len(vals)),
            ("n", round(min(vals), 2), STYLE_1DP),
            ("n", round(med, 2), STYLE_1DP),
            ("n", round(statistics.mean(vals), 2), STYLE_1DP),
            ("n", round(max(vals), 2), STYLE_1DP),
            ("n", round(std, 2), STYLE_1DP),
        ])

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Excel spreadsheet with eval human completion times"
    )
    parser.add_argument("--input-files", nargs="+", type=pathlib.Path, required=True)
    parser.add_argument("--task-source", type=str, required=True,
                        help="Task source (SWAA, HCAST, RE-Bench)")
    parser.add_argument("--output-file", type=pathlib.Path, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper())

    all_tasks: dict[str, dict] = {}
    for f in args.input_files:
        logger.info(f"Loading {args.task_source} tasks from {f}")
        tasks = load_human_times(f, args.task_source)
        for tid, info in tasks.items():
            if tid not in all_tasks:
                all_tasks[tid] = info

    if not all_tasks:
        raise SystemExit(f"No {args.task_source} tasks found")

    task_list = list(all_tasks.values())
    minutes_values = [t["human_minutes"] for t in task_list]
    task_families: dict[str, list[float]] = {}
    for t in task_list:
        task_families.setdefault(t["task_family"], []).append(t["human_minutes"])

    logger.info(f"Total unique {args.task_source} tasks: {len(task_list)}")

    wb = SimpleXlsx()
    wb.add_sheet("Raw Timings", build_raw_timings_sheet(args.task_source, task_list))
    wb.add_sheet("Distribution", build_distribution_sheet(args.task_source, minutes_values, task_families))
    wb.save(args.output_file)

    print(f"Spreadsheet saved to: {args.output_file}")
    print(f"  Sheet 1 'Raw Timings': {len(task_list)} tasks")
    print(f"  Sheet 2 'Distribution': summary stats, buckets, per-family breakdown")


if __name__ == "__main__":
    main()
