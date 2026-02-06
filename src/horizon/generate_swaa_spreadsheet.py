"""Generate an Excel spreadsheet with SWAA eval human completion times.

Sheet 1 ("Raw Timings"): One row per unique SWAA task with task_id,
    task_family, human_minutes, human_seconds.
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
# Data loading (reused from swaa_eval_time_analysis.py)
# ---------------------------------------------------------------------------

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


def percentile(data: list[float], p: float) -> float:
    s = sorted(data)
    n = len(s)
    k = (p / 100) * (n - 1)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] + (k - lo) * (s[hi] - s[lo])


# ---------------------------------------------------------------------------
# Minimal .xlsx writer using only stdlib
# ---------------------------------------------------------------------------

def _cell_ref(col: int, row: int) -> str:
    """Convert 0-based col, row to Excel cell reference like A1, B2, etc."""
    letters = ""
    c = col
    while True:
        letters = chr(ord("A") + c % 26) + letters
        c = c // 26 - 1
        if c < 0:
            break
    return f"{letters}{row + 1}"


def _string_cell(col: int, row: int, ssi: int) -> str:
    ref = _cell_ref(col, row)
    return f'<c r="{ref}" t="s"><v>{ssi}</v></c>'


def _number_cell(col: int, row: int, value: float, style: int = 0) -> str:
    ref = _cell_ref(col, row)
    s_attr = f' s="{style}"' if style else ""
    return f'<c r="{ref}"{s_attr}><v>{value}</v></c>'


class SimpleXlsx:
    """Minimal xlsx writer — supports strings and numbers, multiple sheets,
    bold headers, column widths, and number formats."""

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
        """Add a sheet. Each cell is a tuple: ('s', text) or ('n', number)
        or ('n', number, style_id)."""
        self._sheets.append((name, rows))

    def save(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", self._content_types())
            zf.writestr("_rels/.rels", self._rels())
            zf.writestr("xl/workbook.xml", self._workbook())
            zf.writestr("xl/_rels/workbook.xml.rels", self._workbook_rels())
            zf.writestr("xl/styles.xml", self._styles())
            for i, (name, rows) in enumerate(self._sheets, 1):
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
            f'<Relationship Id="rId{ss_id}" '
            f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" '
            f'Target="sharedStrings.xml"/>'
            f'<Relationship Id="rId{st_id}" '
            f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
            f'Target="styles.xml"/>'
            '</Relationships>'
        )

    def _styles(self) -> str:
        # Style 0: default
        # Style 1: bold (for headers)
        # Style 2: number with 4 decimal places
        # Style 3: number with 1 decimal place
        # Style 4: percentage with 1 decimal place
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
            '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'              # 0: default
            '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>'  # 1: bold
            '<xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'  # 2: 4dp
            '<xf numFmtId="165" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'  # 3: 1dp
            '<xf numFmtId="166" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'  # 4: pct
            '</cellXfs>'
            '</styleSheet>'
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
                    # empty cell
                    pass
            rows_xml.append(f'<row r="{r + 1}">{"".join(cells_xml)}</row>')

        # Column widths
        col_xml = '<cols>'
        max_cols = max((len(row) for row in rows), default=1)
        for i in range(1, max_cols + 1):
            col_xml += f'<col min="{i}" max="{i}" width="18" customWidth="1"/>'
        col_xml += '</cols>'

        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'{col_xml}'
            f'<sheetData>{"".join(rows_xml)}</sheetData>'
            '</worksheet>'
        )

    def _shared_strings_xml(self) -> str:
        count = len(self._shared_strings)
        items = "".join(
            f"<si><t>{escape(s)}</t></si>" for s in self._shared_strings
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            f'count="{count}" uniqueCount="{count}">{items}</sst>'
        )


# ---------------------------------------------------------------------------
# Spreadsheet generation
# ---------------------------------------------------------------------------

STYLE_BOLD = 1
STYLE_4DP = 2
STYLE_1DP = 3
STYLE_PCT = 4


def build_raw_timings_sheet(tasks: list[dict]) -> list[list[tuple]]:
    """Sheet 1: one row per task with raw timing data."""
    header = [
        ("s", "task_id", STYLE_BOLD),
        ("s", "task_family", STYLE_BOLD),
        ("s", "human_minutes", STYLE_BOLD),
        ("s", "human_seconds", STYLE_BOLD),
    ]
    rows: list[list[tuple]] = [header]
    for t in sorted(tasks, key=lambda x: x["human_minutes"]):
        rows.append([
            ("s", t["task_id"]),
            ("s", t["task_family"]),
            ("n", round(t["human_minutes"], 6), STYLE_4DP),
            ("n", round(t["human_minutes"] * 60, 1), STYLE_1DP),
        ])
    return rows


def build_distribution_sheet(
    human_secs: list[float],
    task_families: dict[str, list[float]],
) -> list[list[tuple]]:
    """Sheet 2: distribution summary with buckets, percentiles, and family stats."""
    rows: list[list[tuple]] = []

    # --- Section 1: Summary Statistics ---
    rows.append([("s", "Summary Statistics", STYLE_BOLD)])
    rows.append([("s", "Metric", STYLE_BOLD), ("s", "Value", STYLE_BOLD)])
    stats = [
        ("Unique tasks", len(human_secs)),
        ("Min (seconds)", round(min(human_secs), 1)),
        ("Max (seconds)", round(max(human_secs), 1)),
        ("Mean (seconds)", round(statistics.mean(human_secs), 1)),
        ("Std dev (seconds)", round(statistics.stdev(human_secs), 1)),
        ("P10 (seconds)", round(percentile(human_secs, 10), 1)),
        ("P25 (seconds)", round(percentile(human_secs, 25), 1)),
        ("P50 / Median (seconds)", round(percentile(human_secs, 50), 1)),
        ("P75 (seconds)", round(percentile(human_secs, 75), 1)),
        ("P90 (seconds)", round(percentile(human_secs, 90), 1)),
    ]
    for label, val in stats:
        rows.append([("s", label), ("n", val, STYLE_1DP)])

    rows.append([("e",)])  # blank row

    # --- Section 2: Bucket Breakdown ---
    rows.append([("s", "Bucket Breakdown", STYLE_BOLD)])
    rows.append([
        ("s", "Bucket", STYLE_BOLD),
        ("s", "Count", STYLE_BOLD),
        ("s", "Percentage", STYLE_BOLD),
    ])
    buckets = [
        ("0-3s", 0, 3),
        ("3-5s", 3, 5),
        ("5-10s", 5, 10),
        ("10-15s", 10, 15),
        ("15-30s", 15, 30),
        ("30s+", 30, float("inf")),
    ]
    n = len(human_secs)
    for label, lo, hi in buckets:
        c = sum(1 for s in human_secs if lo <= s < hi)
        rows.append([
            ("s", label),
            ("n", c),
            ("n", round(c / n, 3), STYLE_PCT),
        ])

    rows.append([("e",)])  # blank row

    # --- Section 3: Per-Family Statistics ---
    rows.append([("s", "Per Task Family Statistics", STYLE_BOLD)])
    rows.append([
        ("s", "Task Family", STYLE_BOLD),
        ("s", "Count", STYLE_BOLD),
        ("s", "Min (s)", STYLE_BOLD),
        ("s", "Median (s)", STYLE_BOLD),
        ("s", "Mean (s)", STYLE_BOLD),
        ("s", "Max (s)", STYLE_BOLD),
        ("s", "Std dev (s)", STYLE_BOLD),
    ])
    for fam in sorted(task_families.keys()):
        vals = task_families[fam]
        med = percentile(vals, 50) if len(vals) > 1 else vals[0]
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        rows.append([
            ("s", fam),
            ("n", len(vals)),
            ("n", round(min(vals), 1), STYLE_1DP),
            ("n", round(med, 1), STYLE_1DP),
            ("n", round(statistics.mean(vals), 1), STYLE_1DP),
            ("n", round(max(vals), 1), STYLE_1DP),
            ("n", round(std, 1), STYLE_1DP),
        ])

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Excel spreadsheet with SWAA eval human completion times"
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
        required=True,
        help="Output path for the .xlsx file",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper())

    # Load and deduplicate
    all_tasks: dict[str, dict] = {}
    for f in args.input_files:
        logger.info(f"Loading SWAA tasks from {f}")
        tasks = load_swaa_human_times(f)
        for tid, info in tasks.items():
            if tid not in all_tasks:
                all_tasks[tid] = info

    task_list = list(all_tasks.values())
    human_secs = [t["human_minutes"] * 60 for t in task_list]

    task_families: dict[str, list[float]] = {}
    for t in task_list:
        task_families.setdefault(t["task_family"], []).append(t["human_minutes"] * 60)

    logger.info(f"Total unique SWAA tasks: {len(task_list)}")

    # Build spreadsheet
    wb = SimpleXlsx()
    wb.add_sheet("Raw Timings", build_raw_timings_sheet(task_list))
    wb.add_sheet("Distribution", build_distribution_sheet(human_secs, task_families))
    wb.save(args.output_file)

    print(f"Spreadsheet saved to: {args.output_file}")
    print(f"  Sheet 1 'Raw Timings': {len(task_list)} tasks")
    print(f"  Sheet 2 'Distribution': summary stats, buckets, per-family breakdown")


if __name__ == "__main__":
    main()
