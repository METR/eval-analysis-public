from __future__ import annotations

import argparse
import pathlib
from collections import defaultdict
from typing import Any, Dict, TypedDict

import yaml


class FamilyData(TypedDict):
    summary: str
    avg_time: float
    time_source: str
    category: str
    contamination: str
    expertise_level: str


def _load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _format_time(minutes: float) -> str:
    """Format time in a readable way (minutes if <60, hours otherwise)."""
    if minutes < 60:
        return f"{minutes:.0f}m"
    return f"{minutes/60:.1f}h"


# TODO: should we take geometric mean????
def _calculate_avg_time(
    family_key: str,
    task_difficulty_data: dict[str, dict[str, Any]],
    blueprint_tasks: list[str],
) -> tuple[float, str]:
    """Calculate average time for a family from task difficulty data."""
    family_tasks = {
        task_id: data
        for task_id, data in task_difficulty_data.items()
        if task_id.startswith(f"{family_key}/")
        and task_id.split("/")[1] in blueprint_tasks
    }

    if not family_tasks:
        return 0.0, "N/A"

    times = [float(data["minutes"]) for data in family_tasks.values()]

    source_counts = {}
    for data in family_tasks.values():
        source = data.get("source", "estimate")
        source_counts[source] = source_counts.get(source, 0) + 1

    # Use most common source, defaulting to "estimate" for ties
    if source_counts:
        majority_source = max(
            source_counts.items(),
            key=lambda x: (
                x[1],
                x[0] == "estimate",
            ),  # Break ties in favor of "estimate"
        )[0]
        return sum(times) / len(times), majority_source

    return sum(times) / len(times), "estimate"


def _clean_text_for_latex(text: str) -> str:
    """Clean text for LaTeX."""
    text = text.replace("_", "\\_")
    special_chars = ["&", "%", "$", "#", "{", "}", "~", "^"]
    for char in special_chars:
        text = text.replace(char, f"\\{char}")
    text = text.replace("<", "\\textless")
    text = text.replace(">", "\\textgreater")
    return text


def _clean_family_summary_for_latex(text: str) -> str:
    # Remove code blocks, preserving text before and after
    if "```" in text:
        parts = text.split("```")
        cleaned_parts = [part.strip() for part in parts[0::2]]
        text = "\n\n".join(part for part in cleaned_parts if part)

    # Remove markdown links, keeping just the text
    while "[" in text and "](" in text and ")" in text:
        start = text.find("[")
        mid = text.find("](", start)
        end = text.find(")", mid)
        if start == -1 or mid == -1 or end == -1:
            break
        text = text[:start] + text[start + 1 : mid] + text[end + 1 :]

    # Take first paragraph only if too long (>500 chars)
    if len(text) > 500:
        first_para = text.split("\n\n")[0].strip()
        text = first_para + "..."

    return _clean_text_for_latex(text)


def _get_qualitative_label_stats(
    labels_path: pathlib.Path,
    blueprint_path: pathlib.Path,
) -> tuple[Dict[str, int], int]:
    """Get qualitative labels and count tasks per label.

    Returns:
        Tuple of (label counts dict, total number of families)
    """
    all_labels = _load_yaml(labels_path)
    blueprint = _load_yaml(blueprint_path)
    total_families = len(blueprint["task_families"])

    label_counts: Dict[str, int] = defaultdict(int)

    for family_key, _ in blueprint["task_families"].items():
        if family_key not in all_labels:
            print(f"Family {family_key} not found in family_labels.yaml")
            continue

        family_labels = all_labels[family_key]
        for label, value in family_labels.items():
            if isinstance(value, bool) and value:
                label_counts[label] += 1

    return label_counts, total_families


def _get_subdomain_stats(
    labels_path: pathlib.Path, blueprint_path: pathlib.Path
) -> tuple[
    Dict[str, Dict[str, int]],
    int,
]:
    """Analyze task family labels and count tasks per category and tag."""
    all_labels = _load_yaml(labels_path)
    benchmark_blueprint = _load_yaml(blueprint_path)

    # Map subdomains to major categories
    subdomain_to_major_category = {
        "MLE training/finetuning": "MLE",
        "MLE debugging": "MLE",
        "MLE scaffolding": "MLE",
        "Data science": "MLE",
        "Data engineering": "MLE",
        "SWE implementation": "SWE",
        "SWE debugging": "SWE",
        "Low-level algorithmic optimization": "SWE",
        "DevOps": "SWE",
        "Cybersecurity CTF": "Cybersecurity",
        "Cryptography": "Cybersecurity",
        "Cryptocurrency exploit": "Cybersecurity",
        "Common sense reasoning": "General Reasoning",
        "Algorithmic reasoning": "General Reasoning",
        "Logical reasoning": "General Reasoning",
        "Statistical reasoning": "General Reasoning",
        "Other": "General Reasoning",
    }

    category_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    labeled_families = 0

    for family, _ in benchmark_blueprint["task_families"].items():
        if family not in all_labels:
            print(f"Family {family} not found in family_labels.yaml")
            continue

        labeled_families += 1
        labels = all_labels[family]

        category = labels.get("category")
        if not category:
            continue

        major_category = subdomain_to_major_category.get(category, "Other")
        category_counts[major_category][category] += 1

    return category_counts, labeled_families


def _get_contamination_level(manifest_path: pathlib.Path) -> str:
    """Get contamination level from a manifest file."""
    if not manifest_path.exists():
        return "unknown"

    manifest_data = _load_yaml(manifest_path)
    meta = manifest_data.get("meta")
    if meta is None:
        return "unknown"

    contamination_data = meta.get("data_contamination")
    if contamination_data is None or not isinstance(contamination_data, dict):
        return "unknown"

    level = contamination_data.get("level", "unknown")
    # Merge semi_private into public_problem for consistency
    if level == "semi_private":
        level = "public_problem"

    return level


def _generate_family_table(
    task_repo_dir: pathlib.Path,
    difficulty_file: pathlib.Path,
    blueprint_file: pathlib.Path,
    labels_file: pathlib.Path,
    summaries_file: pathlib.Path,
) -> str:
    """Generate LaTeX table with family summaries and times."""
    blueprint_data = _load_yaml(blueprint_file)
    difficulty_data = _load_yaml(difficulty_file)
    summaries = _load_yaml(summaries_file)
    labels_data = _load_yaml(labels_file)
    family_data: Dict[str, FamilyData] = {}
    no_meta_families = []
    no_contamination_families = []

    for family_key, family_info in blueprint_data.get("task_families", {}).items():
        summary = summaries[family_key]
        avg_time, time_source = _calculate_avg_time(
            family_key, difficulty_data, family_info["tasks"]
        )

        # Get category and expertise level from labels
        category = "Unknown"
        expertise_level = "Unknown"
        if family_key in labels_data:
            family_labels = labels_data[family_key]
            category = family_labels.get("category", "Unknown")
            expertise_level = family_labels.get("expertise_level", "Unknown")

        # Get contamination level from manifest
        manifest_path = task_repo_dir / family_key / "manifest.yaml"
        contamination = _get_contamination_level(manifest_path)
        if contamination == "unknown":
            no_contamination_families.append(family_key)

        family_data[family_key] = {
            "summary": summary,
            "avg_time": avg_time,
            "time_source": time_source,
            "category": category,
            "contamination": contamination,
            "expertise_level": expertise_level,
        }

    if no_meta_families:
        print(f"Families with no meta: {no_meta_families}")
    if no_contamination_families:
        print(f"Families with no contamination data: {no_contamination_families}")

    # Generate LaTeX table
    latex_table_lines = [
        "\\begin{longtable}{lp{12cm}}",
        "\\toprule",
        "\\multicolumn{2}{l}{Task Family Information} \\\\",
        "\\midrule",
        "\\endhead",
        "",
        "\\midrule",
        "\\multicolumn{2}{r}{Continued on next page...} \\\\",
        "\\endfoot",
        "",
        "\\bottomrule",
        "\\endlastfoot",
        "",
    ]

    sorted_items = sorted(family_data.items(), key=lambda x: x[1]["avg_time"])
    for i, (family_key, data) in enumerate(sorted_items):
        if data["time_source"] == "no_data":
            continue

        summary = _clean_family_summary_for_latex(data["summary"])
        family_name = _clean_text_for_latex(family_key)

        # Format metadata parts
        metadata_parts = [
            f"{_format_time(data['avg_time'])} ({_clean_text_for_latex(data['time_source'])})",
            _clean_text_for_latex(data["category"]),
            _clean_text_for_latex(data["contamination"]),
            _clean_text_for_latex(data["expertise_level"]),
        ]
        metadata_line = " · ".join(f"\\textit{{{part}}}" for part in metadata_parts)

        # Add family name as full-width row
        latex_table_lines.append(f"\\multicolumn{{2}}{{c}}{{{family_name}}} \\\\")

        # Add metadata as full-width row
        latex_table_lines.append(f"\\multicolumn{{2}}{{c}}{{{metadata_line}}} \\\\")

        # Add summary
        latex_table_lines.append(f"& {summary} \\\\")

        # Add midrule between families (except after the last one)
        if i < len(sorted_items) - 1:
            latex_table_lines.append("\\midrule")

    latex_table_lines.extend(
        [
            "",
            "\\caption{Task Family Information}",
            "\\label{tab:task_family_summaries}",
            "\\end{longtable}",
        ]
    )

    return "\n".join(latex_table_lines)


def _generate_category_table(
    category_counts: Dict[str, Dict[str, int]],
    labeled_families: int,
) -> str:
    """Generate LaTeX table showing category distribution."""
    categories_table = [
        "% AUTO-GENERATED: DO NOT EDIT THIS TABLE MANUALLY",
        "% TO EDIT THIS TABLE, EDIT THE `generate_latex_tables.py` SCRIPT in `task-meta`",
        r"\begin{table}",
        r"\centering \small",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Category & Subdomain & Families & \% of families \\",
        r"\midrule",
    ]

    # Order of major categories
    major_categories = ["MLE", "SWE", "Cybersecurity", "General Reasoning"]

    for major_category in major_categories:
        subdomains = category_counts[major_category]
        if not subdomains:
            continue

        # Sort subdomains by count (descending) then name
        sorted_subdomains = sorted(subdomains.items(), key=lambda x: (-x[1], x[0]))

        # Calculate subtotal for this major category
        subtotal = sum(count for _, count in sorted_subdomains)
        subtotal_percentage = (subtotal / labeled_families) * 100
        num_subdomains = len(sorted_subdomains)

        # Add multirow entry for major category
        categories_table.append(
            f"\\multirow{{{num_subdomains}}}{{*}}{{{major_category}}}"
        )

        # Add each subdomain
        for i, (subdomain, count) in enumerate(sorted_subdomains):
            percentage = (count / labeled_families) * 100

            if i == 0:
                categories_table.append(
                    f" & {subdomain} & {count} & {percentage:.1f}\\% \\\\"
                )
            else:
                categories_table.append(
                    f"& {subdomain} & {count} & {percentage:.1f}\\% \\\\"
                )

        # Add subtotal for this category
        categories_table.extend(
            [
                r"\cmidrule{2-4}",
                f"& \\textit{{Subtotal}} & \\textit{{{subtotal}}} & \\textit{{{subtotal_percentage:.1f}\\%}} \\\\",
                r"\midrule",
            ]
        )

    # Add final total
    categories_table.extend(
        [
            f"Total & & {labeled_families} & 100.0\\% \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Distribution of task family subdomains grouped by major categories.}",
            r"\label{tab:grouped_subdomain_distribution}",
            r"\end{table}",
        ]
    )

    return "\n".join(categories_table)


def _get_contamination_counts(
    blueprint_path: pathlib.Path,
    task_repo_dir: pathlib.Path,
) -> Dict[str, tuple[int, int]]:
    """Analyze data contamination levels across families."""
    benchmark_data = _load_yaml(blueprint_path)
    contamination_counts = {
        "fully_private": (0, 0),  # (families, tasks)
        "public_problem": (0, 0),  # Will include semi_private
        "public_solution": (0, 0),
        "easy_to_memorize": (0, 0),
        "unknown": (0, 0),
    }

    for family_key, family_info in benchmark_data["task_families"].items():
        manifest_path = task_repo_dir / family_key / "manifest.yaml"
        level = _get_contamination_level(manifest_path)

        # Update counts
        family_count, task_count = contamination_counts[level]
        contamination_counts[level] = (
            family_count + 1,
            task_count + len(family_info["tasks"]),
        )

    return contamination_counts


def _generate_contamination_table(
    contamination_counts: Dict[str, tuple[int, int]],
) -> str:
    """Generate LaTeX table showing data contamination distribution."""
    # Group levels by solution privacy
    private_levels = ["fully_private", "public_problem"]
    public_levels = ["public_solution", "easy_to_memorize"]

    # Define display names
    level_names = {
        "fully_private": "Problem private",
        "public_problem": "Problem public",
        "public_solution": "Hard to memorize",
        "easy_to_memorize": "Easy to memorize",
    }

    total_families = sum(count[0] for count in contamination_counts.values())
    total_tasks = sum(count[1] for count in contamination_counts.values())

    table_lines = [
        "% AUTO-GENERATED: DO NOT EDIT THIS TABLE MANUALLY",
        "% TO EDIT THIS TABLE, EDIT THE `generate_latex_tables.py` SCRIPT in `task-meta`",
        r"\begin{table}",
        r"\centering \small",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Privacy & Level & Families & Tasks \\",
        r"\midrule",
    ]

    # Add private solution rows
    private_rows = [
        (level, contamination_counts[level])
        for level in private_levels
        if level in contamination_counts
    ]
    if private_rows:
        num_private = len(private_rows)
        table_lines.append(f"\\multirow{{{num_private}}}{{*}}{{Solution private}}")
        for i, (level, (family_count, task_count)) in enumerate(private_rows):
            if i == 0:
                table_lines.append(
                    f" & {level_names[level]} & {family_count} & {task_count} \\\\"
                )
            else:
                table_lines.append(
                    f"& {level_names[level]} & {family_count} & {task_count} \\\\"
                )

        # Calculate and add private subtotal
        private_family_total = sum(count[0] for _, count in private_rows)
        private_task_total = sum(count[1] for _, count in private_rows)
        table_lines.extend(
            [
                r"\cmidrule{2-4}",
                f"& \\textit{{Subtotal}} & \\textit{{{private_family_total}}} & \\textit{{{private_task_total}}} \\\\",
                r"\midrule",
            ]
        )

    # Add public solution rows
    public_rows = [
        (level, contamination_counts[level])
        for level in public_levels
        if level in contamination_counts
    ]
    if public_rows:
        num_public = len(public_rows)
        table_lines.append(f"\\multirow{{{num_public}}}{{*}}{{Solution public}}")
        for i, (level, (family_count, task_count)) in enumerate(public_rows):
            if i == 0:
                table_lines.append(
                    f" & {level_names[level]} & {family_count} & {task_count} \\\\"
                )
            else:
                table_lines.append(
                    f"& {level_names[level]} & {family_count} & {task_count} \\\\"
                )

        # Calculate and add public subtotal
        public_family_total = sum(count[0] for _, count in public_rows)
        public_task_total = sum(count[1] for _, count in public_rows)
        table_lines.extend(
            [
                r"\cmidrule{2-4}",
                f"& \\textit{{Subtotal}} & \\textit{{{public_family_total}}} & \\textit{{{public_task_total}}} \\\\",
            ]
        )

    table_lines.extend(
        [
            r"\midrule",
            f"Total & & {total_families} & {total_tasks} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Distribution of task families and tasks by data contamination level, grouped by solution privacy.}",
            r"\label{tab:contamination_distribution}",
            r"\end{table}",
        ]
    )

    return "\n".join(table_lines)


def _get_expertise_counts(
    labels_path: pathlib.Path,
    blueprint_path: pathlib.Path,
) -> Dict[str, int]:
    """Analyze expertise level distribution across task families."""
    all_labels = _load_yaml(labels_path)
    benchmark_blueprint = _load_yaml(blueprint_path)

    expertise_counts: Dict[str, int] = defaultdict(int)

    for family in benchmark_blueprint["task_families"]:
        if family not in all_labels:
            print(f"Family {family} not found in family_labels.yaml")
            continue

        labels = all_labels[family]
        expertise = labels.get("expertise_level", "Unknown")
        expertise_counts[expertise] += 1

    return expertise_counts


def _generate_expertise_table(expertise_counts: Dict[str, int]) -> str:
    """Generate LaTeX table showing expertise level distribution."""
    total_families = sum(expertise_counts.values())

    # Define expertise level order (from least to most experience)
    expertise_order = {
        "No experience needed": 0,
        "<1 year of experience": 1,
        "1-3 years of experience": 2,
        ">3 years of experience": 3,
        "Unknown": 4,
    }

    table_lines = [
        "% AUTO-GENERATED: DO NOT EDIT THIS TABLE MANUALLY",
        "% TO EDIT THIS TABLE, EDIT THE `generate_latex_tables.py` SCRIPT in `task-meta`",
        r"\begin{table}",
        r"\centering \small",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Experience Required & Families & \% of families \\",
        r"\midrule",
    ]

    # Sort by expertise level order
    sorted_levels = sorted(
        expertise_counts.items(),
        key=lambda x: expertise_order.get(x[0], 999),  # Unknown levels go at the end
    )

    # Add each level
    for level, count in sorted_levels:
        percentage = (count / total_families) * 100
        table_lines.append(
            f"{_clean_text_for_latex(level)} & {count} & {percentage:.1f}\\% \\\\"
        )

    # Add final total
    table_lines.extend(
        [
            r"\midrule",
            f"Total & {total_families} & 100.0\\% \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Distribution of task families by required expertise level.}",
            r"\label{tab:expertise_distribution}",
            r"\end{table}",
        ]
    )

    return "\n".join(table_lines)


def _generate_qualitative_table(
    label_counts: Dict[str, int], total_families: int
) -> str:
    """Generate LaTeX table showing distribution of qualitative labels."""
    # Sort labels by count (descending) then alphabetically
    sorted_labels = sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))

    table_lines = [
        "% AUTO-GENERATED: DO NOT EDIT THIS TABLE MANUALLY",
        "% TO EDIT THIS TABLE, EDIT THE `generate_latex_tables.py` SCRIPT in `task-meta`",
        r"\begin{table}",
        r"\centering \small",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Property & Families & \% of families \\",
        r"\midrule",
    ]

    # Add each property
    for label, count in sorted_labels:
        # Clean up label name for display
        display_label = label.replace("_", " ").replace("~", " ")
        display_label = display_label.title()
        percentage = (count / total_families) * 100

        table_lines.append(
            f"{_clean_text_for_latex(display_label)} & {count} & {percentage:.1f}\\% \\\\"
        )

    # Add final total
    table_lines.extend(
        [
            r"\midrule",
            f"Total families & {total_families} & 100.0\\% \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Distribution of qualitative properties across task families.}",
            r"\label{tab:qualitative_properties}",
            r"\end{table}",
        ]
    )

    return "\n".join(table_lines)


def main(
    task_repo_dir: pathlib.Path,
    difficulty_file: pathlib.Path,
    blueprint_file: pathlib.Path,
    labels_file: pathlib.Path,
    summaries_file: pathlib.Path,
    output_dir: pathlib.Path,
) -> None:
    """Generate all LaTeX tables."""
    output_dir.mkdir(parents=True, exist_ok=True)

    family_table = _generate_family_table(
        task_repo_dir, difficulty_file, blueprint_file, labels_file, summaries_file
    )
    with open(output_dir / "family_summary_table.tex", "w") as f:
        f.write(family_table)

    contamination_counts = _get_contamination_counts(blueprint_file, task_repo_dir)
    contamination_table = _generate_contamination_table(contamination_counts)
    with open(output_dir / "contamination_distribution_table.tex", "w") as f:
        f.write(contamination_table)

    category_counts, labeled_families = _get_subdomain_stats(
        labels_file, blueprint_file
    )
    categories_table = _generate_category_table(category_counts, labeled_families)
    with open(output_dir / "category_distribution_table.tex", "w") as f:
        f.write(categories_table)

    expertise_counts = _get_expertise_counts(labels_file, blueprint_file)
    expertise_table = _generate_expertise_table(expertise_counts)
    with open(output_dir / "expertise_distribution_table.tex", "w") as f:
        f.write(expertise_table)

    label_counts, total_families = _get_qualitative_label_stats(
        labels_file, blueprint_file
    )
    qualitative_table = _generate_qualitative_table(label_counts, total_families)
    with open(output_dir / "qualitative_properties_table.tex", "w") as f:
        f.write(qualitative_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-repo-dir", type=pathlib.Path, required=True)
    parser.add_argument("--difficulty-file", type=pathlib.Path, required=True)
    parser.add_argument("--blueprint-file", type=pathlib.Path, required=True)
    parser.add_argument("--labels-file", type=pathlib.Path, required=True)
    parser.add_argument("--summaries-file", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = vars(parser.parse_args())
    main(**args)
