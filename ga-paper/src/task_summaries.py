import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from anthropic import Anthropic

API_KEY = os.environ.get("ANTHROPIC_API_KEY")

PROMPT_TEMPLATE = """
I will provide you information about a task. This task is part of a benchmark intended to evaluate the autonomous/agentic capabilities of
advanced AI systems. I need you to summarize the task in a single sentence. The summary will go into a paper describing the benchmark.
There are many tasks, so your summary should just highlight the core goal or requirement of the task, and what is required to complete it.

# Examples
Here are a couple examples of the information you will be provided, and the summaries you should generate.

{few_shot_examples}

# Task Family Information
Now, here is the information for the task family you are summarizing.

## Manifest
{manifest_block}

## Module Code
```python
{module_code}
```

## Metadata
{metadata_section}

# Final Instructions
Focus on making this summary clear and useful for someone who has no familiarity with the task family, and just needs a single sentence describing the task, and what is required to complete it.

Phrase your summary without any preamble---e.g. without saying "The core goal of the task is...", "The task requires...", or anything like that. Just give the summary. It should typically start with a verb.
"""

SYSTEM_PROMPT = """You are an expert at analyzing and summarizing code and technical documentation. Provide a single sentence summary that captures the core goal or requirement of the task family in a single sentence."""


@dataclass
class TaskFamilySummary:
    """Represents a task family with its metadata, manifest, and module code."""

    name: str
    task_repo_dir: Path = Path("/tmp/tasks")
    manifest: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    module_code: str = ""
    summary: str = ""
    max_chars: int = 10000

    def _truncate_str(self, text: str) -> str:
        """Truncate string to max_chars if needed."""
        if len(text) > self.max_chars:
            return (
                text[: self.max_chars]
                + f"\n... [truncated, {len(text) - self.max_chars} more characters]"
            )
        return text

    def load_data(self) -> None:
        """Load all data for the task family and truncate to max_chars."""
        family_dir = Path(self.task_repo_dir) / self.name

        manifest_path = family_dir / "manifest.yaml"
        with manifest_path.open("r") as f:
            self.manifest = self._truncate_str(f.read())

        module_path = family_dir / f"{self.name}.py"
        with module_path.open("r") as f:
            self.module_code = self._truncate_str(f.read())

        meta_dir = family_dir / "meta"
        self.metadata = {}
        if meta_dir.exists():
            for file_path in meta_dir.iterdir():
                key = file_path.stem
                if file_path.suffix == ".txt" or file_path.suffix == ".md":
                    with file_path.open("r") as f:
                        self.metadata[key] = self._truncate_str(f.read())


def _discover_families(task_repo_dir: Path) -> List[str]:
    """Discover all task families in the base directory, defined as
    directories with a manifest.yaml file and a module file.
    """
    if not task_repo_dir.exists():
        return []

    family_keys = []
    for item_path in task_repo_dir.iterdir():
        if item_path.is_dir():
            manifest_exists = (item_path / "manifest.yaml").exists()
            module_exists = (item_path / f"{item_path.name}.py").exists()

            if manifest_exists and module_exists:
                family_keys.append(item_path.name)

    return family_keys


def _get_claude_summary(
    task_family: TaskFamilySummary,
    few_shot_examples: Optional[str] = None,
    model: str = "claude-3-7-sonnet-20250219",
) -> str:
    """Generate a summary of the task family using Claude."""

    prompt = PROMPT_TEMPLATE.format(
        manifest_block=task_family.manifest,
        module_code=task_family.module_code,
        metadata_section=task_family.metadata,
        few_shot_examples=few_shot_examples or "",
    )

    client = Anthropic(api_key=API_KEY)

    message = client.messages.create(
        model=model,
        max_tokens=4000,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    summary_text = ""
    for content_block in message.content:
        if content_block.type == "text":
            summary_text += content_block.text

    task_family.summary = summary_text or ""
    return task_family.summary


def _summarize_task_families(
    task_repo_dir: Path,
    output_path: Path,
    family_key: Optional[str] = None,
    model: str = "claude-3-7-sonnet-20250219",
    few_shot_examples: Optional[str] = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Dict[str, str]:
    """Generate summaries for task families.

    Handles loading existing summaries, generating new ones using Claude API,
    and saving the results to the specified output path. Supports single family
    or batch processing modes with options for overwriting and dry runs.

    Args:
        task_repo_dir: Path to the repository containing task families.
        output_path: Path where the generated summaries will be saved as YAML.
        family_key: Optional key of a specific task family to summarize. If None, summarizes all families.
        model: The Claude model identifier to use for generating summaries.
        few_shot_examples: Optional string containing few-shot examples of summaries.
        overwrite: Whether to regenerate summaries for families that already have them.
        dry_run: If True, returns the prompt that would be sent to Claude instead of generating summaries.

    Returns:
        Dictionary mapping family names to their summaries, or prompt details if dry_run is True.
    """
    existing_summaries = {}
    if output_path.exists():
        with output_path.open("r") as f:
            existing_summaries = yaml.safe_load(f)

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if family_key:
        task_family = TaskFamilySummary(name=family_key, task_repo_dir=task_repo_dir)
        task_family.load_data()

        if dry_run:
            prompt = PROMPT_TEMPLATE.format(
                manifest_block=task_family.manifest,
                module_code=task_family.module_code,
                metadata_section=task_family.metadata,
                few_shot_examples=few_shot_examples or "",
            )
            return {"system_prompt": SYSTEM_PROMPT, "user_prompt": prompt}

        if family_key in existing_summaries and not overwrite:
            return {family_key: existing_summaries[family_key]}

        summary = _get_claude_summary(task_family, few_shot_examples, model)
        existing_summaries[family_key] = summary

        with output_path.open("w") as f:
            yaml.dump(existing_summaries, f)
    else:
        if dry_run:
            raise ValueError(
                "Dry run requires specifying a single family with --family"
            )

        family_keys = _discover_families(task_repo_dir)

        for family_key in family_keys:
            if family_key in existing_summaries and not overwrite:
                continue

            task_family = TaskFamilySummary(
                name=family_key, task_repo_dir=task_repo_dir
            )
            task_family.load_data()
            summary = _get_claude_summary(task_family, few_shot_examples, model)
            existing_summaries[family_key] = summary

            with output_path.open("w") as f:
                yaml.dump(existing_summaries, f)

    print(f"Completed with {len(existing_summaries)} summaries in {output_path}")

    return existing_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize task families using Claude")
    parser.add_argument(
        "--task-repo-dir",
        type=Path,
        default="/tmp/tasks",
        help="Base directory where task families are stored",
    )
    parser.add_argument("--family", help="Specific task family to summarize")
    parser.add_argument(
        "--output-path",
        type=Path,
        default="task_summaries.yaml",
        help="YAML file to save summaries",
    )
    parser.add_argument(
        "--list-potential-families",
        action="store_true",
        help="List available task families",
    )
    parser.add_argument(
        "--few-shot-examples",
        type=Path,
        help="Path to a file containing few-shot examples",
    )
    parser.add_argument(
        "--model", default="claude-3-7-sonnet-20250219", help="Claude model to use"
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Only output the prompt without making API calls",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing summaries"
    )

    args = parser.parse_args()

    few_shot_examples = None
    if args.few_shot_examples:
        few_shot_examples = args.few_shot_examples.read_text()

    if args.list_potential_families:
        families = _discover_families(args.task_repo_dir)
        if families:
            print("Available task families:")
            for family in families:
                print(f"  - {family}")
        else:
            print("No task families found.")
        return

    result = _summarize_task_families(
        task_repo_dir=args.task_repo_dir,
        output_path=args.output_path,
        family_key=args.family,
        model=args.model,
        few_shot_examples=few_shot_examples,
        overwrite=args.overwrite,
        dry_run=args.dry,
    )

    if args.dry:
        print("\n=== SYSTEM PROMPT ===")
        print(result["system_prompt"])
        print("\n=== USER PROMPT ===")
        print(result["user_prompt"])
    elif args.family:
        print(f"Summary for {args.family}: {result[args.family]}")
        print(f"Saved to: {args.output_path}")
    else:
        print(f"Summaries saved to: {args.output_path}")


if __name__ == "__main__":
    main()
