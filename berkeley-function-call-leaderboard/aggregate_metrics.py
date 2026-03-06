from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

CATEGORY_MAPPING: Dict[str, List[str]] = {
    "Simple": ["simple", "java", "javascript"],
    "Multiple": ["multiple"],
    "Simple (User-Contributed)": ["live_simple"],
    "Multiple (User-Contributed)": ["live_multiple"],
    "Relevance": ["live_relevance"],
    "Irrelevance": ["irrelevance", "live_irrelevance"],
    "Multi-Turn": [
        "multi_turn_base",
        "multi_turn_miss_func",
        "multi_turn_miss_param",
        "multi_turn_long_context",
    ],
}

DESIRED_ORDER = list(CATEGORY_MAPPING.keys())
AVG_COL_NAME = "High-Level Avg Accuracy, %"


def extract_test_category(input_string: Union[str, Path]) -> str:
    input_string = str(input_string)
    pattern = r".*BFCL_v3_(\w+?)(?:_score|_result)?\.json"
    match = re.search(pattern, input_string)
    if match:
        return match.group(1)
    raise ValueError(
        f"Could not extract the test category from the input string: {input_string}"
    )


def json_stats_to_dataframe(folder_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load per-category JSON summaries and return a MultiIndex DataFrame:
      index: ['High-Level Category', 'Subcategory']
      columns: ['Correct Count', 'Total Count', 'Accuracy, %', ...]
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder_path}")

    subcategory_to_highlevel = {
        sub: high_level
        for high_level, subcategories in CATEGORY_MAPPING.items()
        for sub in subcategories
    }

    records = []

    for json_file in sorted(folder_path.glob("*.json")):
        try:
            subcategory = extract_test_category(json_file)
            high_level = subcategory_to_highlevel.get(subcategory)
            if high_level is None:
                print(
                    f"Warning: No mapping for subcategory '{subcategory}' in file: {json_file.name}"
                )
                continue

            df_line = pd.read_json(json_file, lines=True, nrows=1)
            correct = int(df_line["correct_count"].iloc[0])
            total = int(df_line["total_count"].iloc[0])
            accuracy_percent = round(float(df_line["accuracy"].iloc[0]) * 100, 2)

            records.append(
                {
                    "High-Level Category": high_level,
                    "Subcategory": subcategory,
                    "Correct Count": correct,
                    "Total Count": total,
                    "Accuracy, %": accuracy_percent,
                }
            )
        except Exception as exc:
            print(f"Error processing {json_file.name}: {exc}")
            continue

    if not records:
        columns = ["Correct Count", "Total Count", "Accuracy, %", AVG_COL_NAME]
        return (
            pd.DataFrame(columns=["High-Level Category", "Subcategory", *columns])
            .astype(
                {
                    "Correct Count": "int64",
                    "Total Count": "int64",
                    "Accuracy, %": "float64",
                    AVG_COL_NAME: "float64",
                }
            )
            .set_index(["High-Level Category", "Subcategory"])
        )

    df = pd.DataFrame(records).set_index(["High-Level Category", "Subcategory"])

    df.index = df.index.set_levels(
        pd.CategoricalIndex(df.index.levels[0], categories=DESIRED_ORDER, ordered=True),
        level=0,
    )
    df = df.sort_index()

    df[AVG_COL_NAME] = (
        df.groupby(level="High-Level Category")["Accuracy, %"].transform("mean").round(2)
    )
    first_in_group = df.groupby(level="High-Level Category").cumcount() == 0
    df[AVG_COL_NAME] = df[AVG_COL_NAME].where(first_in_group)

    return df


def _escape_markdown_cell(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """
    Render a readable markdown table without requiring optional dependencies.
    """
    table_df = df.reset_index().copy()

    for col in ["Correct Count", "Total Count"]:
        if col in table_df.columns:
            table_df[col] = table_df[col].map(lambda x: f"{int(x)}")

    if "Accuracy, %" in table_df.columns:
        table_df["Accuracy, %"] = table_df["Accuracy, %"].map(
            lambda x: f"{float(x):.2f}"
        )

    if AVG_COL_NAME in table_df.columns:
        table_df[AVG_COL_NAME] = table_df[AVG_COL_NAME].map(
            lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
        )

    if "High-Level Category" in table_df.columns:
        previous = None
        rendered_category: List[str] = []
        for value in table_df["High-Level Category"].tolist():
            if value == previous:
                rendered_category.append("")
            else:
                rendered_category.append(str(value))
                previous = value
        table_df["High-Level Category"] = rendered_category

    headers = table_df.columns.tolist()
    numeric_cols = {"Correct Count", "Total Count", "Accuracy, %", AVG_COL_NAME}

    escaped_headers = [_escape_markdown_cell(h) for h in headers]
    escaped_rows: List[List[str]] = []
    for _, row in table_df.iterrows():
        escaped_rows.append([_escape_markdown_cell(row[h]) for h in headers])

    widths: List[int] = []
    for idx, header in enumerate(escaped_headers):
        cell_width = max((len(r[idx]) for r in escaped_rows), default=0)
        widths.append(max(len(header), cell_width))

    def _format_cell(col_name: str, value: str, width: int) -> str:
        if col_name in numeric_cols:
            return value.rjust(width)
        return value.ljust(width)

    header_line = "| " + " | ".join(
        _format_cell(headers[i], escaped_headers[i], widths[i])
        for i in range(len(headers))
    ) + " |"

    separator_cells: List[str] = []
    for i, col_name in enumerate(headers):
        width = max(3, widths[i])
        if col_name in numeric_cols:
            separator_cells.append("-" * (width - 1) + ":")
        else:
            separator_cells.append("-" * width)
    separator_line = "| " + " | ".join(separator_cells) + " |"

    lines = [header_line, separator_line]

    for row in escaped_rows:
        line = "| " + " | ".join(
            _format_cell(headers[i], row[i], widths[i]) for i in range(len(headers))
        ) + " |"
        lines.append(line)

    return "\n".join(lines)


def write_markdown_report(df: pd.DataFrame, source_folder: Path, output_path: Path) -> None:
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    markdown = dataframe_to_markdown(df)
    content = "\n".join(
        [
            "# BFCL Aggregated Metrics",
            "",
            f"- Source folder: `{source_folder}`",
            f"- Generated at: `{timestamp_utc}`",
            "",
            markdown,
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate BFCL JSON score files into a metrics table."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing BFCL_v3_*_score.json or BFCL_v3_*_result.json files.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help=(
            "Markdown output path. If omitted, writes to "
            "<folder>/aggregate_metrics.md."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = json_stats_to_dataframe(args.folder)

    md_output_path = args.output_md or (args.folder / "aggregate_metrics.md")
    write_markdown_report(df, args.folder, md_output_path)
    print(f"✅ Markdown report written to: {md_output_path.resolve()}")


if __name__ == "__main__":
    main()
