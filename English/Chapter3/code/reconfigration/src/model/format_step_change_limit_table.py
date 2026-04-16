import argparse
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from analyze_step_change_limit_sensitivity import build_table  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Format the step-change sensitivity summary into publication-ready CSV and LaTeX tables."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=(SCRIPT_DIR / "../../output/validation/step_change_limit_sensitivity/step_change_limit_summary.csv").resolve(),
        help="Input summary CSV produced by analyze_step_change_limit_sensitivity.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(SCRIPT_DIR / "../../output/validation/step_change_limit_sensitivity").resolve(),
        help="Directory where the formatted table files will be written.",
    )
    parser.add_argument(
        "--objective",
        choices=["evac", "normal", "multi"],
        default="normal",
        help="Objective used in the summary analysis.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(args.summary_csv).sort_values("step_change_limit")
    table_df = build_table(summary_df, args.objective)

    csv_path = output_dir / "step_change_limit_table.csv"
    tex_path = output_dir / "step_change_limit_table.tex"

    table_df.to_csv(csv_path, index=False)
    latex = table_df.to_latex(
        index=False,
        escape=False,
        float_format=lambda x: f"{x:.3f}",
        caption="Sensitivity of reconfiguration performance to the maximum number of link changes per step.",
        label="tab:step_change_limit_sensitivity",
        column_format="lrrrrrrrrr",
    )
    tex_path.write_text(latex, encoding="utf-8")

    print(f"Saved CSV table to {csv_path}")
    print(f"Saved LaTeX table to {tex_path}")


if __name__ == "__main__":
    main()
