import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = (SCRIPT_DIR / "../../output/validation/optimizer_comparison").resolve()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot optimizer comparison from optimization_history.csv files."
    )
    parser.add_argument(
        "--series",
        action="append",
        default=[],
        help="Series definition in the form Label=/path/to/glob/**/optimization_history.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--main-labels",
        type=str,
        default="CEM+ZDD,GA-balanced,GA-explore,GA-exploit",
        help="Comma-separated labels to include in the main figure.",
    )
    parser.add_argument(
        "--appendix-labels",
        type=str,
        default="CEM+ZDD,GA+ZDD-balanced",
        help="Comma-separated labels to include in the appendix figure.",
    )
    parser.add_argument(
        "--objective",
        choices=["evac", "normal", "multi"],
        default="normal",
        help="Used to convert stored objective values back to the original metric scale.",
    )
    parser.add_argument(
        "--show-band",
        action="store_true",
        help="Show mean +/- std shaded bands.",
    )
    parser.add_argument(
        "--band-alpha",
        type=float,
        default=0.1,
        help="Alpha value for shaded uncertainty bands.",
    )
    parser.add_argument(
        "--show-runs",
        action="store_true",
        help="Overlay each run as a faint line behind the mean curve.",
    )
    parser.add_argument(
        "--run-alpha",
        type=float,
        default=0.18,
        help="Alpha value for per-run faint lines.",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=2.4,
        help="Line width for mean curves.",
    )
    return parser.parse_args()


def parse_series_specs(series_specs: list[str]) -> dict[str, list[Path]]:
    series_map: dict[str, list[Path]] = {}
    for spec in series_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --series specification: {spec}")
        label, pattern = spec.split("=", 1)
        paths = sorted(Path("/").glob(pattern[1:])) if pattern.startswith("/") else sorted(Path().glob(pattern))
        csv_paths = [path.resolve() for path in paths if path.is_file()]
        if not csv_paths:
            print(f"Warning: no files matched for {label}: {pattern}")
            continue
        series_map[label] = csv_paths
    if not series_map:
        raise FileNotFoundError("No optimization history files were found.")
    return series_map


def load_runs(paths: list[Path], label: str) -> list[pd.DataFrame]:
    runs = []
    for run_idx, path in enumerate(paths):
        df = pd.read_csv(path).sort_values("generation")
        df["label"] = label
        df["run_id"] = run_idx
        df["source_path"] = str(path)
        runs.append(df)
    return runs


def aggregate_runs(runs: list[pd.DataFrame], x_col: str, y_col: str) -> pd.DataFrame:
    common_x = sorted(set(np.concatenate([run[x_col].to_numpy(dtype=float) for run in runs])))
    common_index = pd.Index(common_x, name=x_col)

    aligned = []
    for run in runs:
        run_series = (
            run[[x_col, y_col]]
            .drop_duplicates(subset=x_col, keep="last")
            .set_index(x_col)[y_col]
            .sort_index()
            .reindex(common_index, method="ffill")
            .bfill()
        )
        aligned.append(run_series.to_numpy(dtype=float))

    aligned_arr = np.vstack(aligned)
    return pd.DataFrame(
        {
            x_col: common_x,
            "mean": aligned_arr.mean(axis=0),
            "std": aligned_arr.std(axis=0),
            "min": aligned_arr.min(axis=0),
            "max": aligned_arr.max(axis=0),
        }
    )


def metric_label(objective: str) -> str:
    if objective == "evac":
        return "TET"
    if objective == "normal":
        return "TTT"
    return "Weighted TTT/TET"


def objective_to_metric(values: pd.Series | np.ndarray) -> np.ndarray:
    return -np.asarray(values, dtype=float) * 1e5


def plot_two_panel(
    label_to_runs: dict[str, list[pd.DataFrame]],
    labels: list[str],
    output_path: Path,
    title: str,
    objective: str,
    show_band: bool,
    band_alpha: float,
    show_runs: bool,
    run_alpha: float,
    line_width: float,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:brown"]
    color_map = {label: colors[idx % len(colors)] for idx, label in enumerate(labels)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    panel_specs = [
        ("cumulative_simulation_evaluations", "best_so_far_cost", "Cumulative simulation evaluations"),
        ("elapsed_time_sec", "best_so_far_cost", "Wall-clock time (s)"),
    ]

    for ax, (x_col, y_col, xlabel) in zip(axes, panel_specs):
        for label in labels:
            runs = label_to_runs.get(label, [])
            if not runs:
                continue
            if show_runs:
                for run in runs:
                    run_df = (
                        run[[x_col, y_col]]
                        .drop_duplicates(subset=x_col, keep="last")
                        .sort_values(x_col)
                    )
                    ax.plot(
                        run_df[x_col],
                        objective_to_metric(run_df[y_col]),
                        color=color_map[label],
                        linewidth=1.0,
                        alpha=run_alpha,
                    )
            agg_df = aggregate_runs(runs, x_col=x_col, y_col=y_col)
            mean_metric = objective_to_metric(agg_df["mean"])
            std_metric = agg_df["std"].to_numpy(dtype=float) * 1e5
            ax.plot(agg_df[x_col], mean_metric, color=color_map[label], linewidth=line_width, label=label)
            if show_band:
                ax.fill_between(
                    agg_df[x_col],
                    mean_metric - std_metric,
                    mean_metric + std_metric,
                    color=color_map[label],
                    alpha=band_alpha,
                )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"Best-so-far {metric_label(objective)} (veh min)")
        ax.grid(True, alpha=0.25)

    axes[0].legend()
    fig.suptitle(title, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def write_readme(output_dir: Path, series_map: dict[str, list[Path]], main_labels: list[str], appendix_labels: list[str]):
    lines = [
        "Optimizer comparison plots",
        "- Each curve shows the mean best-so-far objective across runs.",
        "- Left panel uses cumulative simulation evaluations; right panel uses wall-clock time.",
        f"- Main figure labels: {', '.join(main_labels)}",
        f"- Appendix labels: {', '.join(appendix_labels)}",
    ]
    for label, paths in series_map.items():
        lines.append(f"- {label}: {len(paths)} run(s)")
    (output_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    series_map = parse_series_specs(args.series)
    label_to_runs = {
        label: load_runs(paths, label)
        for label, paths in series_map.items()
    }

    main_labels = [label.strip() for label in args.main_labels.split(",") if label.strip()]
    appendix_labels = [label.strip() for label in args.appendix_labels.split(",") if label.strip()]

    if main_labels:
        plot_two_panel(
            label_to_runs=label_to_runs,
            labels=main_labels,
            output_path=output_dir / "main_comparison.png",
            title="CEM+ZDD vs GA on transition-sequence optimization",
            objective=args.objective,
            show_band=args.show_band,
            band_alpha=args.band_alpha,
            show_runs=args.show_runs,
            run_alpha=args.run_alpha,
            line_width=args.line_width,
        )
    if appendix_labels:
        plot_two_panel(
            label_to_runs=label_to_runs,
            labels=appendix_labels,
            output_path=output_dir / "appendix_comparison.png",
            title="CEM+ZDD vs GA+ZDD on transition-sequence optimization",
            objective=args.objective,
            show_band=args.show_band,
            band_alpha=args.band_alpha,
            show_runs=args.show_runs,
            run_alpha=args.run_alpha,
            line_width=args.line_width,
        )
    write_readme(output_dir, series_map, main_labels, appendix_labels)
    print(f"Saved optimizer comparison plots to {output_dir}")


if __name__ == "__main__":
    main()
