import argparse
import math
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parents[3]
RECONFIG_DIR = WORKSPACE_DIR / "code" / "reconfigration"
SUMO_OUTPUT_DIR = WORKSPACE_DIR / "code" / "sumo" / "output"
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.append(str(SCRIPT_DIR.parent))
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))


DEFAULT_RANDOM_PATTERN = "zone*"
DEFAULT_OVERLAY_DIRS = [
    SUMO_OUTPUT_DIR / "0.8normal_1.0evac_baseline_93_ver3",
    SUMO_OUTPUT_DIR / "1.0normal_1.0evac_baseline_93_ver3",
    #SUMO_OUTPUT_DIR / "0.8normal_1.0evac_control_normal_93_tele1800",
    #SUMO_OUTPUT_DIR / "1.0normal_1.0evac_control_normal_93",
]
EDGE_FILE_RE = re.compile(r"edge_data_(\d+)\.csv$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate zoning homogeneity using zone-level MFD scaling fits."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RECONFIG_DIR / "output" / "validation" / "zoning_homogeneity" / "93_max",
        help="Directory for plots, fit tables, and raw aggregated data.",
    )
    parser.add_argument(
        "--network-path",
        type=Path,
        default=RECONFIG_DIR / "data" / "raw" / "network.geojson",
        help="Path to the network GeoJSON used by the zoning definition.",
    )
    parser.add_argument(
        "--cluster-path",
        type=Path,
        default=RECONFIG_DIR / "data" / "raw" / "zoning.csv",
        help="Path to the zoning CSV file.",
    )
    parser.add_argument(
        "--random-dir",
        dest="random_dirs",
        action="append",
        default=None,
        help="Optional explicit SUMO output directory to use for fitting. Can be specified multiple times.",
    )
    parser.add_argument(
        "--overlay-dir",
        dest="overlay_dirs",
        action="append",
        default=None,
        help="Optional explicit SUMO output directory to overlay as actual-evacuation points. Can be specified multiple times.",
    )
    parser.add_argument(
        "--max-points-per-zone",
        type=int,
        default=0,
        help="Optional cap on plotted points per zone for readability. Use 0 to keep all points.",
    )
    return parser.parse_args()


def resolve_random_fit_dirs(args) -> list[Path]:
    if args.random_dirs:
        requested = [Path(path).resolve() for path in args.random_dirs]
    else:
        requested = sorted(path.resolve() for path in SUMO_OUTPUT_DIR.glob(DEFAULT_RANDOM_PATTERN) if path.is_dir())

    fit_dirs = [path for path in requested if path.name.startswith("zone") and path.exists()]
    if not fit_dirs:
        raise FileNotFoundError("No zone* directories were found for scaled MFD fitting.")
    return fit_dirs


def resolve_overlay_dirs(args) -> list[Path]:
    requested = [Path(path).resolve() for path in args.overlay_dirs] if args.overlay_dirs else DEFAULT_OVERLAY_DIRS
    overlay_dirs = []
    for path in requested:
        if path.exists():
            overlay_dirs.append(path)
        else:
            print(f"Skipping missing overlay directory: {path}")
    return overlay_dirs


def load_zoning(network_path: Path, cluster_path: Path):
    from data.mfd_zoning import MFD_Zoning

    return MFD_Zoning(network_path=str(network_path), cluster_path=str(cluster_path))


def extract_edge_time(edge_path: Path) -> int:
    match = EDGE_FILE_RE.match(edge_path.name)
    if not match:
        raise ValueError(f"Unexpected edge file name: {edge_path.name}")
    return int(match.group(1))


def list_edge_files(edge_dir: Path) -> list[Path]:
    return sorted(edge_dir.glob("edge_data_*.csv"), key=extract_edge_time)


def aggregate_zone_points(scenario_dir: Path, edge_to_zone: dict[str, int], point_kind: str) -> pd.DataFrame:
    edge_dir = scenario_dir / "edge"
    edge_files = list_edge_files(edge_dir)
    if not edge_files:
        raise FileNotFoundError(f"No edge_data_*.csv files found in {edge_dir}")

    records = []
    for edge_path in edge_files:
        edge_time_sec = extract_edge_time(edge_path)
        df = pd.read_csv(edge_path, usecols=["edge_id", "accumulation", "link_weighted_flow"])
        df["zone_id"] = df["edge_id"].map(edge_to_zone)
        df = df.dropna(subset=["zone_id"]).copy()
        df["zone_id"] = df["zone_id"].astype(int)

        grouped = (
            df.groupby("zone_id", as_index=False)[["accumulation", "link_weighted_flow"]]
            .sum()
            .rename(columns={"link_weighted_flow": "production"})
        )
        grouped["scenario"] = scenario_dir.name
        grouped["time_sec"] = edge_time_sec
        grouped["point_kind"] = point_kind
        records.append(grouped)

    return pd.concat(records, ignore_index=True)


def maybe_downsample(zone_df: pd.DataFrame, max_points_per_zone: int) -> pd.DataFrame:
    if max_points_per_zone <= 0 or len(zone_df) <= max_points_per_zone:
        return zone_df
    return (
        zone_df.sort_values(["scenario", "time_sec"])
        .iloc[np.linspace(0, len(zone_df) - 1, max_points_per_zone, dtype=int)]
        .reset_index(drop=True)
    )


def filter_fit_points_for_zone(fit_points: pd.DataFrame, zone_id: int) -> pd.DataFrame:
    zone_token = f"zone{zone_id}"
    return fit_points[
        (fit_points["zone_id"] == zone_id)
        & (fit_points["scenario"].str.contains(zone_token, regex=False))
    ].copy()


def truncate_fit_curve_for_plot(fit_df: pd.DataFrame, fit_summary_row: pd.Series) -> pd.DataFrame:
    coef1 = float(fit_summary_row["coef1"])
    coef2 = float(fit_summary_row["coef2"])
    coef3 = float(fit_summary_row["coef3"])

    roots = np.roots([3.0 * coef1, 2.0 * coef2, coef3])
    real_roots = sorted(
        root.real
        for root in roots
        if abs(root.imag) < 1e-8 and root.real > 0
    )
    if not real_roots:
        return fit_df

    cutoff = real_roots[-1]
    truncated = fit_df[fit_df["accumulation"] <= cutoff].copy()
    if truncated.empty:
        return fit_df
    return truncated


def scaled_mfd_model(N, alpha, beta, A_unit, B_unit, C_unit):
    n_scaled = N / beta
    return alpha * (A_unit * n_scaled**3 + B_unit * n_scaled**2 + C_unit * n_scaled)


def fit_zone_scaled_mfd(zone_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    from scipy.optimize import curve_fit
    from sklearn.metrics import mean_squared_error, r2_score

    A_unit = 4.133e-11 * 3600
    B_unit = -8.282e-7 * 3600
    C_unit = 4.2e-3 * 3600
    N_jam_unit = 1.0e4

    X = zone_df["accumulation"].to_numpy(dtype=float)
    y = zone_df["production"].to_numpy(dtype=float)

    initial_guess = [0.5, 0.5]
    popt, _ = curve_fit(
        lambda N, alpha, beta: scaled_mfd_model(N, alpha, beta, A_unit, B_unit, C_unit),
        X,
        y,
        p0=initial_guess,
        bounds=(0, [1.0, 1.0]),
        maxfev=20000,
    )

    alpha_opt, beta_opt = popt
    y_pred = scaled_mfd_model(X, alpha_opt, beta_opt, A_unit, B_unit, C_unit)
    rmse = math.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    coef1 = alpha_opt * A_unit * (1.0 / beta_opt**3)
    coef2 = alpha_opt * B_unit * (1.0 / beta_opt**2)
    coef3 = alpha_opt * C_unit * (1.0 / beta_opt)
    n_jam_opt = N_jam_unit * beta_opt

    x_upper = max(float(X.max()), float(n_jam_opt))
    x_fit = np.linspace(0.0, x_upper, 200)
    y_fit = scaled_mfd_model(x_fit, alpha_opt, beta_opt, A_unit, B_unit, C_unit)
    fit_df = pd.DataFrame(
        {
            "accumulation": x_fit,
            "fitted_production": y_fit,
        }
    )

    summary = {
        "zone_id": int(zone_df["zone_id"].iloc[0]),
        "n_fit_points": int(len(zone_df)),
        "alpha": float(alpha_opt),
        "beta": float(beta_opt),
        "n_jam_scaled": float(n_jam_opt),
        "coef1": float(coef1),
        "coef2": float(coef2),
        "coef3": float(coef3),
        "r2_fit": float(r2),
        "rmse_fit": float(rmse),
        "accumulation_min": float(zone_df["accumulation"].min()),
        "accumulation_max": float(zone_df["accumulation"].max()),
        "production_min": float(zone_df["production"].min()),
        "production_max": float(zone_df["production"].max()),
    }
    return fit_df, summary


def evaluate_overlay(zone_overlay_df: pd.DataFrame, fit_summary: dict) -> dict:
    if zone_overlay_df.empty:
        return {
            "n_overlay_points": 0,
            "r2_overlay": np.nan,
            "rmse_overlay": np.nan,
        }

    from sklearn.metrics import mean_squared_error, r2_score

    A_unit = 4.133e-11 * 3600
    B_unit = -8.282e-7 * 3600
    C_unit = 4.2e-3 * 3600

    x = zone_overlay_df["accumulation"].to_numpy(dtype=float)
    y = zone_overlay_df["production"].to_numpy(dtype=float)
    y_pred = scaled_mfd_model(x, fit_summary["alpha"], fit_summary["beta"], A_unit, B_unit, C_unit)

    return {
        "n_overlay_points": int(len(zone_overlay_df)),
        "r2_overlay": float(r2_score(y, y_pred)) if len(zone_overlay_df) >= 2 else np.nan,
        "rmse_overlay": float(math.sqrt(mean_squared_error(y, y_pred))),
    }


def save_zone_plots(fit_points: pd.DataFrame, overlay_points: pd.DataFrame, fit_results: dict[int, pd.DataFrame], fit_summary_df: pd.DataFrame, output_dir: Path, max_points_per_zone: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    axes = axes.flatten()

    for zone_id in sorted(fit_points["zone_id"].unique()):
        ax = axes[zone_id]
        zone_fit_points = maybe_downsample(
            filter_fit_points_for_zone(fit_points, zone_id).sort_values(["scenario", "time_sec"]),
            max_points_per_zone,
        )
        zone_overlay_points = maybe_downsample(
            overlay_points[overlay_points["zone_id"] == zone_id].sort_values(["scenario", "time_sec"]),
            max_points_per_zone,
        )
        summary_row = fit_summary_df.loc[fit_summary_df["zone_id"] == zone_id].iloc[0]
        fit_df = truncate_fit_curve_for_plot(fit_results[zone_id], summary_row)

        ax.scatter(
            zone_fit_points["accumulation"],
            zone_fit_points["production"],
            s=12,
            alpha=0.35,
            color="tab:blue",
            label="Random demand samples",
        )
        if not zone_overlay_points.empty:
            ax.scatter(
                zone_overlay_points["accumulation"],
                zone_overlay_points["production"],
                s=18,
                alpha=0.75,
                color="tab:orange",
                label="Actual evacuation cases",
            )
        ax.plot(
            fit_df["accumulation"],
            fit_df["fitted_production"],
            color="crimson",
            linewidth=2.0,
            label="Scaled MFD fit",
        )
        ax.set_title(f"Zone {zone_id}")
        #ax.grid(True, alpha=0.25)
        ax.grid(False)
        overlay_r2_text = f"{summary_row['r2_overlay']:.3f}" if not pd.isna(summary_row["r2_overlay"]) else "NA"
        overlay_rmse_text = f"{summary_row['rmse_overlay']:.1f}" if not pd.isna(summary_row["rmse_overlay"]) else "NA"
        ax.text(
            0.03,
            0.97,
            (
                f"R² fit={summary_row['r2_fit']:.3f}\n"
                f"RMSE fit={summary_row['rmse_fit']:.1f}\n"
                #f"R² overlay={overlay_r2_text}\n"
                #f"RMSE overlay={overlay_rmse_text}"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
        if zone_id % 3 == 0:
            ax.set_ylabel(r"Production (veh $\cdot$ km/h)")
        if zone_id >= 6:
            ax.set_xlabel("Accumulation (veh)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Zone-level scaled MFD fits with evacuation overlays", fontsize=18, y=0.985)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / "zone_scaled_mfd.png", dpi=300)
    plt.close(fig)

    individual_dir = output_dir / "per_zone_plots"
    individual_dir.mkdir(parents=True, exist_ok=True)
    for zone_id in sorted(fit_points["zone_id"].unique()):
        zone_fit_points = maybe_downsample(
            filter_fit_points_for_zone(fit_points, zone_id).sort_values(["scenario", "time_sec"]),
            max_points_per_zone,
        )
        zone_overlay_points = maybe_downsample(
            overlay_points[overlay_points["zone_id"] == zone_id].sort_values(["scenario", "time_sec"]),
            max_points_per_zone,
        )
        summary_row = fit_summary_df.loc[fit_summary_df["zone_id"] == zone_id].iloc[0]
        fit_df = truncate_fit_curve_for_plot(fit_results[zone_id], summary_row)

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.scatter(
            zone_fit_points["accumulation"],
            zone_fit_points["production"],
            s=14,
            alpha=0.35,
            color="tab:blue",
            label="Random demand samples",
        )
        if not zone_overlay_points.empty:
            ax.scatter(
                zone_overlay_points["accumulation"],
                zone_overlay_points["production"],
                s=22,
                alpha=0.8,
                color="tab:orange",
                label="Actual evacuation cases",
            )
        ax.plot(
            fit_df["accumulation"],
            fit_df["fitted_production"],
            color="crimson",
            linewidth=2.0,
            label="Scaled MFD fit",
        )
        ax.set_title(f"Zone {zone_id}")
        ax.set_xlabel("Accumulation (veh)")
        ax.set_ylabel(r"Production (veh $\cdot$ km/h)")
        #ax.grid(True, alpha=0.25)
        ax.grid(False)
        ax.legend()
        overlay_r2_text = f"{summary_row['r2_overlay']:.3f}" if not pd.isna(summary_row["r2_overlay"]) else "NA"
        overlay_rmse_text = f"{summary_row['rmse_overlay']:.1f}" if not pd.isna(summary_row["rmse_overlay"]) else "NA"
        ax.text(
            0.03,
            0.97,
            (
                f"alpha={summary_row['alpha']:.3f}\n"
                f"beta={summary_row['beta']:.3f}\n"
                f"R² fit={summary_row['r2_fit']:.3f}\n"
                f"RMSE fit={summary_row['rmse_fit']:.1f}\n"
                f"R² overlay={overlay_r2_text}\n"
                f"RMSE overlay={overlay_rmse_text}"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
        plt.tight_layout()
        plt.savefig(individual_dir / f"zone_{zone_id}_scaled_mfd.png", dpi=300)
        plt.close(fig)


def write_readme(output_dir: Path, fit_dirs: list[Path], overlay_dirs: list[Path], fit_summary_df: pd.DataFrame):
    lines = [
        "Zoning homogeneity validation",
        "- Objective: assess whether the fixed 9-zone partition yields consistent zone-level MFDs.",
        "- Fit data: SUMO output directories whose names start with zone, matching the execution block in mfd_zoning.py.",
        "- Overlay data: actual evacuation scenarios are plotted in a different color to check consistency with the fitted zone MFD.",
        "- Variable definitions: accumulation = sum of link accumulations in the zone; production = sum of link_weighted_flow in the zone.",
        "- Fit model: scaled version of the given unit MFD in fit_scaled_mfd_per_cluster, using alpha and beta as scaling factors.",
        f"- Number of fit directories: {len(fit_dirs)}",
        f"- Number of overlay directories used: {len(overlay_dirs)}",
        f"- Mean fit R² across zones: {fit_summary_df['r2_fit'].mean():.4f}",
        f"- Mean fit RMSE across zones: {fit_summary_df['rmse_fit'].mean():.2f}",
    ]
    if fit_summary_df["n_overlay_points"].sum() > 0:
        lines.append(f"- Mean overlay RMSE across zones: {fit_summary_df['rmse_overlay'].mean():.2f}")
    (output_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    fit_dirs = resolve_random_fit_dirs(args)
    overlay_dirs = resolve_overlay_dirs(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    zoning = load_zoning(args.network_path.resolve(), args.cluster_path.resolve())
    edge_to_zone = {str(edge_id): zone_id for edge_id, zone_id in zoning.edge2cluster.items()}

    fit_frames = []
    for scenario_dir in fit_dirs:
        print(f"Aggregating fit points from {scenario_dir.name}...")
        fit_frames.append(aggregate_zone_points(scenario_dir, edge_to_zone, point_kind="fit"))
    fit_points = pd.concat(fit_frames, ignore_index=True)
    fit_points.to_csv(output_dir / "zone_fit_points_all.csv", index=False)

    overlay_points = pd.DataFrame(columns=["zone_id", "accumulation", "production", "scenario", "time_sec", "point_kind"])
    overlay_frames = []
    for scenario_dir in overlay_dirs:
        print(f"Aggregating overlay points from {scenario_dir.name}...")
        overlay_frames.append(aggregate_zone_points(scenario_dir, edge_to_zone, point_kind="overlay"))
    if overlay_frames:
        overlay_points = pd.concat(overlay_frames, ignore_index=True)
        overlay_points.to_csv(output_dir / "zone_overlay_points_all.csv", index=False)

    raw_data_dir = output_dir / "per_zone_data"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    fit_results: dict[int, pd.DataFrame] = {}
    summary_rows = []
    for zone_id in sorted(fit_points["zone_id"].unique()):
        zone_fit_df = filter_fit_points_for_zone(fit_points, zone_id)
        if zone_fit_df.empty:
            raise ValueError(f"No fitting points found for zone {zone_id}.")
        zone_fit_df.to_csv(raw_data_dir / f"zone_{zone_id}_fit_points.csv", index=False)

        fit_df, summary = fit_zone_scaled_mfd(zone_fit_df)
        fit_df.to_csv(raw_data_dir / f"zone_{zone_id}_scaled_fit.csv", index=False)
        fit_results[zone_id] = fit_df

        zone_overlay_df = overlay_points[overlay_points["zone_id"] == zone_id].copy() if not overlay_points.empty else pd.DataFrame()
        if not zone_overlay_df.empty:
            zone_overlay_df.to_csv(raw_data_dir / f"zone_{zone_id}_overlay_points.csv", index=False)
        overlay_eval = evaluate_overlay(zone_overlay_df, summary)
        summary.update(overlay_eval)
        summary_rows.append(summary)

    fit_summary_df = pd.DataFrame(summary_rows).sort_values("zone_id")
    fit_summary_df.to_csv(output_dir / "zone_scaled_mfd_summary.csv", index=False)

    save_zone_plots(
        fit_points=fit_points,
        overlay_points=overlay_points,
        fit_results=fit_results,
        fit_summary_df=fit_summary_df,
        output_dir=output_dir,
        max_points_per_zone=args.max_points_per_zone,
    )
    write_readme(output_dir, fit_dirs, overlay_dirs, fit_summary_df)

    print(f"Saved zoning validation outputs to {output_dir}")


if __name__ == "__main__":
    main()
