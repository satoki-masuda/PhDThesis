import argparse
import copy
import math
import os
import sys
import tempfile
from pathlib import Path
from parameters_ndp import Parameters

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))


SCENARIOS = [
    {
        "name": "baseline_back1.0",
        "background_ratio": 1.0,
        "micro_dir": Path(
            "/Users/masudasatoki/Desktop/MFD_evac/code/sumo/output/1.0normal_1.0evac_baseline_93"
        ),
        "macro_reference_dir": Path(
            "/Users/masudasatoki/Desktop/MFD_evac/code/reconfigration/output/mfd_dynamics/route_update_60min/demand_36h/93_max/back1.0/0_24"
        ),
        "best_sequence_csv": None,
    },
    {
        "name": "control_normal_back1.0",
        "background_ratio": 1.0,
        "micro_dir": Path(
            "/Users/masudasatoki/Desktop/MFD_evac/code/sumo/output/1.0normal_1.0evac_control_normal_93"
        ),
        "macro_reference_dir": Path(
            "/Users/masudasatoki/Desktop/MFD_evac/code/reconfigration/output/mfd_dynamics/route_update_60min/demand_36h/93_max/back1.0/0_24_control6_normal"
        ),
        "best_sequence_csv": Path(
            "/Users/masudasatoki/Desktop/MFD_evac/code/reconfigration/output/mfd_dynamics/route_update_60min/demand_36h/93_max/back1.0/0_24_control6_normal/best_sequence.csv"
        ),
    },
]

MICRO_START_SEC = 22200
MICRO_END_SEC = 54000
MICRO_INTERVAL_SEC = 600
EDGE_TIME_OFFSET_SEC = MICRO_START_SEC


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare existing microscopic outputs with recomputed macroscopic MFD outputs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/Users/masudasatoki/Desktop/MFD_evac/code/reconfigration/output/validation/macro_micro/93_max"
        ),
        help="Directory for figures and summary tables.",
    )
    parser.add_argument(
        "--reuse-macro-cache",
        action="store_true",
        help="Reuse cached macro time series if they already exist in the output directory.",
    )
    parser.add_argument(
        "--zones",
        type=str,
        default=None,
        help="Comma-separated zone IDs to display in the plots, e.g. '0,1,4,7'. If omitted, all zones are shown.",
    )
    return parser.parse_args()


def vector_to_matrix(vector: np.ndarray, num_zones: int):
    mat = np.zeros((num_zones, num_zones))
    mat[np.where(~np.eye(num_zones, dtype=bool))] = vector
    return mat


def parse_zone_list(raw: str | None, available_zones: int):
    if raw is None or raw.strip() == "":
        return list(range(available_zones))
    zones = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if any(zone < 0 or zone >= available_zones for zone in zones):
        raise ValueError(f"Zone IDs must be between 0 and {available_zones - 1}.")
    return zones


def build_sample_times():
    return list(range(MICRO_START_SEC, MICRO_END_SEC + MICRO_INTERVAL_SEC, MICRO_INTERVAL_SEC))


def discover_available_micro_times(scenario: dict):
    summary_dir = scenario["micro_dir"] / "simulation_summary"
    edge_dir = scenario["micro_dir"] / "edge"

    summary_times = {
        int(path.stem.split("_")[-1])
        for path in summary_dir.glob("zone_statistics_summary_*.csv")
    }
    edge_times = {
        int(path.stem.split("_")[-1])
        for path in edge_dir.glob("edge_data_*.csv")
    }

    requested = set(build_sample_times())
    available_summary = sorted(requested & summary_times)
    available_edge_summary_clock = sorted(
        t for t in requested
        if (t - EDGE_TIME_OFFSET_SEC) in edge_times
    )
    if not available_summary:
        raise ValueError(f"No summary timestamps found for {scenario['name']}.")
    return {
        "summary_times": available_summary,
        "edge_times_on_summary_clock": set(available_edge_summary_clock),
    }


def edge_time_from_summary_time(summary_time_sec: int):
    return summary_time_sec - EDGE_TIME_OFFSET_SEC


def load_zone_clusters():
    from data.mfd_zoning import MFD_Zoning

    zoning = MFD_Zoning(
        network_path="../../data/raw/network.geojson",
        cluster_path="../../data/raw/zoning.csv",
    )
    return {zone_id: set(cluster) for zone_id, cluster in enumerate(zoning.clusters)}


def compute_micro_edge_metrics(edge_csv: Path, zone_clusters: dict):
    df = pd.read_csv(edge_csv)
    metrics = {}
    for zone_id, edge_ids in zone_clusters.items():
        mask = df["edge_id"].isin(edge_ids)
        accumulation = float(df.loc[mask, "accumulation"].sum())
        production = float(df.loc[mask, "link_weighted_flow"].sum())
        speed = np.nan
        if accumulation > 0:
            speed = production / accumulation
        metrics[zone_id] = {
            "accumulation": accumulation,
            "production": production,
            "mean_speed": speed,
        }
    return metrics


def load_micro_timeseries(scenario: dict, params, zone_clusters: dict):
    zone_rows = []
    time_info = discover_available_micro_times(scenario)
    sample_times = time_info["summary_times"]
    edge_times_on_summary_clock = time_info["edge_times_on_summary_clock"]

    for sample_sec in sample_times:
        summary_path = scenario["micro_dir"] / "simulation_summary" / f"zone_statistics_summary_{sample_sec}.csv"
        summary_df = pd.read_csv(summary_path)
        edge_metrics = {
            zone_id: {"accumulation": np.nan, "production": np.nan, "mean_speed": np.nan}
            for zone_id in zone_clusters
        }
        if sample_sec in edge_times_on_summary_clock:
            edge_time_sec = edge_time_from_summary_time(sample_sec)
            edge_path = scenario["micro_dir"] / "edge" / f"edge_data_{edge_time_sec}.csv"
            edge_metrics = compute_micro_edge_metrics(edge_path, zone_clusters)

        for row in summary_df.itertuples(index=False):
            zone_id = int(row.zone_id)
            mean_speed = edge_metrics[zone_id]["mean_speed"]
            avg_tt = np.nan
            if pd.notna(mean_speed) and mean_speed > 0:
                avg_tt = float(60.0 * params.L_m[zone_id] / mean_speed)

            zone_rows.append(
                {
                    "scenario": scenario["name"],
                    "time_sec": sample_sec,
                    "time_hour": sample_sec / 3600.0,
                    "zone_id": zone_id,
                    "micro_mean_speed": mean_speed,
                    "micro_accumulation": edge_metrics[zone_id]["accumulation"],
                    "micro_production": edge_metrics[zone_id]["production"],
                    "micro_avg_tt": avg_tt,
                    "micro_summary_mean_speed": float(row.mean_speed),
                }
            )

    return pd.DataFrame(zone_rows)


def run_macro_baseline(params):
    from mfd_dynamics import MFD_Dynamics

    sim = MFD_Dynamics(params, output_path=None)
    sim.run_simulation()
    return sim


def load_best_sequence(sequence_csv: Path):
    return pd.read_csv(sequence_csv, header=None).values.astype(int)


def apply_control_sequence(params, sim, best_control_sequence, reconf_start_time=6 * 60, reconf_end_time=12 * 60, seq_interval=60):
    num_zones = params.num_zones
    link_indices = [
        (i, j)
        for i in range(num_zones)
        for j in range(num_zones)
        if params.adj_matrix[i, j] == 1
    ]
    original_capacity = copy.deepcopy(params.max_boundary_capacity)

    for _ in range(sim.sim_start_step, sim.sim_end_step):
        if ((sim.step % seq_interval) == 0) and (reconf_start_time < sim.step <= reconf_end_time):
            sim.params.max_boundary_capacity = copy.deepcopy(original_capacity)
            step_idx = (sim.step // seq_interval) - (reconf_start_time // seq_interval) - 1
            contraflow_idx = best_control_sequence[step_idx]
            for i, j in [link_indices[idx] for idx in np.where(contraflow_idx == 1)[0]]:
                sim.params.max_boundary_capacity[i, j] *= params.contra_ratio
                sim.params.max_boundary_capacity[j, i] *= (2 - params.contra_ratio)
        sim.step_simulation()

    return sim


def macro_zone_timeseries(sim, params):
    num_zones = params.num_zones
    records = []

    for step_idx, (x, x_background) in enumerate(zip(sim.xs, sim.xs_background), start=1):
        time_sec = step_idx * 60

        n_m = x[:num_zones]
        n_s = x[num_zones : 2 * num_zones]
        n_o = x[2 * num_zones : num_zones**2 + num_zones]
        n_d = x[num_zones**2 + 2 * num_zones :]
        n_m_back = x_background[:num_zones]
        n_o_back = x_background[num_zones : num_zones**2]

        n_o_mat = vector_to_matrix(n_o, num_zones)
        n_d_mat = vector_to_matrix(n_d, num_zones)
        n_o_back_mat = vector_to_matrix(n_o_back, num_zones)

        road_accumulation = n_m + n_s + n_o_mat.sum(axis=1) + n_d_mat.sum(axis=1) + n_m_back + n_o_back_mat.sum(axis=1)
        production = sim.mfd(road_accumulation) * 60.0
        mean_speed_km_per_min = np.divide(
            production / 60.0,
            road_accumulation,
            out=np.zeros_like(road_accumulation),
            where=road_accumulation > 0,
        )
        mean_speed = mean_speed_km_per_min * 60.0
        avg_tt = np.divide(
            60.0 * params.L_m,
            mean_speed,
            out=np.full_like(mean_speed, np.nan, dtype=float),
            where=mean_speed > 0,
        )

        for zone_id in range(num_zones):
            records.append(
                {
                    "time_sec": time_sec,
                    "time_hour": time_sec / 3600.0,
                    "zone_id": zone_id,
                    "macro_mean_speed": float(mean_speed[zone_id]),
                    "macro_accumulation": float(road_accumulation[zone_id]),
                    "macro_production": float(production[zone_id]),
                    "macro_avg_tt": float(avg_tt[zone_id]) if not math.isnan(avg_tt[zone_id]) else np.nan,
                }
            )

    return pd.DataFrame(records)


def get_macro_series(scenario: dict, output_dir: Path, reuse_cache: bool):
    from mfd_dynamics import MFD_Dynamics

    cache_path = output_dir / scenario["name"] / "macro_timeseries.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if reuse_cache and cache_path.exists():
        return pd.read_csv(cache_path)

    params = Parameters(background_ratio=scenario["background_ratio"])
    params.path_update_interval = 60
    sim = MFD_Dynamics(params, output_path=None)

    if scenario["best_sequence_csv"] is None:
        sim.run_simulation()
    else:
        best_sequence = load_best_sequence(scenario["best_sequence_csv"])
        sim = apply_control_sequence(params, sim, best_sequence)

    macro_df = macro_zone_timeseries(sim, params)
    macro_df.to_csv(cache_path, index=False)
    return macro_df


def merge_micro_macro(micro_df: pd.DataFrame, macro_df: pd.DataFrame):
    sample_times = sorted(micro_df["time_sec"].unique().tolist())
    macro_slice = macro_df[macro_df["time_sec"].isin(sample_times)].copy()
    merged = micro_df.merge(macro_slice, on=["zone_id", "time_sec", "time_hour"], how="inner")
    return merged


def safe_mape(actual: pd.Series, pred: pd.Series):
    denom = actual.abs()
    valid = denom > 1e-8
    if not valid.any():
        return np.nan
    return float((np.abs(actual[valid] - pred[valid]) / denom[valid]).mean() * 100.0)


def compute_error_tables(merged_df: pd.DataFrame):
    metric_pairs = [
        ("mean_speed", "micro_mean_speed", "macro_mean_speed"),
        ("accumulation", "micro_accumulation", "macro_accumulation"),
        ("production", "micro_production", "macro_production"),
        ("avg_tt", "micro_avg_tt", "macro_avg_tt"),
    ]

    rows = []
    for (scenario, zone_id), group in merged_df.groupby(["scenario", "zone_id"]):
        for metric_name, micro_col, macro_col in metric_pairs:
            valid = group[[micro_col, macro_col]].dropna()
            if valid.empty:
                continue
            diff = valid[macro_col] - valid[micro_col]
            mae = float(np.abs(diff).mean())
            rmse = float(np.sqrt(np.mean(np.square(diff))))
            bias = float(diff.mean())
            mape = safe_mape(valid[micro_col], valid[macro_col])
            rows.append(
                {
                    "scenario": scenario,
                    "zone_id": zone_id,
                    "metric": metric_name,
                    "mae": mae,
                    "rmse": rmse,
                    "bias": bias,
                    "mape_percent": mape,
                }
            )

    zone_error_df = pd.DataFrame(rows)
    summary_error_df = (
        zone_error_df.groupby(["scenario", "metric"], as_index=False)[["mae", "rmse", "bias", "mape_percent"]]
        .mean()
        .sort_values(["scenario", "metric"])
    )
    return zone_error_df, summary_error_df


def plot_metric_subplots(
    merged_df: pd.DataFrame,
    scenario_name: str,
    metric: str,
    micro_col: str,
    macro_col: str,
    ylabel: str,
    output_path: Path,
    selected_zones: list[int],
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenario_df = merged_df[merged_df["scenario"] == scenario_name]
    n_plots = len(selected_zones)
    ncols = 3 if n_plots > 1 else 1
    nrows = int(math.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.0 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for ax in axes[n_plots:]:
        ax.axis("off")

    for ax, zone_id in zip(axes[:n_plots], selected_zones):
        zone_df = scenario_df[scenario_df["zone_id"] == zone_id].sort_values("time_sec")
        ax.plot(zone_df["time_hour"], zone_df[micro_col],  linewidth=1.5, label="Micro")
        ax.plot(zone_df["time_hour"], zone_df[macro_col],  linewidth=1.5, label="Macro")
        ax.set_title(f"Zone {zone_id}")
        ax.grid(True, alpha=0.3)
        if zone_id % 3 == 0:
            ax.set_ylabel(ylabel)
        if zone_id >= 6:
            ax.set_xlabel("Time (hour)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(f"{scenario_name}: micro vs macro {metric}", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def write_readme(output_dir: Path):
    text = (
        "Macro-micro validation setup\n"
        "- Microscopic data: existing SUMO outputs only.\n"
        "- Macroscopic data: recomputed for each scenario.\n"
        "- Baseline scenarios use the MFD model without reconfiguration control.\n"
        "- Control scenarios use the matching reconfiguration control sequence from best_sequence.csv.\n"
        "- Comparison window: 22200s to 54000s at 600s intervals.\n"
        "- Microscopic edge_data_*.csv uses relative time with 22200s mapped to edge_data_0.csv.\n"
        "- Accumulation compares on-road vehicles only; parked vehicles are excluded on the macro side to match link-based micro accumulation.\n"
        "- Microscopic mean speed is recomputed as zone production / zone accumulation, not the simple edge-speed average in simulation_summary.\n"
        "- Production is compared in veh km/h.\n"
        "- Average travel time is approximated per zone as 60 * average_trip_length / mean_speed.\n"
        "- Scenarios whose microscopic input folder is unavailable are skipped automatically.\n"
    )
    (output_dir / "README.txt").write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_readme(output_dir)

    from parameters_ndp import Parameters

    zone_clusters = load_zone_clusters()
    selected_zones = parse_zone_list(args.zones, available_zones=len(zone_clusters))
    params_for_micro = Parameters()

    merged_frames = []
    for scenario in SCENARIOS:
        scenario_out = output_dir / scenario["name"]
        scenario_out.mkdir(parents=True, exist_ok=True)

        if not scenario["micro_dir"].exists():
            print(
                f"Skipping {scenario['name']}: microscopic directory not found: "
                f"{scenario['micro_dir']}"
            )
            continue

        print(f"Loading micro data for {scenario['name']}...")
        micro_df = load_micro_timeseries(scenario, params_for_micro, zone_clusters)
        micro_df.to_csv(scenario_out / "micro_timeseries.csv", index=False)

        print(f"Computing macro series for {scenario['name']}...")
        macro_df = get_macro_series(scenario, output_dir, args.reuse_macro_cache)

        merged_df = merge_micro_macro(micro_df, macro_df)
        merged_df.to_csv(scenario_out / "macro_micro_timeseries.csv", index=False)
        merged_frames.append(merged_df)

        plot_metric_subplots(
            merged_df,
            scenario["name"],
            "mean speed",
            "micro_mean_speed",
            "macro_mean_speed",
            "Speed (km/h)",
            scenario_out / "mean_speed_comparison.png",
            selected_zones,
        )
        plot_metric_subplots(
            merged_df,
            scenario["name"],
            "accumulation",
            "micro_accumulation",
            "macro_accumulation",
            "Accumulation (veh)",
            scenario_out / "accumulation_comparison.png",
            selected_zones,
        )
        plot_metric_subplots(
            merged_df,
            scenario["name"],
            "production",
            "micro_production",
            "macro_production",
            r"Production (veh km/h)",
            scenario_out / "production_comparison.png",
            selected_zones,
        )
        plot_metric_subplots(
            merged_df,
            scenario["name"],
            "average travel time",
            "micro_avg_tt",
            "macro_avg_tt",
            "Average travel time (min/veh)",
            scenario_out / "avg_tt_comparison.png",
            selected_zones,
        )

    if not merged_frames:
        raise FileNotFoundError("No scenarios were processed because no microscopic directories were found.")

    merged_all = pd.concat(merged_frames, ignore_index=True)
    zone_error_df, summary_error_df = compute_error_tables(merged_all)
    zone_error_df.to_csv(output_dir / "zone_error_table.csv", index=False)
    summary_error_df.to_csv(output_dir / "summary_error_table.csv", index=False)
    merged_all.to_csv(output_dir / "macro_micro_timeseries_all.csv", index=False)

    print(f"Saved comparison outputs to {output_dir}")


if __name__ == "__main__":
    main()
