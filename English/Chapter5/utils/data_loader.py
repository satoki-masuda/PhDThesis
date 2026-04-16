"""Shared data-loading entry point for estimation and simulation."""
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from model.transition_model import TransitionModel
from model.data_reading import (
    read_pt_data,
    read_pop_data,
    read_building_data,
    read_los_data,
)


@dataclass
class SimulationDataBundle:
    """Container for the datasets used in estimation and forward simulation."""

    transition_model: TransitionModel
    df_pt: pd.DataFrame
    data: pd.DataFrame
    Dzone_list: List[int]
    Rzone_list: List[int]
    Darea_list: List[float]
    columns_zone_fe: List[str]
    columns_year_fe: List[str]
    los_std: List[float]
    develop_std: List[float]
    features: List[np.ndarray]


def load_simulation_data(config) -> SimulationDataBundle:
    """Prepare the shared data bundle used by `main.py` and `counterfactual.py`."""
    transition_model = TransitionModel(
        start_year=config.start_year,
        end_year=config.end_year,
        ref_year=config.ref_year,
        dev_zone=config.dev_zone,
        res_zone=config.res_zone,
        config=config,
    )

    Dzone_list = list(map(int, transition_model.Dzone_list))
    Rzone_list = list(map(int, transition_model.zone_list))

    df_pt = read_pt_data(
        config.start_year,
        config.end_year,
        zoning_type=config.res_zone,
    )

    pop_dict_D = read_pop_data(config.start_year, config.end_year, config.dev_zone)
    develop_dict_D = read_building_data(
        config.start_year - config.ref_year - 1,
        config.end_year,
        config.dev_zone,
    )
    los_dict_D = read_los_data(config.start_year, config.end_year, config.dev_zone)

    Dzoning = transition_model.Dzoning
    Darea_list = [
        Dzoning.loc[Dzoning["選択ゾーン"] == zone, "area_km2"].values[0]
        for zone in Dzone_list
    ]
    Darea_dict = {int(zone): Darea_list[i] for i, zone in enumerate(Dzone_list)}

    data = pd.DataFrame(
        [
            (year, int(zone))
            for year in range(config.start_year, config.end_year + 1)
            for zone in Dzone_list
        ],
        columns=["year", "zone"],
    )
    data["pop_dense_t"] = (
        data.apply(
            lambda x: pop_dict_D[x["year"]]["人口"][x["zone"]]
            / Darea_dict[int(x["zone"])],
            axis=1,
        ).values
        / config.scale_pop_dense
    )
    data["population_t_raw"] = data.apply(
        lambda x: pop_dict_D[x["year"]]["人口"][x["zone"]], axis=1
    ).values
    data["population_t"] = data["population_t_raw"] / config.scale_pop
    data["los_t_raw"] = data.apply(
        lambda x: los_dict_D[x["year"]][x["zone"]]["total"], axis=1
    )
    data["los_t"] = data["los_t_raw"] / config.scale_los
    data["develop_res_t"] = (
        data.apply(
            lambda x: develop_dict_D[x["year"]]["面積"].get((x["zone"], 1), 0), axis=1
        )
        / config.scale_dev_res
    )
    data["develop_res_t_raw"] = data["develop_res_t"] * config.scale_dev_res
    data["develop_shop_t"] = (
        data.apply(
            lambda x: develop_dict_D[x["year"]]["面積"].get((x["zone"], 2), 0),
            axis=1,
        ).astype(float)
        / config.scale_dev_shop
    )
    data["land_price_t"] = data.apply(
        lambda x: Dzoning.loc[
            Dzoning["選択ゾーン"] == x["zone"], f"LandPrice_{int(x['year'])}"
        ].astype(float).values[0],
        axis=1,
    ) / config.scale_price
    data["land_price_t_raw"] = data["land_price_t"] * config.scale_price
    data["distance_CBD"] = data.apply(
        lambda x: transition_model.dist_convert_D(x["zone"], 0), axis=1
    )
    data["risk"] = data.apply(
        lambda x: Dzoning.loc[
            Dzoning["選択ゾーン"] == x["zone"], ["tsunami_area", "sinsui_area"]
        ].sum(axis=1).values[0],
        axis=1,
    )

    # Fixed effects.
    zone_d = pd.get_dummies(data["zone"], prefix="zone", dtype=float)
    year_d = pd.get_dummies(data["year"], prefix="year", dtype=float)
    data = pd.concat([data, zone_d, year_d], axis=1)
    data["const"] = 1.0

    columns_zone_fe = [col for col in data.columns if col.startswith("zone_")]
    columns_year_fe = [col for col in data.columns if col.startswith("year_")]

    # Standard deviations used when constructing perturbed policy models.
    los_std = [
        data.loc[data["zone"] == zone, "los_t_raw"].std() for zone in Dzone_list
    ]
    develop_std = [
        data.loc[data["zone"] == zone, "develop_res_t_raw"].std()
        for zone in Dzone_list
    ]
    los_std = [1e-3 if std == 0 or np.isnan(std) else std for std in los_std]
    develop_std = [1e-3 if std == 0 or np.isnan(std) else std for std in develop_std]

    # Feature arrays used inside forward simulation.
    develop_shop_t = data.groupby(["year", "zone"])["develop_shop_t"].sum().unstack().values
    distance_CBD = data.groupby(["year", "zone"])["distance_CBD"].mean().unstack().values
    risk = data.groupby(["year", "zone"])["risk"].mean().unstack().values
    features = [develop_shop_t, distance_CBD, risk]

    return SimulationDataBundle(
        transition_model=transition_model,
        df_pt=df_pt,
        data=data,
        Dzone_list=Dzone_list,
        Rzone_list=Rzone_list,
        Darea_list=Darea_list,
        columns_zone_fe=columns_zone_fe,
        columns_year_fe=columns_year_fe,
        los_std=los_std,
        develop_std=develop_std,
        features=features,
    )
