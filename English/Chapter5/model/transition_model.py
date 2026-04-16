"""Core module for residential choice and population transitions."""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import Rook
import sys
from scipy.optimize import minimize
# Allow running from the parent directory during local scripts.
sys.path.append("..")

from model.mnl import MNL, MNL_2step
from model.mxl import MXL
from model.data_reading import read_zone_code, read_pt_data, read_pop_data, read_building_data, read_los_data, distance_matrix, read_obs_share_data, read_move_data
from utils.config_manager import ConfigManager

class TransitionModel:
    """Main container for the Chapter 5 transition model."""
    STATIC_FEATURE_COLUMNS = [
        "tsunami_area",
        "sinsui_area",
        "school_distance",
        "kosodate_density",
        "hospital_density",
        "park_area",
        "commercial_area",
        "risk_area",
        "lowuse_area",
    ]

    def __init__(
        self,
        start_year: int,
        end_year: int,
        ref_year: int,
        dev_zone: str,
        res_zone: str,
        config: Optional[ConfigManager] = None,
    ):
        assert start_year - ref_year + 1 >= 2008, "start_year - ref_year + 1 must be greater than or equal to 2008"
        self.start_year = start_year
        self.end_year = end_year
        self.ref_year = ref_year
        self.dev_zone = dev_zone
        self.res_zone = res_zone
        self.config = config or ConfigManager("config.yaml")
        self.set_init()
        self.set_mxl()
        #self.set_mnl()
        self.set_zone_correspondence()
        self.set_distance_matrix()
        self.set_spatial_weight_matrix(type="distance") # contiguity or distance

    def set_init(self):
        """Load scaling constants and baseline datasets used throughout the model."""
        # Scaling and configuration values.
        self.num_features_gov = self.config.num_features_gov
        self.num_features_dev = self.config.num_features_dev
        self.scale_pop_dense = self.config.scale_pop_dense
        self.scale_pop = self.config.scale_pop
        self.scale_los = self.config.scale_los
        self.scale_dev_res = self.config.scale_dev_res
        self.scale_dev_shop = self.config.scale_dev_shop
        self.scale_price = self.config.scale_price
        self.scale_land_demand = self.config.scale_land_demand

        self.zone_code_in, self.zone_code_all, self.zoning = read_zone_code(self.res_zone)
        self.distance_df = distance_matrix(self.zoning, self.res_zone)
        self.zone_list = np.unique(self.zoning["選択ゾーン"].values).tolist()
        print(f"Number of zones: {len(self.zone_list)}")
        self.area_dict = {zone: self.zoning.loc[self.zoning["選択ゾーン"]==zone, "area_km2"].values[0] for zone in self.zone_list}
        self.pop_dict = read_pop_data(self.start_year, self.end_year, self.res_zone)
        self.develop_dict = read_building_data(self.start_year-self.ref_year-1, self.end_year, self.res_zone)
        self.los_dict = read_los_data(self.start_year, self.end_year, self.res_zone)
        self.los_dict_D = read_los_data(self.start_year, self.end_year, self.dev_zone)
        self.city_center_zone = 0  # Central reference zone.
        self.dist_CBD_threshold = 3.0 / 10  # Distance threshold in model scale.
        self.center_distances = np.array([self.dist_convert(zone, self.city_center_zone) for zone in self.zone_list])
        self.static_features = self.zoning[self.STATIC_FEATURE_COLUMNS].values
        self.move_ratio, self.exiting_ratio, self.inflow_pop = read_move_data(self.zoning)
        
        np.random.seed(self.config.seed)
    
    def set_mxl(self):
        """Define variables and dimensions used by the MXL specification."""
        self.var_L_1_hetero = ["dist_CBD", "los_all", "flood_risk", "school", "commercial_area"] # "dist_CBD", "dist_CBD<threshold", "dist_CBD>threshold", "los_all", "los_bus", "los_train", "flood_risk", "school", "child_care", "hospitals", "park_area", "commercial_area"
        self.var_L_1_mean = ["dist_prev"] # "dist_prev"
        self.var_L_2 = ["dist_CBD", "los_all", "flood_risk", "school", "commercial_area", "price"] # "dist_CBD", "dist_CBD<threshold", "dist_CBD>threshold", "los_all", "los_bus", "los_train", "flood_risk", "school", "child_care", "hospitals", "park_area", "commercial_area", "price"
        self.var_Z = ["age", "car", "family", "move_in"] # "age_head", "car_ownership", "has_children", "income", "move_in"
        self.var_W = ["lowuse_area"] # "risk_area", "lowuse_area"
        # Mask controlling which random-coefficient terms interact with household attributes.
        self.XZ_mask = np.array([
            [0, 0, 0, 0, 1],  # Household-head age.
            [0, 1, 0, 0, 0],  # Car ownership.
            [0, 0, 0, 1, 1],  # Children in household.
            [1, 0, 0, 0, 0],  # In-mover indicator.
        ], dtype=int)
                
        self.J = len(self.zone_list) # Number of alternatives
        self.L_1_hetero = len(self.var_L_1_hetero)  # Number of zonal features in X1_hetero
        self.L_1_mean = len(self.var_L_1_mean)  # Number of zonal features in X1_mean
        self.L_2 = len(self.var_L_2)  # Number of zonal features in X2
        self.K = len(self.var_Z)  # Number of personal attributes in Z
        self.T = self.end_year - self.start_year + 1  # Number of time steps
        assert self.XZ_mask.shape == (self.K, self.L_1_hetero), "XZ_mask shape mismatch"
    
    def set_mnl(self):
        """Define the variable set used by the MNL benchmark."""
        # Variables for the MNL specification.
        self.var_mnl = ["dist_prev", "dist_CBD", "los_all", "flood_risk", "school", "commercial_area", "price"]
        self.asc_cols = None
        self.J = len(self.zone_list) # Number of alternatives
        
    def set_zone_correspondence(self):
        """Precompute the correspondence between development and residential zones."""
        if self.dev_zone != self.res_zone:
            self.Dzone_code_in, self.Dzone_code_all, self.Dzoning = read_zone_code(self.dev_zone)
            self.distance_df_D = distance_matrix(self.Dzoning, self.dev_zone)
            self.Dzone_list = np.unique(self.Dzoning["選択ゾーン"].values).tolist()
            Rzone_size = self.zoning["area_km2"].values
            base_path = Path(__file__).resolve().parent.parent
            gdf = gpd.read_file(base_path / "data/raw/zoning/census_zone.geojson")     
            self.zone_conversion = np.array([[gdf.loc[(gdf["S_NAME"].isin(self.Dzone_code_in.loc[self.Dzone_code_in["選択ゾーン"]==Dzone, "町丁字名"]))&(gdf["S_NAME"].isin(self.zone_code_in.loc[self.zone_code_in["選択ゾーン"]==Rzone, "町丁字名"])), "area_km2"].sum() for Rzone in self.zone_list] for Dzone in self.Dzone_list]) / Rzone_size[np.newaxis,:]  # (Dzone, Rzone) area-share mapping. Each column sums to 1.
            self.develop_distribution = [[self.develop_dict[self.start_year]["面積"].get((zone, 1), 0) * self.zone_conversion[Dzone,zone] for zone in self.zone_list] for Dzone in self.Dzone_list]
            self.develop_distribution = np.array([np.array(self.develop_distribution[i]) / sum(self.develop_distribution[i]) for i in self.Dzone_list])
            self.investment_distribution = [[self.los_dict[self.start_year][zone]["total"] * self.zone_conversion[Dzone,zone] for zone in self.zone_list] for Dzone in self.Dzone_list]  # Allocate small-zone investment to the matching larger development zone.
            self.investment_distribution = np.array([np.array(self.investment_distribution[i]) / sum(self.investment_distribution[i]) for i in self.Dzone_list])  # Within each development zone, distribution across residential zones.
            self.population_distribution = [[self.pop_dict[self.start_year]["人口"][zone] * self.zone_conversion[Dzone,zone] for zone in self.zone_list] for Dzone in self.Dzone_list]
            self.population_distribution = np.array([np.array(self.population_distribution[i]) / sum(self.population_distribution[i]) for i in self.Dzone_list])  # Population shares across residential zones within each development zone.
        else:
            self.Dzone_code_in = self.zone_code_in.copy()
            self.Dzone_code_all = self.zone_code_all.copy()
            self.Dzoning = self.zoning.copy()
            self.Dzone_list = self.zone_list.copy()
            self.zone_conversion = np.eye(self.J)
            self.develop_distribution = np.eye(self.J)
            self.investment_distribution = np.eye(self.J)
            self.population_distribution = np.eye(self.J)
            
        # Convert in-movers into land demand.
        self.land_demand_zone_R = np.ones(self.J) * self.config.new_house_ratio * self.config.per_person_land_demand
        self.land_demand_zone_D = np.ones(len(self.Dzone_list)) * self.config.new_house_ratio * self.config.per_person_land_demand
        # https://www.pref.ehime.jp/uploaded/attachment/141646.pdf
        
        self.D_to_R_idx = [np.flatnonzero(self.zone_conversion[d, :] > 0)
                        for d in range(self.zone_conversion.shape[0])]  # Residential-zone indices that belong to each development zone.
    
        self.price_data_R = np.zeros((self.J, self.T))
        for zone_idx, zone in enumerate(self.zone_list):
            temp = self.zoning.loc[self.zoning["選択ゾーン"]==zone]
            for year_idx, year in enumerate(range(self.start_year, self.end_year + 1)):
                self.price_data_R[zone_idx, year_idx] = temp[f"UnitPrice_Attached_{year}"].values[0] if not np.isnan(temp[f"UnitPrice_Attached_{year}"].values[0]) else temp[f"UnitPrice_Condo_{year}"].values[0]
                self.price_data_R[zone_idx, year_idx] /= self.scale_price

        self.investment_data_R = np.zeros((self.J, self.T))
        for zone_idx, zone in enumerate(self.zone_list):
            for year_idx, year in enumerate(range(self.start_year, self.end_year + 1)):
                self.investment_data_R[zone_idx, year_idx] = self.los_dict[year][zone]["total"]
    
    def set_distance_matrix(self):
        """Precompute the zone-to-zone distance matrix and cache it on the instance."""
        # Store pairwise distances for repeated use.
        if not hasattr(self, '_distance_matrix'):
            self._distance_matrix = np.zeros((self.J, self.J))
            for i, zone_i in enumerate(self.zone_list):
                for j, zone_j in enumerate(self.zone_list):
                    self._distance_matrix[i, j] = self.dist_convert(zone_i, zone_j)   
    
    def set_spatial_weight_matrix(self, type="contiguity"):
        """
        Set spatial weight matrix W based on the specified type.
        Args:
            type (str): Type of spatial weight matrix ("contiguity" or "distance").
        """
        if type == "contiguity":
            W = Rook.from_dataframe(self.zoning, use_index=False).full()[0].astype(int) # Rook隣接
        elif type == "distance":
            W = np.zeros((self.J, self.J))
            for i, zone_i in enumerate(self.zone_list):
                for j, zone_j in enumerate(self.zone_list):
                    if i != j:
                        distance = self._distance_matrix[i, j]
                        W[i, j] = np.exp(-distance) # 1 / distance if distance > 0 else 0 # 
        # Row-standardize W
        row_sums = W.sum(axis=1, keepdims=True)
        W = np.divide(W, row_sums, where=row_sums != 0)
        self.spatial_weight_matrix = W
    
    @staticmethod
    def _ensure_directory(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def _save_numpy_arrays(self, folder_path: Path, arrays: Dict[str, np.ndarray]) -> None:
        self._ensure_directory(folder_path)
        for filename, array in arrays.items():
            np.save(folder_path / filename, array)
        
    def dist_convert(self, zone_before, zone_after):
        """Return the scaled distance between two residential zones."""
        d = self.distance_df.loc[(self.distance_df["zone_1"]==zone_before) & (self.distance_df["zone_2"]==zone_after), "distance_km"].values[0]
        #d = np.log(1 + d) if d > 0 else 0
        d /= 10  # Scale to the model's units.
        return d
    
    def dist_convert_D(self, zone_before, zone_after):
        """Return the scaled distance between two development zones."""
        d = self.distance_df_D.loc[(self.distance_df_D["zone_1"]==zone_before) & (self.distance_df_D["zone_2"]==zone_after), "distance_km"].values[0]
        #d = np.log(1 + d) if d > 0 else 0
        d /= 10
        return d
    
    def los_convert(self, los):
        """Convert LOS into the model's internal scale."""
        # los = np.log(los) if los > 0 else 0
        return los / self.scale_los
    
    def make_estimation_data_mnl(self):
        """
        Generate estimation data for the multinomial logit model.

        Returns:
            X (ndarray): Feature array of shape (n_samples, n_alternatives, n_features)
            y (ndarray): Chosen alternative indices
        """
        df_pool = read_pt_data(self.start_year, self.end_year, self.res_zone)
        df_pool = df_pool[df_pool["転居有無"]==1]
        N = len(df_pool) # Number of samples
        X = np.zeros((N, self.J, len(self.var_mnl)))
        for alt in self.zone_list:
            if "dist_prev" in self.var_mnl:
                var_index = self.var_mnl.index("dist_prev")
                X[:, alt, var_index] = 0.0
                mask = df_pool["居住地_前_ゾーン"] != -1.0
                X[mask, alt, var_index] = df_pool.loc[mask, "居住地_前_ゾーン"].apply(lambda x: self.dist_convert(x, alt)).values
            if "dist_CBD" in self.var_mnl:
                var_index = self.var_mnl.index("dist_CBD")
                X[:, alt, var_index] = self.dist_convert(alt, self.city_center_zone)
            if "dist_CBD<threshold" in self.var_mnl:
                var_index = self.var_mnl.index("dist_CBD<threshold")
                X[:, alt, var_index] = np.maximum(0, np.minimum(self.dist_convert(alt, self.city_center_zone), self.dist_CBD_threshold))
            if "dist_CBD>threshold" in self.var_mnl:
                var_index = self.var_mnl.index("dist_CBD>threshold")
                X[:, alt, var_index] = np.maximum(0, self.dist_convert(alt, self.city_center_zone) - self.dist_CBD_threshold)
            if "los_bus" in self.var_mnl:
                var_index = self.var_mnl.index("los_bus")
                X[:, alt, var_index] = df_pool["年"].apply(lambda x: self.los_convert(self.los_dict[x][alt]["bus"])).values
            if "los_train" in self.var_mnl:
                var_index = self.var_mnl.index("los_train")
                X[:, alt, var_index] = df_pool["年"].apply(lambda x: self.los_convert(self.los_dict[x][alt]["shinai"] + self.los_dict[x][alt]["kougai"])).values
            if "los_all" in self.var_mnl:
                var_index = self.var_mnl.index("los_all")
                X[:, alt, var_index] = df_pool["年"].apply(lambda x: self.los_convert(self.los_dict[x][alt]["total"])).values
            if "price" in self.var_mnl:
                var_index = self.var_mnl.index("price")
                temp = self.zoning.loc[self.zoning["選択ゾーン"]==alt]
                X[:, alt, var_index] = df_pool["年"].apply(lambda x: temp[f"UnitPrice_Attached_{x}"].values[0] if not np.isnan(temp[f"UnitPrice_Attached_{x}"].values[0]) else temp[f"UnitPrice_Condo_{x}"].values[0]).values / self.scale_price
            
        # Time-invariant features shared across all samples.
        static_features = self.static_features
        if "flood_risk" in self.var_mnl:
            var_index = self.var_mnl.index("flood_risk")
            X[:, :, var_index] = np.tile(static_features[:, 0] + static_features[:, 1], (N, 1))
        if "school" in self.var_mnl:
            var_index = self.var_mnl.index("school")
            X[:, :, var_index] = np.tile(static_features[:, 2], (N, 1))
        if "child_care" in self.var_mnl:
            var_index = self.var_mnl.index("child_care")
            X[:, :, var_index] = np.tile(np.log(1 + static_features[:, 3]), (N, 1))
        if "hospitals" in self.var_mnl:
            var_index = self.var_mnl.index("hospitals")
            X[:, :, var_index] = np.tile(np.log(1 + static_features[:, 4]), (N, 1))
        if "park_area" in self.var_mnl:
            var_index = self.var_mnl.index("park_area")
            X[:, :, var_index] = np.tile(np.log(1 + 100 * static_features[:, 5]), (N, 1))
        if "commercial_area" in self.var_mnl:
            var_index = self.var_mnl.index("commercial_area")
            X[:, :, var_index] = np.tile(10 * static_features[:, 6], (N, 1))
            
        # Save the design matrix for reuse.
        base_path = Path(__file__).resolve().parent.parent
        folder_path = base_path / f"data/processed/{self.res_zone}"
        self._save_numpy_arrays(
            folder_path,
            {f"X_mnl_{self.start_year}_{self.end_year}.npy": X},
        )
        
        y = df_pool["居住地_後_ゾーン"].values.astype(int)
        return X, y, self.asc_cols

    def generate_prediction_data_mnl(self, investment_list):
        df_pool = pd.DataFrame({
            #"居住地_前_ゾーン": np.array([item for sublist in [[idx] * int(zone_population) for idx, zone_population in enumerate(population_list)] for item in sublist]), # disaggregated
            "居住地_前_ゾーン": [zone for zone in self.zone_list]  # Aggregated origin-zone representation.
        })
        N = len(df_pool)
        X = np.zeros((N, self.J, 5))
        
        for alt in self.zone_list:
            if "dist_prev" in self.var_mnl:
                var_index = self.var_mnl.index("dist_prev")
                X[:, alt, var_index] = 0.0
                mask = df_pool["居住地_前_ゾーン"] != -1.0
                X[mask, alt, var_index] = df_pool.loc[mask, "居住地_前_ゾーン"].apply(lambda x: self.dist_convert(x, alt)).values
            if "dist_CBD" in self.var_mnl:
                var_index = self.var_mnl.index("dist_CBD")
                X[:, alt, var_index] = self.dist_convert(alt, self.city_center_zone)
            if "dist_CBD<threshold" in self.var_mnl:
                var_index = self.var_mnl.index("dist_CBD<threshold")
                X[:, alt, var_index] = np.maximum(0, np.minimum(self.dist_convert(alt, self.city_center_zone), self.dist_CBD_threshold))
            if "dist_CBD>threshold" in self.var_mnl:
                var_index = self.var_mnl.index("dist_CBD>threshold")
                X[:, alt, var_index] = np.maximum(0, self.dist_convert(alt, self.city_center_zone) - self.dist_CBD_threshold)
            if "los_bus" in self.var_mnl:
                var_index = self.var_mnl.index("los_bus")
                X[:, alt, var_index] = self.los_convert(investment_list[alt]["bus"])
            if "los_train" in self.var_mnl:
                var_index = self.var_mnl.index("los_train")
                X[:, alt, var_index] = self.los_convert(investment_list[alt]["shinai"] + investment_list[alt]["kougai"])
            if "los_all" in self.var_mnl:
                var_index = self.var_mnl.index("los_all")
                X[:, alt, var_index] = self.los_convert(investment_list[alt]["total"])
            if "price" in self.var_mnl:
                var_index = self.var_mnl.index("price")
                temp = self.zoning.loc[self.zoning["選択ゾーン"]==alt]
                X[:, alt, var_index] = temp[f"UnitPrice_Attached_{self.start_year}"].values[0] if not np.isnan(temp[f"UnitPrice_Attached_{self.start_year}"].values[0]) else temp[f"UnitPrice_Condo_{self.start_year}"].values[0]
                X[:, alt, var_index] /= self.scale_price
        
        # Time-invariant features shared across all samples.
        static_features = self.static_features
        if "flood_risk" in self.var_mnl:
            var_index = self.var_mnl.index("flood_risk")
            X[:, :, var_index] = np.tile(static_features[:, 0] + static_features[:, 1], (N, 1))
        if "school" in self.var_mnl:
            var_index = self.var_mnl.index("school")
            X[:, :, var_index] = np.tile(static_features[:, 2], (N, 1))
        if "child_care" in self.var_mnl:
            var_index = self.var_mnl.index("child_care")
            X[:, :, var_index] = np.tile(np.log(1 + static_features[:, 3]), (N, 1))
        if "hospitals" in self.var_mnl:
            var_index = self.var_mnl.index("hospitals")
            X[:, :, var_index] = np.tile(np.log(1 + static_features[:, 4]), (N, 1))
        if "park_area" in self.var_mnl:
            var_index = self.var_mnl.index("park_area")
            X[:, :, var_index] = np.tile(np.log(1 + 100 * static_features[:, 5]), (N, 1))
        if "commercial_area" in self.var_mnl:
            var_index = self.var_mnl.index("commercial_area")
            X[:, :, var_index] = np.tile(10 * static_features[:, 6], (N, 1))
        
        return X
    
    def make_estimation_data_2step(self):
        """
        Generate estimation data for the two-step model.
        Returns:
            X1_hetero (ndarray): Feature array of shape (n_samples, n_alternatives, n_features)
            X1_mean (ndarray): Mean feature array of shape (n_samples, n_alternatives, n_features)
            X2 (ndarray): Zonal features of shape (time_step * n_alternatives, n_zonal_features)
            Z (ndarray): Personal attribute array of shape (n_samples, n_personal_attributes)
            y (ndarray): Chosen alternative indices
            obs_share (ndarray): Observed shares of alternatives of shape (n_samples, n_alternatives)
        """
        
        base_path = Path(__file__).resolve().parent.parent
        folder_path = base_path / f"data/processed/{self.res_zone}"
        """
        if os.path.exists(folder_path + f"/X1_hetero_{self.start_year}_{self.end_year}.npy"):
            # Load preprocessed data if available
            X1_hetero = np.load(os.path.join(folder_path, f"X1_hetero_{self.start_year}_{self.end_year}.npy"))
            X1_mean = np.load(os.path.join(folder_path, f"X1_mean_{self.start_year}_{self.end_year}.npy"))
            X2 = np.load(os.path.join(folder_path, f"X2_{self.start_year}_{self.end_year}.npy"))
            Z = np.load(os.path.join(folder_path, f"Z_{self.start_year}_{self.end_year}.npy"))
            W = np.load(os.path.join(folder_path, f"W_{self.start_year}_{self.end_year}.npy"))
            y = np.load(os.path.join(folder_path, f"y_{self.start_year}_{self.end_year}.npy"))
            obs_share = np.load(os.path.join(folder_path, f"obs_share_{self.start_year}_{self.end_year}.npy"))
            relocation_years = np.load(os.path.join(folder_path, f"relocation_years_{self.start_year}_{self.end_year}.npy"))
            #N = X1_hetero.shape[0]
            return X1_hetero, X1_mean, X2, Z, W, y, obs_share, relocation_year
        """
        
        df_pool = read_pt_data(self.start_year, self.end_year, self.res_zone)
        df_pool = df_pool[df_pool["転居有無"]==1]
        N = len(df_pool) # Number of samples
        
        X1_hetero = np.zeros((N, self.J, self.L_1_hetero))
        X1_mean = np.zeros((N, self.J, self.L_1_mean))
        
        # Precompute reusable arrays for a vectorized implementation.
        years = list(range(self.start_year, self.end_year + 1))
        n_years = len(years)
        
        # Precompute LOS arrays by zone and year.
        los_bus = np.zeros((self.J, n_years))
        los_train = np.zeros((self.J, n_years))
        los_all = np.zeros((self.J, n_years))
        
        for zone_idx, zone in enumerate(self.zone_list):
            for year_idx, year in enumerate(years):
                # LOS data by mode.
                los_value = self.los_dict[year][zone]["bus"]
                los_bus[zone_idx, year_idx] = self.los_convert(los_value)
                los_value = self.los_dict[year][zone]["shinai"] + self.los_dict[year][zone]["kougai"]
                los_train[zone_idx, year_idx] = self.los_convert(los_value)
                los_value = self.los_dict[year][zone]["total"]
                los_all[zone_idx, year_idx] = self.los_convert(los_value)
        
        # Vectorized distance calculations from current residential zones.
        year_indices = df_pool["年"].values.astype(int) - self.start_year
        center_distances = self.center_distances
        
        # Vectorized construction of X1_mean.
        if "dist_prev" in self.var_L_1_mean:
            var_index = self.var_L_1_mean.index("dist_prev")
            prev_zones = df_pool["居住地_前_ゾーン"].values.astype(int)
            X1_mean[prev_zones == -1, :, var_index] = 0.0
            for alt in range(self.J):
                X1_mean[prev_zones != -1, alt, var_index] = self._distance_matrix[prev_zones[prev_zones != -1], alt]
        
        # Vectorized construction of X1_hetero.
        if "dist_CBD" in self.var_L_1_hetero:  # Distance to the city center.
            var_index = self.var_L_1_hetero.index("dist_CBD")
            X1_hetero[:, :, var_index] = np.tile(center_distances, (N, 1))
        if "dist_CBD<threshold" in self.var_L_1_hetero:  # Distance to the center below the threshold.
            var_index = self.var_L_1_hetero.index("dist_CBD<threshold")
            X1_hetero[:, :, var_index] = np.tile(np.maximum(0, np.minimum(center_distances, self.dist_CBD_threshold)), (N, 1))
        if "dist_CBD>threshold" in self.var_L_1_hetero:  # Distance to the center above the threshold.
            var_index = self.var_L_1_hetero.index("dist_CBD>threshold")
            X1_hetero[:, :, var_index] = np.tile(np.maximum(0, center_distances - self.dist_CBD_threshold), (N, 1))
        if "los_bus" in self.var_L_1_hetero:  # Bus LOS.
            var_index = self.var_L_1_hetero.index("los_bus")
            X1_hetero[:, :, var_index] = los_bus[:, year_indices].T
        if "los_train" in self.var_L_1_hetero:  # Rail LOS.
            var_index = self.var_L_1_hetero.index("los_train")
            X1_hetero[:, :, var_index] = los_train[:, year_indices].T
        if "los_all" in self.var_L_1_hetero:  # Total LOS.
            var_index = self.var_L_1_hetero.index("los_all")
            X1_hetero[:, :, var_index] = los_all[:, year_indices].T

        # Time-invariant features shared across all samples.
        static_features = self.static_features
        if "flood_risk" in self.var_L_1_hetero:  # Flood risk.
            var_index = self.var_L_1_hetero.index("flood_risk")
            X1_hetero[:, :, var_index] = np.tile(static_features[:, 0] + static_features[:, 1], (N, 1)) #  > 0).astype(float) > 0).astype(float)
        if "school" in self.var_L_1_hetero:  # Distance to the nearest school.
            var_index = self.var_L_1_hetero.index("school")
            X1_hetero[:, :, var_index] = np.tile(static_features[:, 2], (N, 1))
        if "child_care" in self.var_L_1_hetero:  # Child-care facility density.
            var_index = self.var_L_1_hetero.index("child_care")
            X1_hetero[:, :, var_index] = np.tile(np.log(1 + static_features[:, 3]), (N, 1))
        if "hospitals" in self.var_L_1_hetero:  # Hospital density.
            var_index = self.var_L_1_hetero.index("hospitals")
            X1_hetero[:, :, var_index] = np.tile(np.log(1 + static_features[:, 4]), (N, 1))
        if "park_area" in self.var_L_1_hetero:  # Park area.
            var_index = self.var_L_1_hetero.index("park_area")
            X1_hetero[:, :, var_index] = np.tile(np.log(1 + 100 * static_features[:, 5]), (N, 1))
        if "commercial_area" in self.var_L_1_hetero:  # Commercial land area.
            var_index = self.var_L_1_hetero.index("commercial_area")
            X1_hetero[:, :, var_index] = np.tile(10 * static_features[:, 6], (N, 1))
        
        y = df_pool["居住地_後_ゾーン"].values.astype(int)
        
        # Zonal features for the second-step equation.
        X2 = np.zeros((self.J * self.T, self.L_2))
        
        # Process all zone-year combinations at once.
        zone_indices = np.repeat(np.arange(self.J), n_years)
        year_indices = np.tile(np.arange(n_years), self.J)
        
        if "dist_CBD" in self.var_L_2:  # Distance to the city center.
            var_index = self.var_L_2.index("dist_CBD")
            X2[:, var_index] = np.repeat(center_distances, n_years)
        if "dist_CBD<threshold" in self.var_L_2:  # Distance to the center below the threshold.
            var_index = self.var_L_2.index("dist_CBD<threshold")
            X2[:, var_index] = np.repeat(np.maximum(0, np.minimum(center_distances, self.dist_CBD_threshold)), n_years)
        if "dist_CBD>threshold" in self.var_L_2:  # Distance to the center above the threshold.
            var_index = self.var_L_2.index("dist_CBD>threshold")
            X2[:, var_index] = np.repeat(np.maximum(0, center_distances - self.dist_CBD_threshold), n_years)
        if "los_bus" in self.var_L_2:  # Bus LOS.
            var_index = self.var_L_2.index("los_bus")
            X2[:, var_index] = los_bus[zone_indices, year_indices]
        if "los_train" in self.var_L_2:  # Rail LOS.
            var_index = self.var_L_2.index("los_train")
            X2[:, var_index] = los_train[zone_indices, year_indices]
        if "los_all" in self.var_L_2:  # Total LOS.
            var_index = self.var_L_2.index("los_all")
            X2[:, var_index] = los_all[zone_indices, year_indices]
        
        static_repeated = np.repeat(static_features, n_years, axis=0)
        
        if "flood_risk" in self.var_L_2:  # Flood risk.
            var_index = self.var_L_2.index("flood_risk")
            X2[:, var_index] = static_repeated[:, 0] + static_repeated[:, 1]# > 0).astype(float)
        if "school" in self.var_L_2:  # Distance to the nearest school.
            var_index = self.var_L_2.index("school")
            X2[:, var_index] = static_repeated[:, 2]
        if "child_care" in self.var_L_2:  # Child-care facility density.
            var_index = self.var_L_2.index("child_care")
            X2[:, var_index] = np.log(1 + static_repeated[:, 3])
        if "hospitals" in self.var_L_2:  # Hospital density.
            var_index = self.var_L_2.index("hospitals")
            X2[:, var_index] = np.log(1 + static_repeated[:, 4])
        if "park_area" in self.var_L_2:  # Park area.
            var_index = self.var_L_2.index("park_area")
            X2[:, var_index] = np.log(1 + 100 * static_repeated[:, 5])
        if "commercial_area" in self.var_L_2:  # Commercial land area.
            var_index = self.var_L_2.index("commercial_area")
            X2[:, var_index] = 10 * static_repeated[:, 6]
        if "price" in self.var_L_2:  # Housing price.
            var_index = self.var_L_2.index("price")
            X2[:, var_index] = self.price_data_R[zone_indices, year_indices]
        
        W = np.zeros((self.J * self.T, len(self.var_W)))
        if "risk_area" in self.var_W:  # Disaster-risk area.
            var_index = self.var_W.index("risk_area")
            W[:, var_index] = static_repeated[:, 7]
        if "lowuse_area" in self.var_W:  # Underused land area.
            var_index = self.var_W.index("lowuse_area")
            W[:, var_index] = static_repeated[:, 8]
            
        # Individual attributes enter only the utility of staying or moving as specified by the model.
        Z_dict = {"age": "世帯主年齢", "car": "自動車有無", "family": "子供有無", "income": "世帯年収", "move_in": "市外転入"}
        Z = df_pool[[Z_dict[var] for var in self.var_Z]].copy()
        if "age" in self.var_Z:
            #Z["世帯主年齢"] = Z["世帯主年齢"] / 100
            Z["世帯主年齢"] = (Z["世帯主年齢"] >= 65).astype(float)  # Age 65 or older.
        if "income" in self.var_Z:
            Z["世帯年収"] = Z["世帯年収"] / 1000
        Z = Z.values
        
        # Relocation year index for each sample.
        relocation_years = df_pool["年"].values - self.start_year
        
        # Observed market shares.
        if self.res_zone == "chou_zone":
            obs_share = df_pool.groupby(["居住地_後_ゾーン", "転居年"])["転居有無"].count().unstack().fillna(0).replace(0, 0.1).T
            for i in set(self.zone_list) - set(obs_share.columns.tolist()):
                obs_share.loc[:,i] = 0.1
            # Reorder columns to match the zone ordering.
            obs_share = obs_share[self.zone_list]
            obs_share = obs_share.values
            obs_share /= np.sum(obs_share, axis=1, keepdims=True)
        else:
            if self.start_year >= 2015:
                obs_share = read_obs_share_data(self.start_year, self.end_year, self.zoning, self.res_zone)
            else:
                obs_share1 = df_pool.groupby(["居住地_後_ゾーン", "転居年"])["転居有無"].count().unstack().fillna(0).replace(0, 0.1).T
                for i in set(self.zone_list) - set(obs_share1.columns.tolist()):
                    obs_share1.loc[:,i] = 0.1
                # Reorder columns to match the zone ordering.
                obs_share1 = obs_share1[self.zone_list]
                obs_share1 = obs_share1.values
                obs_share1 /= np.sum(obs_share1, axis=1, keepdims=True)
                obs_share2 = read_obs_share_data(2015, self.end_year, self.zoning, self.res_zone)
                # Use the external observed-share data from 2015 onward.
                obs_share = obs_share1
                obs_share[2015 - self.start_year:] = obs_share2
        
        # Save processed arrays for reuse.
        self._save_numpy_arrays(
            folder_path,
            {
                f"X1_hetero_{self.start_year}_{self.end_year}.npy": X1_hetero,
                f"X1_mean_{self.start_year}_{self.end_year}.npy": X1_mean,
                f"X2_{self.start_year}_{self.end_year}.npy": X2,
                f"Z_{self.start_year}_{self.end_year}.npy": Z,
                f"W_{self.start_year}_{self.end_year}.npy": W,
                f"y_{self.start_year}_{self.end_year}.npy": y,
                f"obs_share_{self.start_year}_{self.end_year}.npy": obs_share,
                f"relocation_years_{self.start_year}_{self.end_year}.npy": relocation_years,
            },
        )
        
        return X1_hetero, X1_mean, X2, Z, W, y, obs_share, relocation_years


    def generate_prediction_data_2step(self, df_year, investment_lists, price_lists, year):
        assert "los_bus" not in self.var_L_1_hetero or "los_all" not in self.var_L_1_hetero, "los_bus and los_all cannot be used at the same time"
        N = len(df_year)  # Number of samples
        X1_hetero = np.zeros((N, self.J, self.L_1_hetero))
        X1_mean = np.zeros((N, self.J, self.L_1_mean))        
        # Reuse the cached distance matrix or build it if needed.
        if not hasattr(self, '_distance_matrix'):
            self._distance_matrix = np.zeros((self.J, self.J))
            for i, zone_i in enumerate(self.zone_list):
                for j, zone_j in enumerate(self.zone_list):
                    self._distance_matrix[i, j] = self.dist_convert(zone_i, zone_j)
        
        # Indices of previous residential zones.
        prev_zones = df_year["居住地_前_ゾーン"].values.astype(int)
        
        # X1_mean.
        if "dist_prev" in self.var_L_1_mean:
            var_index = self.var_L_1_mean.index("dist_prev")
            X1_mean[prev_zones == -1, :, var_index] = 0.0
            for alt in range(self.J):
                X1_mean[prev_zones != -1, alt, var_index] = self._distance_matrix[prev_zones[prev_zones != -1], alt]
        
        # X1_hetero.
        center_distances = self.center_distances
        if "dist_CBD" in self.var_L_1_hetero:
            var_index = self.var_L_1_hetero.index("dist_CBD")
            X1_hetero[:, :, var_index] = np.tile(center_distances, (N, 1))  # Distance to the city center.
        if "dist_CBD<threshold" in self.var_L_1_hetero:
            var_index = self.var_L_1_hetero.index("dist_CBD<threshold")
            X1_hetero[:, :, var_index] = np.tile(np.maximum(0, np.minimum(center_distances, self.dist_CBD_threshold)), (N, 1))  # Distance below the threshold.
        if "dist_CBD>threshold" in self.var_L_1_hetero:
            var_index = self.var_L_1_hetero.index("dist_CBD>threshold")
            X1_hetero[:, :, var_index] = np.tile(np.maximum(0, center_distances - self.dist_CBD_threshold), (N, 1))  # Distance above the threshold.
                
        price_array = np.array(price_lists)        
        # Investment data aggregated back to residential zones.
        investment_Rzone = np.zeros(self.J)
        for Dzone in self.Dzone_list:
            investment_Rzone += investment_lists[Dzone] * self.investment_distribution[Dzone]
        investment_Rzone = np.maximum(investment_Rzone, 1)
        if "los_all" in self.var_L_1_hetero:
            var_index = self.var_L_1_hetero.index("los_all")
            X1_hetero[:, :, var_index] = np.tile(self.los_convert(investment_Rzone), (N, 1))
        if "los_bus" in self.var_L_1_hetero:
            var_index = self.var_L_1_hetero.index("los_bus")
            X1_hetero[:, :, var_index] = np.tile(self.los_convert(investment_Rzone), (N, 1))
        
        los_train = np.zeros(self.J)
        year_index = year if year <= self.end_year else self.end_year
        for zone_idx, zone in enumerate(self.zone_list):
            los_value = self.los_dict[year_index][zone]["shinai"] + self.los_dict[year_index][zone]["kougai"]
            los_train[zone_idx] = self.los_convert(los_value)
        if "los_train" in self.var_L_1_hetero:
            var_index = self.var_L_1_hetero.index("los_train")
            X1_hetero[:, :, var_index] = np.tile(los_train, (N, 1))  # Rail LOS.
        
        # Static features shared across samples.
        static_features = self.static_features[:, :7]
        
        if "flood_risk" in self.var_L_1_hetero:
            var_index = self.var_L_1_hetero.index("flood_risk")
            X1_hetero[:, :, var_index] = np.tile(static_features[:, 0] + static_features[:, 1], (N, 1))  # Flood risk.
        if "school" in self.var_L_1_hetero:
            var_index = self.var_L_1_hetero.index("school")
            X1_hetero[:, :, var_index] = np.tile(static_features[:, 2], (N, 1))  # School distance.
        if "child_care" in self.var_L_1_hetero:
            var_index = self.var_L_1_hetero.index("child_care")
            X1_hetero[:, :, var_index] = np.tile(np.log(1 + static_features[:, 3]), (N, 1))  # Child-care facilities.
        if "hospitals" in self.var_L_1_hetero:
            var_index = self.var_L_1_hetero.index("hospitals")
            X1_hetero[:, :, var_index] = np.tile(np.log(1 + static_features[:, 4]), (N, 1))  # Hospitals.
        if "park_area" in self.var_L_1_hetero:
            var_index = self.var_L_1_hetero.index("park_area")
            X1_hetero[:, :,var_index] = np.tile(np.log(1 + 100 * static_features[:, 5]), (N, 1))  # Parks.
        if "commercial_area" in self.var_L_1_hetero:
            var_index = self.var_L_1_hetero.index("commercial_area")
            X1_hetero[:, :, var_index] = np.tile(10 * static_features[:, 6], (N, 1))  # Commercial land.

        # Vectorized x2 construction.
        x2 = np.zeros((self.J, self.L_2))
        if "dist_CBD" in self.var_L_2:
            var_index = self.var_L_2.index("dist_CBD")
            x2[:, var_index] = center_distances  # Distance to the city center.
        if "dist_CBD<threshold" in self.var_L_2:
            var_index = self.var_L_2.index("dist_CBD<threshold")
            x2[:, var_index] = np.maximum(0, np.minimum(center_distances, self.dist_CBD_threshold))  # Distance below the threshold.
        if "dist_CBD>threshold" in self.var_L_2:
            var_index = self.var_L_2.index("dist_CBD>threshold")
            x2[:, var_index] = np.maximum(0, center_distances - self.dist_CBD_threshold)  # Distance above the threshold.
        if "los_bus" in self.var_L_2:
            var_index = self.var_L_2.index("los_bus")
            x2[:, var_index] = self.los_convert(investment_Rzone)
        if "los_all" in self.var_L_2:
            var_index = self.var_L_2.index("los_all")
            x2[:, var_index] = self.los_convert(investment_Rzone)
        if "los_train" in self.var_L_2:
            var_index = self.var_L_2.index("los_train")
            x2[:, var_index] = los_train  # Rail LOS.
        if "flood_risk" in self.var_L_2:
            var_index = self.var_L_2.index("flood_risk")
            x2[:, var_index] = static_features[:, 0] + static_features[:, 1]   # Flood risk.
        if "school" in self.var_L_2:
            var_index = self.var_L_2.index("school")
            x2[:, var_index] = static_features[:, 2]  # School distance.
        if "child_care" in self.var_L_2:
            var_index = self.var_L_2.index("child_care")
            x2[:, var_index] = np.log(1 + static_features[:, 3])  # Child-care facilities.
        if "hospitals" in self.var_L_2:
            var_index = self.var_L_2.index("hospitals")
            x2[:, var_index] = np.log(1 + static_features[:, 4])  # Hospitals.
        if "park_area" in self.var_L_2:
            var_index = self.var_L_2.index("park_area")
            x2[:, var_index] = np.log(1 + 100 * static_features[:, 5])  # Parks.
        if "commercial_area" in self.var_L_2:
            var_index = self.var_L_2.index("commercial_area")
            x2[:, var_index] = 10 * static_features[:, 6]  # Commercial land.
        if "price" in self.var_L_2:
            var_index = self.var_L_2.index("price")
            x2[:, var_index] = price_array / self.scale_price  # Housing price.
        
        # Individual attributes used in the prediction step.
        Z_dict = {"age": "世帯主年齢", "car": "自動車有無", "family": "子供有無", "income": "世帯年収", "move_in": "市外転入"}
        Z = df_year[[Z_dict[var] for var in self.var_Z]].copy()
        if "age" in self.var_Z:
            #Z["世帯主年齢"] = Z["世帯主年齢"] / 100
            Z["世帯主年齢"] = (Z["世帯主年齢"] >= 65).astype(float)  # Age 65 or older.
        if "income" in self.var_Z:
            Z["世帯年収"] = Z["世帯年収"] / 1000
        Z = Z.values
            
        base_path = Path(__file__).resolve().parent.parent
        folder_path = base_path / "data/processed/pred"
        self._save_numpy_arrays(
            folder_path,
            {
                f"X1_hetero_{self.start_year}_{self.end_year}.npy": X1_hetero,
                f"X1_mean_{self.start_year}_{self.end_year}.npy": X1_mean,
                f"X2_{self.start_year}_{self.end_year}.npy": x2,
                f"Z_{self.start_year}_{self.end_year}.npy": Z,
            },
        )
        
        return X1_hetero, X1_mean, x2, Z

    def estimate_step2_model(self, step1_path, X2, W=None, method="IV", model="MNL_2step"):
        """
        Parameters:
            step1_path (str): Path to the saved step 1 model
            X2 (ndarray): Feature array for the second step of 2-step model
            W (ndarray): Instrumental variable array for the second step of 2-step model
            method (str): Estimation method, either 
                - "IV" for instrumental variables
                - "OLS" for ordinary least squares
            model (str): Model type, either
                - "MNL_2step" for two-step multinomial logit
                - "MXL_2step" for two-step mixed logit
        Returns:
            beta: Estimated parameter 
            model: Fitted model object
        """
        self.load_estimates(step1_path, step2_path=None, model=model)
        #res2 = self.model.fit_step2(X2)
        res2 = self.model.fit_step2(X2, W=W, method=method)
        return res2.params, self.model

    def estimate_choice_model(self, y, method="mnl", X=None, asc_cols=None, X1_hetero=None, X1_mean=None, X2=None, Z=None, W=None, obs_share=None, relocation_years=None):
        """
        Estimate a multinomial logit model for individual relocation decision.

        Parameters:
            X (ndarray): Feature array of shape (n_samples, n_alternatives, n_features)
            X1_hetero (ndarray): Feature array for the first step of 2-step model
            X1_mean (ndarray): Mean feature array for the first step of 2-step model
            X2 (ndarray): Feature array for the second step of 2-step model
            Z (ndarray): Personal attribute array of shape (n_samples, n_personal_attributes)
            relocation_years (ndarray): Array of relocation years for each sample
            method (str): Estimation method, either 
                - "MNL" for multinomial logit
                - "MNL_2step" for two-step estimation
                - "MXL_2step" for mixed logit
            asc_cols (list): List of columns to be used as alternative specific constants
            y (ndarray): Chosen alternative indices
            obs_share (ndarray): observed share of each alternatives in each market
        Returns:
            beta: Estimated parameter matrix of shape (n_alternatives, n_features)
        """
        if method == "MNL":
            assert X is not None, "X must be provided for MNL estimation"
            self.model = MNL()
            # Fit the model
            res = self.model.fit(X, y, asc_cols)
            return res.x, self.model
        
        elif method == "MNL_2step":
            assert X1_hetero is not None and X1_mean is not None and X2 is not None and Z is not None and relocation_years is not None, "X1_hetero, X2, Z, and relocation_years must be provided for 2-step estimation"
            self.model = MNL_2step(time_steps=self.end_year - self.start_year + 1)
            print("Sample size:", X1_hetero.shape[0])
            # Fit the model
            res = self.model.fit_step1(X1_hetero, X1_mean, Z, self.XZ_mask, y, obs_share, relocation_years)
            print("Estimated Beta mean:", np.round(res.x[:self.L_1_mean], 2))
            beta = res.x[self.L_1_mean:]
            mask = np.cumsum(self.XZ_mask.flatten()) * self.XZ_mask.flatten()
            beta_extend = np.array([beta[i-1] if i > 0 else 0 for i in mask])
            print("Estimated Beta hetero:", np.round(beta_extend.reshape(self.K, self.L_1_hetero), 2))
            print("T-values mean:", np.round(res.tval[:self.L_1_mean], 2))
            tval = res.tval[self.L_1_mean:]
            tval_extend = np.array([tval[i-1] if i > 0 else 0 for i in mask])
            print("T-values hetero:", np.round(tval_extend.reshape(self.K, self.L_1_hetero), 2))
            print("Likelihood Ratio:", res.likelihood)
            print("Adjusted Likelihood Ratio:", res.adjusted_likelihood)
            #res2 = self.model.fit_step2(X2)
            res2 = self.model.fit_step2(X2, W=W, method="OLS")
            return res2.params, self.model
        
        elif method == "MXL_2step":
            assert X1_hetero is not None and X1_mean is not None and X2 is not None and Z is not None and relocation_years is not None, "X1_hetero, X2, Z, and relocation_years must be provided for 2-step estimation"
            self.model = MXL(time_steps=self.end_year - self.start_year + 1)
            print("Sample size:", X1_hetero.shape[0])
            # Fit the model
            res = self.model.fit_step1(X1_hetero, X1_mean, Z, self.XZ_mask, self.spatial_weight_matrix, y, obs_share, relocation_years)
            print("Estimated Beta mean:", np.round(res.x[:self.L_1_mean], 2))
            print("T-values mean:", np.round(res.tval[:self.L_1_mean], 2))
            
            beta = res.x[self.L_1_mean:(self.L_1_mean+np.sum(self.XZ_mask))]
            mask = np.cumsum(self.XZ_mask.flatten()) * self.XZ_mask.flatten()
            beta_extend = np.array([beta[i-1] if i > 0 else 0 for i in mask])
            print("Estimated Beta hetero:", np.round(beta_extend.reshape(self.K, self.L_1_hetero), 2))
            tval = res.tval[self.L_1_mean:(self.L_1_mean+np.sum(self.XZ_mask))]
            tval_extend = np.array([tval[i-1] if i > 0 else 0 for i in mask])
            print("T-values hetero:", np.round(tval_extend.reshape(self.K, self.L_1_hetero), 2))
            
            SAR_par = res.x[-1] # [(self.L_1_mean + np.sum(self.XZ_mask)):]
            print("Estimated sigma and SAR parameter:", np.round(SAR_par, 2))
            tval = res.tval[-1]
            print("T-values SAR:", np.round(tval, 2))
            
            print("Likelihood Ratio:", res.likelihood)
            print("Adjusted Likelihood Ratio:", res.adjusted_likelihood)
            
            res2 = self.model.fit_step2(X2)
            #res2 = self.model.fit_step2(X2, W=W, method="IV")
            return res2.params, self.model
    
    def predict_disequilibrium_population(self, df, population_lists, development_lists, investment_lists, price_lists, year, move_ratio, zoning_type="dev_zone"):
        """
        Predict population transition using the estimated model using disaggregated data and randomness.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        if year <= self.end_year:
            df_year = df[(df["転居年"]==year) & (df["転居有無"]==1)].copy()
        else:
            # Sample a fraction of non-movers when extrapolating beyond the observed period.
            df_year = df[df["転居年"]==0].sample(frac=move_ratio).copy()
        X1_hetero, X1_mean, x2, Z = self.generate_prediction_data_2step(df_year, investment_lists, price_lists, year)
        
        predicted_probs = self.model.predict(X1_hetero, X1_mean, x2, Z, self.XZ_mask, self.spatial_weight_matrix) # (N, J)
        predicted_move = predicted_probs * df_year["拡大係数"].values.reshape(-1, 1)  # (N, J)
        
        # Aggregate predicted inflows and outflows by zone.
        plus_Rzone = predicted_move.sum(axis=0)
        minus_Rzone = np.array([df_year.loc[df_year["居住地_前_ゾーン"]==zone, "拡大係数"].sum() for zone in self.zone_list])
                
        if zoning_type == "dev_zone":
            plus_Dzone = np.sum(plus_Rzone[np.newaxis,:] * self.zone_conversion, axis=1)
            minus_Dzone = np.sum(minus_Rzone[np.newaxis,:] * self.zone_conversion, axis=1)
            predicted_population = np.array(population_lists) + plus_Dzone - minus_Dzone
        else:
            predicted_population = np.array(population_lists) + plus_Rzone - minus_Rzone
            
        # Enforce non-negativity.
        predicted_population = np.maximum(predicted_population, 0).tolist()
              
        return predicted_population
    
    def predict_equilibrium_population(self, df, population_lists, development_lists, investment_lists, initial_price_list_D, year, move_ratio, zoning_type="dev_zone"):
        """
        Predict population transition using the estimated model with demand and supply equilibrium, using individual assignment.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        if year <= self.end_year:
            #df_year = df[(df["転居年"]==year) & (df["転居有無"]==1)].copy()
            df_year = df[(df["年"]==year)].sample(frac=move_ratio).copy()
        else:
            # Sample non-movers when forecasting beyond the observed period.
            df_year = df[df["転居年"]==0].sample(frac=move_ratio).copy()
        X1_hetero, X1_mean, x2, Z = self.generate_prediction_data_2step(df_year, investment_lists, initial_price_list_R, year)
        # Iteratively adjust prices toward market clearing.
        price_lists = np.array(initial_price_list_D.copy())
        price_lists_new = price_lists.copy()
        iteration = 0
        price_unit = self.config.adjust_price_unit  # Price adjustment step size.
        margin = self.config.adjust_margin  # Ignore smaller mismatches for stability.
        upper_limit_D = (1+self.config.change_limit) * np.array(initial_price_list_D.copy())
        lower_limit_D = (1-self.config.change_limit) * np.array(initial_price_list_D.copy())
        max_iterations = self.config.max_iteration  # Maximum number of adjustment rounds.
                
        while True:
            # Predict migration under the current prices.
            predicted_probs = self.model.predict(X1_hetero, X1_mean, x2, Z, self.XZ_mask, self.spatial_weight_matrix) # (N, J)
            predicted_move = predicted_probs * df_year["拡大係数"].values.reshape(-1, 1)  # (N, J)
            
            # Aggregate inflows and outflows.
            plus_Rzone = predicted_move.sum(axis=0)
            minus_Rzone = np.array([df_year.loc[df_year["居住地_前_ゾーン"]==zone, "拡大係数"].sum() for zone in self.zone_list])            
            plus_Dzone = np.sum(plus_Rzone[np.newaxis,:] * self.zone_conversion, axis=1)
            minus_Dzone = np.sum(minus_Rzone[np.newaxis,:] * self.zone_conversion, axis=1)
            
            if zoning_type == "dev_zone":
                predicted_population = np.array(population_lists) + plus_Dzone - minus_Dzone
                land_demand = plus_Dzone * self.land_demand_zone_D  # Convert movers into land demand.
            else:
                predicted_population = np.array(population_lists) + plus_Rzone - minus_Rzone
                land_demand = plus_Rzone * self.land_demand_zone_R 
            # Enforce non-negativity.
            predicted_population = np.maximum(predicted_population, 0)
            # Supply-side capacity constraints could also be added here.
            
            # Raise prices under excess demand and lower them under excess supply.
            excess_demand = land_demand - np.array(development_lists[-1]) > margin
            deficit_demand = land_demand - np.array(development_lists[-1]) < -margin
            
            #print(iteration)
            #print(land_demand - np.array(development_lists[-1]))
            #print(price_lists)
            # Update prices at the development-zone level.
            price_lists_new = np.where(excess_demand, price_lists + price_unit, price_lists)
            price_lists_new = np.where(deficit_demand, price_lists_new - price_unit, price_lists_new)
            diff = price_lists_new - price_lists
            # Stop when prices no longer change meaningfully.
            if np.max(abs(diff)) < margin or iteration >= max_iterations:
                break
            if np.all(diff > 0):
                #raise ValueError("価格がすべて上昇しているため、供給が需要を満たしていない可能性があります。")
                break
                
            # Push the corresponding residential-zone utilities in the same direction.
            Rzone_idx_excess_list = [np.where(self.zone_conversion[Dzone,:]>0)[0].tolist() for Dzone in self.Dzone_list if excess_demand[Dzone]]
            Rzone_idx_deficit_list = [np.where(self.zone_conversion[Dzone,:]>0)[0].tolist() for Dzone in self.Dzone_list if deficit_demand[Dzone]]
            threshold = 0  # Optional filter for targeting only the busiest receiving zones.
            var_index = self.var_L_2.index("price")
            if len(Rzone_idx_excess_list) > 0:
                Rzone_idx_excess = np.unique([item for sublist in Rzone_idx_excess_list for item in sublist if plus_Rzone[item] > threshold])
                x2[Rzone_idx_excess, var_index] += price_unit / self.scale_price
            if len(Rzone_idx_deficit_list) > 0:
                Rzone_idx_deficit = np.unique([item for sublist in Rzone_idx_deficit_list for item in sublist if plus_Rzone[item] > threshold])
                x2[Rzone_idx_deficit, var_index] -= price_unit / self.scale_price
            
            iteration += 1
            x2[:, var_index] = np.clip(x2[:, var_index], 0, None)  # Prices must stay non-negative.
            price_lists = np.clip(price_lists_new.copy(), lower_limit_D, upper_limit_D)
        
        return price_lists.tolist(), predicted_population.tolist(), land_demand.tolist()
    
    
    def predict_aggregate_equilibrium_population(
        self, 
        population_lists, development_lists, investment_lists, initial_price_list_D, 
        year, move_ratio, zoning_type="dev_zone"):
        """
        Predict population transition using the estimated model with demand and supply equilibrium, using aggregate zonal amenity change through delta.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        if year < 2015:
            delta = self.model.delta[0].copy()
        elif 2015 <= year <= self.end_year:
            year_index = year - 2015
            delta = self.model.delta[year_index].copy()
        else:
            year_index = self.end_year - 2015
            delta = self.model.delta[year_index].copy()
        
        # Extra utility term for in-movers from outside the city.
        if "move_in" in self.var_Z:
            move_in_idx = self.var_Z.index("move_in")
            assert self.XZ_mask[move_in_idx, self.var_L_1_hetero.index("dist_CBD")] == 1, "move_in variable must interact with dist_CBD"
            beta_distance_CBD = self.model.beta_z[len(self.var_L_1_mean) + np.sum(self.XZ_mask[:move_in_idx, :]) + self.XZ_mask[move_in_idx, :self.var_L_1_hetero.index("dist_CBD")].sum()]
            #print("Beta distance_CBD for move_in:", beta_distance_CBD)
            self.distance_CBD_utility = beta_distance_CBD * self.center_distances
            
        # Matrix used in the SAR-style spatial interaction term.
        self.SAR_cov_mat = np.linalg.inv(np.eye(self.J) - self.model.beta_z[-1] * self.spatial_weight_matrix)
        self.SAR_cov_mat_mean = np.dot(self.SAR_cov_mat, np.random.standard_normal((self.J, 100))).mean(axis=1)
        
        # Utility effect of distance from the previous residence.
        if "dist_prev" in self.var_L_1_mean:
            var_index = self.var_L_1_mean.index("dist_prev")
            #print("Beta dist_prev:", self.model.beta_z[var_index])
            self.dist_prev_utility = self.model.beta_z[var_index] * self._distance_matrix
        
        # LOS effect.
        investment_Rzone = np.maximum(
            np.dot(np.asarray(investment_lists, dtype=float), self.investment_distribution), 1.0
        )  # shape: (Dzone,) @ (Dzone, Rzone) -> (Rzone,)
        los_idx = 1 + self.var_L_2.index("los_all")  # +1 accounts for the constant term.
        los_year_idx = year - self.start_year if year <= self.end_year else self.end_year - self.start_year
        delta += self.model.beta_x[los_idx] * self.los_convert(investment_Rzone - self.investment_data_R[:, los_year_idx])  # Add the LOS change.
        
        # Price-change effect.
        price_year_idx = year - self.start_year if year <= self.end_year else self.end_year - self.start_year
        price_idx = 1 + self.var_L_2.index("price")
        # Convert development-zone prices into residential-zone prices.
        initial_price_list_R = self.zone_conversion.T @ np.array(initial_price_list_D)  # Weighted average from D-zones to R-zones.
        delta += self.model.beta_x[price_idx] * (initial_price_list_R - self.price_data_R[:,price_year_idx]) / self.scale_price
        
        population_Rzone = np.maximum(
            np.dot(np.asarray(population_lists, dtype=float), self.population_distribution),
            1.0
            )  # (Dzone,) @ (Dzone, Rzone) -> (Rzone,)
        minus_Rzone, exiting_Rzone = self.calculate_moving_out_population(year, population_Rzone)
        minus_Dzone = self.zone_conversion @ minus_Rzone # (Dzone, Rzone) @ (Rzone,) -> (Dzone,)
        exiting_Dzone = self.zone_conversion @ exiting_Rzone
        
        # Iteratively adjust prices toward market clearing.
        price_D=np.array(initial_price_list_D)
        price_beta_idx = 1 + self.var_L_2.index("price")
        price_beta = self.model.beta_x[price_beta_idx]
        margin=self.config.adjust_margin  # Ignore smaller mismatches for stability.
        price_unit=self.config.adjust_price_unit  # Price adjustment step size.
        lower_limit=(1-self.config.change_limit) * price_D  # Lower bound on price updates.
        upper_limit=(1+self.config.change_limit) * price_D  # Upper bound on price updates.
        
        # 1. Sequentially update prices.
        for iteration in range(self.config.max_iteration):
            # Aggregate migration under the current utilities.
            predicted_population, land_demand = self.aggregate_migration(
                year=year,
                delta=delta,
                population_lists=population_lists,
                minus_Rzone=minus_Rzone,
                minus_Dzone=minus_Dzone,
                exiting_Rzone=exiting_Rzone,
                exiting_Dzone=exiting_Dzone,
                zoning_type=zoning_type
            )
            
            # Raise prices under excess demand and lower them under excess supply.
            diff_demand = land_demand - np.asarray(development_lists[-1], dtype=float)  # (Dzone,)
            excess_mask = diff_demand > margin    # Excess demand.
            deficit_mask = diff_demand < -margin  # Excess supply.
            
            # Update prices.
            new_price_D = self.update_price(initial_price_D=price_D, price_unit=price_unit, excess_mask=excess_mask, deficit_mask=deficit_mask)
            if self.check_convergence(diff_demand, threshold=margin):
                price_D = new_price_D.copy()
                break
            
            # Update residential-zone utilities implied by the new prices.
            delta = self.update_delta(delta, price_beta, price_unit, excess_mask, deficit_mask)
            
            # Clip prices and carry them into the next iteration.
            price_D = np.clip(new_price_D, lower_limit, upper_limit)
            
        """
        # 2. Alternative optimization-based price adjustment.
        init_a = np.zeros_like(price_D)
        bounds = [(-price_D[i]*self.config.change_limit/self.scale_price, price_D[i]*self.config.change_limit/self.scale_price) for i in range(len(price_D))]
        res = minimize(self.objective_function, x0=init_a, args=(year, delta, price_beta, population_lists, development_lists, minus_Rzone, minus_Dzone, exiting_Rzone, exiting_Dzone, zoning_type), method="L-BFGS-B", bounds=bounds)
        price_D += res.x * self.scale_price
        predicted_population, land_demand = self.demand_from_price(
            price_change=res.x,
            year=year,
            delta=delta,
            price_beta=price_beta,
            population_lists=population_lists,
            minus_Rzone=minus_Rzone,
            minus_Dzone=minus_Dzone,
            exiting_Rzone=exiting_Rzone,
            exiting_Dzone=exiting_Dzone,
            zoning_type=zoning_type
        )
        """
        
        #print(np.sum(np.abs(land_demand - np.asarray(development_lists[-1], dtype=float))))
        # Total population should remain approximately conserved.
        #assert abs(np.sum(predicted_population) - np.sum(population_lists)) < 1e-3, f"Total population mismatch. Input: {np.sum(population_lists)}, Predicted: {np.sum(predicted_population)}"
        
        return price_D.tolist(), predicted_population.tolist(), land_demand.tolist()
    
    def calculate_moving_out_population(self, year, population_Rzone):
        if year < 2015:
            move_prob = self.move_ratio[:,0]
            exiting_prob = self.exiting_ratio[:,0]
        elif 2015 <= year <= 2024:
            year_index = year - 2015
            move_prob = self.move_ratio[:,year_index]
            exiting_prob = self.exiting_ratio[:,year_index]
        else:
            year_index = 2024 - 2015
            move_prob = self.move_ratio[:,year_index]
            exiting_prob = self.exiting_ratio[:,year_index]
        minus_Rzone = population_Rzone * move_prob
        exiting_Rzone = population_Rzone * exiting_prob
        return minus_Rzone, exiting_Rzone
        
    def calculate_inflow_population(self, year, delta=None):
        # Baseline exogenous inflow totals by zone.
        if year < 2011:
            inflow_pop = self.inflow_pop[:,0]
        elif 2011 <= year <= 2024:
            year_index = year - 2011
            inflow_pop = self.inflow_pop[:,year_index]
        else:
            year_index = 2024 - 2011
            inflow_pop = self.inflow_pop[:,year_index]
        # Reallocate total inflow endogenously using current utilities.
        exp_util = np.exp(delta + self.distance_CBD_utility + self.SAR_cov_mat_mean)  # (Rzone,)
        predicted_probs = exp_util / np.sum(exp_util) # (Rzone,)
        inflow_pop = predicted_probs * inflow_pop.sum()
        
        return inflow_pop
    
    def objective_function(self, price_change, year, delta, price_beta, population_lists, development_lists, minus_Rzone, minus_Dzone, exiting_Rzone, exiting_Dzone, zoning_type):
        _, land_demand = self.demand_from_price(price_change, year, delta, price_beta, population_lists, minus_Rzone, minus_Dzone, exiting_Rzone, exiting_Dzone, zoning_type)
        return np.sum(np.abs(land_demand - np.asarray(development_lists[-1], dtype=float)))
    
    def demand_from_price(self, price_change, year, delta, price_beta, population_lists, minus_Rzone, minus_Dzone, exiting_Rzone, exiting_Dzone, zoning_type):
        #############
        mat = (self.zone_conversion.T > 0).astype(float)
        mat /= mat.sum(axis=1, keepdims=True)
        ############
        price_change = mat @ price_change  # (Rzone, Dzone) @ (Dzone,) -> (Rzone,)
        delta += price_beta * price_change
        exp_delta = np.exp(delta[np.newaxis,:] + self.SAR_cov_mat_mean[np.newaxis,:] + self.dist_prev_utility) # (Rzone,Rzone)
        predicted_probs = exp_delta / np.sum(exp_delta, axis=1, keepdims=True)  # (Rzone,Rzone)
        plus_Rzone =(minus_Rzone[np.newaxis,:] @ predicted_probs).reshape(-1) + self.calculate_inflow_population(year, delta=delta) # (Rzone,)
        plus_Dzone = self.zone_conversion @ plus_Rzone # (Dzone,Rzone) @ (Rzone,) -> (Dzone,)
        
        if zoning_type == "dev_zone":
            predicted_population = np.array(population_lists) + plus_Dzone - minus_Dzone - exiting_Dzone
            land_demand = plus_Dzone * self.land_demand_zone_D  # Convert movers into land demand.
        else:
            predicted_population = np.array(population_lists) + plus_Rzone - minus_Rzone - exiting_Rzone
            land_demand = plus_Rzone * self.land_demand_zone_R
        # Enforce non-negativity.
        predicted_population = np.maximum(predicted_population, 0.0)
        return predicted_population, land_demand
    
    def aggregate_migration(self, year, delta, population_lists, minus_Rzone, minus_Dzone, exiting_Rzone, exiting_Dzone, zoning_type):
        exp_delta = np.exp(delta[np.newaxis,:] + self.SAR_cov_mat_mean[np.newaxis,:] + self.dist_prev_utility) # (Rzone,Rzone)
        predicted_probs = exp_delta / np.sum(exp_delta, axis=1, keepdims=True)  # (Rzone,Rzone)
        plus_Rzone =(minus_Rzone[np.newaxis,:] @ predicted_probs).reshape(-1) + self.calculate_inflow_population(year, delta=delta) # (Rzone,)
        plus_Dzone = self.zone_conversion @ plus_Rzone # (Dzone,Rzone) @ (Rzone,) -> (Dzone,)
        
        if zoning_type == "dev_zone":
            predicted_population = np.array(population_lists) + plus_Dzone - minus_Dzone - exiting_Dzone
            land_demand = plus_Dzone * self.land_demand_zone_D  # Convert movers into land demand.
        else:
            predicted_population = np.array(population_lists) + plus_Rzone - minus_Rzone - exiting_Rzone
            land_demand = plus_Rzone * self.land_demand_zone_R
        # Enforce non-negativity.
        predicted_population = np.maximum(predicted_population, 0.0)
        return predicted_population, land_demand
    
    def update_price(self, initial_price_D, price_unit, excess_mask, deficit_mask):       
        # Price update at the development-zone level.
        new_price_D = initial_price_D.copy()
        if np.any(excess_mask):
            new_price_D = np.where(excess_mask, new_price_D + price_unit, new_price_D)
        if np.any(deficit_mask):
            new_price_D = np.where(deficit_mask, new_price_D - price_unit, new_price_D)
        return new_price_D
    
    def check_convergence(self, diff_demand, threshold):
        if np.all(np.abs(diff_demand) <= threshold):
            return True
        return False
    
    def update_delta(self, delta, price_beta, price_unit, excess_mask, deficit_mask):
        # Update the implied residential-zone utilities.
        if np.any(excess_mask):
            r_flag = np.unique(np.concatenate([self.D_to_R_idx[d] for d in np.nonzero(excess_mask)[0]]))  # Residential zones linked to excess-demand development zones.
            delta[r_flag] += (price_unit / self.scale_price) * price_beta
        if np.any(deficit_mask):
            r_flag = np.unique(np.concatenate([self.D_to_R_idx[d] for d in np.nonzero(deficit_mask)[0]]))  # (Rzone,)
            delta[r_flag] -= (price_unit / self.scale_price) * price_beta
        return delta
    
    
    def load_estimates(self, step1_path, step2_path, model):
        """
        Load the estimated model parameters from files.
        Parameters:
            step1_path (str): Path to the saved step 1 model
            step2_path (str): Path to the saved step 2 model
            model (str): Model type, either
                - "MNL_2step" for two-step multinomial logit
                - "MXL_2step" for two-step mixed logit
        """
        if model == "MNL_2step":
            self.model = MNL_2step(time_steps=self.end_year - self.start_year + 1)
        elif model == "MXL_2step":
            self.model = MXL(time_steps=self.end_year - self.start_year + 1)
        self.model.load_step1(step1_path)
        self.model.load_step2(step2_path)
        print("Model parameters loaded successfully.")
    
    def expected_maximum_utility(self, delta):
        return np.log(np.sum(np.exp(delta)))  # log-sum-exp
    
    def calculate_welfare(self, price_D, investment_D, year):
        """
        Calculate consumer welfare based on the estimated model.
        Parameters:
            price (ndarray): Price array for each zone
            los (ndarray): Level of service array for each zone
            year (int): Year for which to calculate welfare
        Returns:
            welfare (float): Calculated consumer welfare
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Calculate delta for each zone
        if year < 2015:
            delta = self.model.delta[0].copy()
        elif 2015 <= year <= self.end_year:
            year_index = year - 2015
            delta = self.model.delta[year_index].copy()
        else:
            year_index = self.end_year - 2015
            delta = self.model.delta[year_index].copy()
        
        # LOS effect.
        investment_R = np.maximum(
            np.dot(np.asarray(investment_D, dtype=float), self.investment_distribution), 1.0
        )  # shape: (Dzone,) @ (Dzone, Rzone) -> (Rzone,)
        los_idx = 1 + self.var_L_2.index("los_all")  # +1 accounts for the constant term.
        los_year_idx = year - self.start_year if year <= self.end_year else self.end_year - self.start_year
        delta += self.model.beta_x[los_idx] * self.los_convert(investment_R - self.investment_data_R[:, los_year_idx])  # Add the LOS change.
        
        # Price-change effect.
        price_year_idx = year - self.start_year if year <= self.end_year else self.end_year - self.start_year
        price_idx = 1 + self.var_L_2.index("price")
        # Convert development-zone prices into residential-zone prices.
        price_R = self.zone_conversion.T @ np.array(price_D)  # Weighted average from D-zones to R-zones.
        delta += self.model.beta_x[price_idx] * (price_R - self.price_data_R[:,price_year_idx]) / self.scale_price
        
        # Utility effect of distance from the previous residence.
        var_index = self.var_L_1_mean.index("dist_prev")
        dist_prev_utility = self.model.beta_z[var_index] * self._distance_matrix
        
        # Extra utility component for car-owning households.
        if "car" in self.var_Z:
            car_idx = self.var_Z.index("car")
            beta_los_all = self.model.beta_z[len(self.var_L_1_mean) + np.sum(self.XZ_mask[:car_idx, :]) + self.XZ_mask[car_idx, :self.var_L_1_hetero.index("los_all")].sum()]
            los_all_utility = beta_los_all * self.investment_data_R[:, los_year_idx]
        
        # Expected maximum utility for car owners.
        delta_car = delta + los_all_utility + dist_prev_utility
        expected_utility_car = self.expected_maximum_utility(delta_car)
        # Expected maximum utility for non-car owners.
        delta_no_car = delta + dist_prev_utility
        expected_utility_no_car = self.expected_maximum_utility(delta_no_car)
        
        # Convert utilities into money-metric welfare.
        welfare_car = expected_utility_car * (self.scale_price / abs(self.model.beta_x[price_idx]))
        welfare_no_car = expected_utility_no_car * (self.scale_price / abs(self.model.beta_x[price_idx]))
        
        return welfare_car, welfare_no_car
        

if __name__ == "__main__":
    # Simple local entry point for estimation-data generation.
    start_year = 2015
    end_year = 2023
    ref_year = 1
    transition_model = TransitionModel(start_year=start_year, end_year=end_year, ref_year=ref_year, dev_zone="19_zone", res_zone="pop_zone")
    
    """
    # for mnl
    X, y, asc_cols = transition_model.make_estimation_data_mnl()
    # Estimate the individual choice model
    beta, _ = transition_model.estimate_choice_model(y, method="MNL", X=X, asc_cols=asc_cols)
    """
    
    # for 2step
    X1_hetero, X1_mean, X2, Z, W, y, obs_share, relocation_years = transition_model.make_estimation_data_2step()    
    # Estimate the individual choice model
    #_, estimated_model = transition_model.estimate_choice_model(y, method="MNL_2step", X1_hetero=X1_hetero, X1_mean=X1_mean, X2=X2, Z=Z, W=W, obs_share=obs_share, relocation_years=relocation_years)
    #_, estimated_model = transition_model.estimate_step2_model(step1_path="output/estimates/mnl_step1_results.json", X2=X2, W=W, method="IV", model="MNL_2step")
    _, estimated_model = transition_model.estimate_choice_model(y, method="MXL_2step", X1_hetero=X1_hetero, X1_mean=X1_mean, X2=X2, Z=Z, W=W, obs_share=obs_share, relocation_years=relocation_years)

    # predict
    df = read_pt_data(transition_model.start_year, transition_model.end_year, transition_model.res_zone)
    development_lists = [[transition_model.develop_dict[start_year-i-1]["面積"].get((Dzone, 1), 0) for Dzone in transition_model.Dzone_list] for i in range(ref_year)]
    investment_lists = [transition_model.los_dict[start_year][zone]["total"] for zone in transition_model.Dzone_list]
    #price_list_R = transition_model.zoning[f"UnitPrice_Attached_{start_year}"].values / transition_model.scale_price if not np.isnan(transition_model.zoning[f"UnitPrice_Attached_{start_year}"].values[0]) else transition_model.zoning[f"UnitPrice_Condo_{start_year}"].values[0] / transition_model.scale_price
    price_list_R = np.where(np.isnan(transition_model.zoning[f"UnitPrice_Attached_{start_year}"].values), transition_model.zoning[f"UnitPrice_Condo_{start_year}"].values, transition_model.zoning[f"UnitPrice_Attached_{start_year}"].values) / transition_model.scale_price
    price_list_D = np.where(np.isnan(transition_model.Dzoning[f"UnitPrice_Attached_{start_year}"].values), transition_model.Dzoning[f"UnitPrice_Condo_{start_year}"].values, transition_model.Dzoning[f"UnitPrice_Attached_{start_year}"].values) / transition_model.scale_price
    
    df_start = df[df["年"]==start_year].copy()
    population_lists = [df_start.loc[df_start["居住地_前_ゾーン"]==zone, "拡大係数"].sum() for zone in transition_model.zone_list]
    population_lists = np.sum(np.array(population_lists)[np.newaxis,:] * transition_model.zone_conversion, axis=1).tolist()
    price_list_D, predicted_population, land_demand = transition_model.predict_aggregate_equilibrium_population(
        population_lists=population_lists, development_lists=development_lists, investment_lists=investment_lists, initial_price_list_D=price_list_D, 
        year=start_year, move_ratio=transition_model.config.move_ratio, zoning_type="dev_zone")
    print(np.round(np.array(predicted_population)))
    
