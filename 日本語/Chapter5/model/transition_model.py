"""居住地選択モデルと人口遷移をまとめて扱う中核モジュール。"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import Rook
import sys
from scipy.optimize import minimize
# 一つ上の階層で実行
sys.path.append("..")

from model.mnl import MNL, MNL_2step
from model.mxl import MXL
from model.data_reading import read_zone_code, read_pt_data, read_pop_data, read_building_data, read_los_data, distance_matrix, read_obs_share_data, read_move_data
from utils.config_manager import ConfigManager

class TransitionModel:
    """Chapter 5 の遷移モデル全体を管理するクラス。"""
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
        """スケール設定と基礎データを読み込み、モデルの初期状態を作る。"""
        # スケーリング・設定値
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
        print(f"ゾーン数: {len(self.zone_list)}")
        self.area_dict = {zone: self.zoning.loc[self.zoning["選択ゾーン"]==zone, "area_km2"].values[0] for zone in self.zone_list}
        self.pop_dict = read_pop_data(self.start_year, self.end_year, self.res_zone)
        self.develop_dict = read_building_data(self.start_year-self.ref_year-1, self.end_year, self.res_zone)
        self.los_dict = read_los_data(self.start_year, self.end_year, self.res_zone)
        self.los_dict_D = read_los_data(self.start_year, self.end_year, self.dev_zone)
        self.city_center_zone = 0 # 一番町
        self.dist_CBD_threshold = 3.0 / 10 # 5km (scale)
        self.center_distances = np.array([self.dist_convert(zone, self.city_center_zone) for zone in self.zone_list])
        self.static_features = self.zoning[self.STATIC_FEATURE_COLUMNS].values
        self.move_ratio, self.exiting_ratio, self.inflow_pop = read_move_data(self.zoning)
        
        np.random.seed(self.config.seed)
    
    def set_mxl(self):
        """MXL 推定で使う説明変数セットと次元情報を定義する。"""
        self.var_L_1_hetero = ["dist_CBD", "los_all", "flood_risk", "school", "commercial_area"] # "dist_CBD", "dist_CBD<threshold", "dist_CBD>threshold", "los_all", "los_bus", "los_train", "flood_risk", "school", "child_care", "hospitals", "park_area", "commercial_area"
        self.var_L_1_mean = ["dist_prev"] # "dist_prev"
        self.var_L_2 = ["dist_CBD", "los_all", "flood_risk", "school", "commercial_area", "price"] # "dist_CBD", "dist_CBD<threshold", "dist_CBD>threshold", "los_all", "los_bus", "los_train", "flood_risk", "school", "child_care", "hospitals", "park_area", "commercial_area", "price"
        self.var_Z = ["age", "car", "family", "move_in"] # "age_head", "car_ownership", "has_children", "income", "move_in"
        self.var_W = ["lowuse_area"] # "risk_area", "lowuse_area"
        # MXL
        self.XZ_mask = np.array([
            [0, 0, 0, 0, 1],  # 世帯主年齢
            [0, 1, 0, 0, 0],  # 自動車有無
            [0, 0, 0, 1, 1],   # 子供有無
            [1, 0, 0, 0, 0] # 転入
        ], dtype=int)
                
        self.J = len(self.zone_list) # Number of alternatives
        self.L_1_hetero = len(self.var_L_1_hetero)  # Number of zonal features in X1_hetero
        self.L_1_mean = len(self.var_L_1_mean)  # Number of zonal features in X1_mean
        self.L_2 = len(self.var_L_2)  # Number of zonal features in X2
        self.K = len(self.var_Z)  # Number of personal attributes in Z
        self.T = self.end_year - self.start_year + 1  # Number of time steps
        assert self.XZ_mask.shape == (self.K, self.L_1_hetero), "XZ_mask shape mismatch"
    
    def set_mnl(self):
        """MNL 推定用の説明変数セットを定義する。"""
        # MNL用
        self.var_mnl = ["dist_prev", "dist_CBD", "los_all", "flood_risk", "school", "commercial_area", "price"]
        self.asc_cols = None
        self.J = len(self.zone_list) # Number of alternatives
        
    def set_zone_correspondence(self):
        """開発ゾーンと居住ゾーンの対応関係を前計算する。"""
        if self.dev_zone != self.res_zone:
            self.Dzone_code_in, self.Dzone_code_all, self.Dzoning = read_zone_code(self.dev_zone)
            self.distance_df_D = distance_matrix(self.Dzoning, self.dev_zone)
            self.Dzone_list = np.unique(self.Dzoning["選択ゾーン"].values).tolist()
            Rzone_size = self.zoning["area_km2"].values
            base_path = Path(__file__).resolve().parent.parent  # model/ から1階層上へ
            gdf = gpd.read_file(base_path / "data/raw/zoning/census_zone.geojson")     
            self.zone_conversion = np.array([[gdf.loc[(gdf["S_NAME"].isin(self.Dzone_code_in.loc[self.Dzone_code_in["選択ゾーン"]==Dzone, "町丁字名"]))&(gdf["S_NAME"].isin(self.zone_code_in.loc[self.zone_code_in["選択ゾーン"]==Rzone, "町丁字名"])), "area_km2"].sum() for Rzone in self.zone_list] for Dzone in self.Dzone_list]) / Rzone_size[np.newaxis,:] # (Dzone, Rzone). 各Dzoneについて、各Rzoneの総面積に対するそのDzoneに含まれるRzoneの面積割合.列和は1.
            self.develop_distribution = [[self.develop_dict[self.start_year]["面積"].get((zone, 1), 0) * self.zone_conversion[Dzone,zone] for zone in self.zone_list] for Dzone in self.Dzone_list]
            self.develop_distribution = np.array([np.array(self.develop_distribution[i]) / sum(self.develop_distribution[i]) for i in self.Dzone_list])
            self.investment_distribution = [[self.los_dict[self.start_year][zone]["total"] * self.zone_conversion[Dzone,zone] for zone in self.zone_list] for Dzone in self.Dzone_list] # 各小ゾーンのinvestmentをその小ゾーンが属する大ゾーンに割り当てる
            self.investment_distribution = np.array([np.array(self.investment_distribution[i]) / sum(self.investment_distribution[i]) for i in self.Dzone_list]) # 各Dzoneの中で各Rzoneのinvestment分布． (Dzone, Rzone)
            self.population_distribution = [[self.pop_dict[self.start_year]["人口"][zone] * self.zone_conversion[Dzone,zone] for zone in self.zone_list] for Dzone in self.Dzone_list]
            self.population_distribution = np.array([np.array(self.population_distribution[i]) / sum(self.population_distribution[i]) for i in self.Dzone_list]) # Dzoneの中で各Rzoneの人口分布割合
        else:
            self.Dzone_code_in = self.zone_code_in.copy()
            self.Dzone_code_all = self.zone_code_all.copy()
            self.Dzoning = self.zoning.copy()
            self.Dzone_list = self.zone_list.copy()
            self.zone_conversion = np.eye(self.J)
            self.develop_distribution = np.eye(self.J)
            self.investment_distribution = np.eye(self.J)
            self.population_distribution = np.eye(self.J)
            
        # 転入人口から土地需要への換算
        self.land_demand_zone_R = np.ones(self.J) * self.config.new_house_ratio * self.config.per_person_land_demand # 転入者のうち新築持ち家の割合 * 1人あたり平均敷地面積
        self.land_demand_zone_D = np.ones(len(self.Dzone_list)) *self.config.new_house_ratio * self.config.per_person_land_demand # 転入者のうち新築持ち家の割合 * 1人あたり平均敷地面積
        # https://www.pref.ehime.jp/uploaded/attachment/141646.pdf
        
        self.D_to_R_idx = [np.flatnonzero(self.zone_conversion[d, :] > 0)
                        for d in range(self.zone_conversion.shape[0])] # 各Dzoneに対応するRzoneのインデックスリスト
    
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
        """ゾーン間距離行列をインスタンス内に前計算する。"""
        # 距離行列を事前計算（zone間の距離）- インスタンス変数として保存
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
        """居住ゾーン間距離をモデル内のスケールに変換して返す。"""
        d = self.distance_df.loc[(self.distance_df["zone_1"]==zone_before) & (self.distance_df["zone_2"]==zone_after), "distance_km"].values[0]
        #d = np.log(1 + d) if d > 0 else 0
        d /= 10 # scale
        return d
    
    def dist_convert_D(self, zone_before, zone_after):
        """開発ゾーン間距離をモデル内のスケールに変換して返す。"""
        d = self.distance_df_D.loc[(self.distance_df_D["zone_1"]==zone_before) & (self.distance_df_D["zone_2"]==zone_after), "distance_km"].values[0]
        #d = np.log(1 + d) if d > 0 else 0
        d /= 10
        return d
    
    def los_convert(self, los):
        """LOS をモデル内のスケールへ変換する。"""
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
            
        # 時間に依存しないデータ（すべてのサンプルで同じ）
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
            
        # Xをnpyとして保存
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
            "居住地_前_ゾーン": [zone for zone in self.zone_list] # aggregated
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
        
        # 時間に依存しないデータ（すべてのサンプルで同じ）
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
        
        base_path = Path(__file__).resolve().parent.parent  # model/ から1階層上へ
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
        
        # 効率的なデータ構造の事前構築
        years = list(range(self.start_year, self.end_year + 1))
        n_years = len(years)
        
        # 住宅開発データを配列として事前計算
        los_bus = np.zeros((self.J, n_years))
        los_train = np.zeros((self.J, n_years))
        los_all = np.zeros((self.J, n_years))
        
        for zone_idx, zone in enumerate(self.zone_list):
            for year_idx, year in enumerate(years):
                # LOS データ
                los_value = self.los_dict[year][zone]["bus"]
                los_bus[zone_idx, year_idx] = self.los_convert(los_value)
                los_value = self.los_dict[year][zone]["shinai"] + self.los_dict[year][zone]["kougai"]
                los_train[zone_idx, year_idx] = self.los_convert(los_value)
                los_value = self.los_dict[year][zone]["total"]
                los_all[zone_idx, year_idx] = self.los_convert(los_value)
        
        # 現在の居住地からの距離をベクトル化計算
        year_indices = df_pool["年"].values.astype(int) - self.start_year
        center_distances = self.center_distances
        
        # X1_meanをベクトル化計算
        if "dist_prev" in self.var_L_1_mean:
            var_index = self.var_L_1_mean.index("dist_prev")
            prev_zones = df_pool["居住地_前_ゾーン"].values.astype(int)
            X1_mean[prev_zones == -1, :, var_index] = 0.0
            for alt in range(self.J):
                X1_mean[prev_zones != -1, alt, var_index] = self._distance_matrix[prev_zones[prev_zones != -1], alt]
        
        # X1_heteroをベクトル化計算
        if "dist_CBD" in self.var_L_1_hetero: # 中心部からの距離（すべてのゾーンで同じ）
            var_index = self.var_L_1_hetero.index("dist_CBD")
            X1_hetero[:, :, var_index] = np.tile(center_distances, (N, 1))
        if "dist_CBD<threshold" in self.var_L_1_hetero: # 中心部からの距離（閾値以下）
            var_index = self.var_L_1_hetero.index("dist_CBD<threshold")
            X1_hetero[:, :, var_index] = np.tile(np.maximum(0, np.minimum(center_distances, self.dist_CBD_threshold)), (N, 1))
        if "dist_CBD>threshold" in self.var_L_1_hetero: # 中心部からの距離（閾値以上）
            var_index = self.var_L_1_hetero.index("dist_CBD>threshold")
            X1_hetero[:, :, var_index] = np.tile(np.maximum(0, center_distances - self.dist_CBD_threshold), (N, 1))
        if "los_bus" in self.var_L_1_hetero: # バスLOS データ
            var_index = self.var_L_1_hetero.index("los_bus")
            X1_hetero[:, :, var_index] = los_bus[:, year_indices].T
        if "los_train" in self.var_L_1_hetero: # 鉄道LOS データ
            var_index = self.var_L_1_hetero.index("los_train")
            X1_hetero[:, :, var_index] = los_train[:, year_indices].T
        if "los_all" in self.var_L_1_hetero: # 全LOS データ
            var_index = self.var_L_1_hetero.index("los_all")
            X1_hetero[:, :, var_index] = los_all[:, year_indices].T

        # 時間に依存しないデータ（すべてのサンプルで同じ）
        static_features = self.static_features
        if "flood_risk" in self.var_L_1_hetero: # 浸水リスク
            var_index = self.var_L_1_hetero.index("flood_risk")
            X1_hetero[:, :, var_index] = np.tile(static_features[:, 0] + static_features[:, 1], (N, 1)) #  > 0).astype(float) > 0).astype(float)
        if "school" in self.var_L_1_hetero: # 最寄りの学校までの距離
            var_index = self.var_L_1_hetero.index("school")
            X1_hetero[:, :, var_index] = np.tile(static_features[:, 2], (N, 1))
        if "child_care" in self.var_L_1_hetero: # 子育て施設の密度
            var_index = self.var_L_1_hetero.index("child_care")
            X1_hetero[:, :, var_index] = np.tile(np.log(1 + static_features[:, 3]), (N, 1))
        if "hospitals" in self.var_L_1_hetero: # 病院の密度
            var_index = self.var_L_1_hetero.index("hospitals")
            X1_hetero[:, :, var_index] = np.tile(np.log(1 + static_features[:, 4]), (N, 1))
        if "park_area" in self.var_L_1_hetero: # 公園の面積
            var_index = self.var_L_1_hetero.index("park_area")
            X1_hetero[:, :, var_index] = np.tile(np.log(1 + 100 * static_features[:, 5]), (N, 1))
        if "commercial_area" in self.var_L_1_hetero: # 商業地の面積
            var_index = self.var_L_1_hetero.index("commercial_area")
            X1_hetero[:, :, var_index] = np.tile(10 * static_features[:, 6], (N, 1))
        
        y = df_pool["居住地_後_ゾーン"].values.astype(int)
        
        # Zonal features for the second step
        X2 = np.zeros((self.J * self.T, self.L_2))
        
        # すべてのゾーンと年の組み合わせを一度に処理
        zone_indices = np.repeat(np.arange(self.J), n_years)
        year_indices = np.tile(np.arange(n_years), self.J)
        
        if "dist_CBD" in self.var_L_2: # 中心部からの距離
            var_index = self.var_L_2.index("dist_CBD")
            X2[:, var_index] = np.repeat(center_distances, n_years)
        if "dist_CBD<threshold" in self.var_L_2: # 中心部からの距離（閾値以下）
            var_index = self.var_L_2.index("dist_CBD<threshold")
            X2[:, var_index] = np.repeat(np.maximum(0, np.minimum(center_distances, self.dist_CBD_threshold)), n_years)
        if "dist_CBD>threshold" in self.var_L_2: # 中心部からの距離（閾値以上）
            var_index = self.var_L_2.index("dist_CBD>threshold")
            X2[:, var_index] = np.repeat(np.maximum(0, center_distances - self.dist_CBD_threshold), n_years)
        if "los_bus" in self.var_L_2: # バスLOS
            var_index = self.var_L_2.index("los_bus")
            X2[:, var_index] = los_bus[zone_indices, year_indices]
        if "los_train" in self.var_L_2: # 鉄道LOS
            var_index = self.var_L_2.index("los_train")
            X2[:, var_index] = los_train[zone_indices, year_indices]
        if "los_all" in self.var_L_2: # 全LOS
            var_index = self.var_L_2.index("los_all")
            X2[:, var_index] = los_all[zone_indices, year_indices]
        
        static_repeated = np.repeat(static_features, n_years, axis=0)
        
        if "flood_risk" in self.var_L_2: # 浸水リスク
            var_index = self.var_L_2.index("flood_risk")
            X2[:, var_index] = static_repeated[:, 0] + static_repeated[:, 1]# > 0).astype(float)
        if "school" in self.var_L_2: # 最寄りの学校までの距離
            var_index = self.var_L_2.index("school")
            X2[:, var_index] = static_repeated[:, 2]
        if "child_care" in self.var_L_2: # 子育て施設の密度
            var_index = self.var_L_2.index("child_care")
            X2[:, var_index] = np.log(1 + static_repeated[:, 3])
        if "hospitals" in self.var_L_2: # 病院の密度
            var_index = self.var_L_2.index("hospitals")
            X2[:, var_index] = np.log(1 + static_repeated[:, 4])
        if "park_area" in self.var_L_2: # 公園の面積
            var_index = self.var_L_2.index("park_area")
            X2[:, var_index] = np.log(1 + 100 * static_repeated[:, 5])
        if "commercial_area" in self.var_L_2: # 商業地の面積
            var_index = self.var_L_2.index("commercial_area")
            X2[:, var_index] = 10 * static_repeated[:, 6]
        if "price" in self.var_L_2: # 価格
            var_index = self.var_L_2.index("price")
            X2[:, var_index] = self.price_data_R[zone_indices, year_indices]
        
        W = np.zeros((self.J * self.T, len(self.var_W)))
        if "risk_area" in self.var_W: # 災害リスク
            var_index = self.var_W.index("risk_area")
            W[:, var_index] = static_repeated[:, 7]
        if "lowuse_area" in self.var_W: # 低未利用地
            var_index = self.var_W.index("lowuse_area")
            W[:, var_index] = static_repeated[:, 8]
            
        # 個人属性は移動しない選択のみ
        Z_dict = {"age": "世帯主年齢", "car": "自動車有無", "family": "子供有無", "income": "世帯年収", "move_in": "市外転入"}
        Z = df_pool[[Z_dict[var] for var in self.var_Z]].copy()
        if "age" in self.var_Z:
            #Z["世帯主年齢"] = Z["世帯主年齢"] / 100
            Z["世帯主年齢"] = (Z["世帯主年齢"] >= 65).astype(float) # 65歳以上か
        if "income" in self.var_Z:
            Z["世帯年収"] = Z["世帯年収"] / 1000
        Z = Z.values
        
        # 各サンプルの転居年のデータ
        relocation_years = df_pool["年"].values - self.start_year
        
        # シェア
        if self.res_zone == "chou_zone":
            obs_share = df_pool.groupby(["居住地_後_ゾーン", "転居年"])["転居有無"].count().unstack().fillna(0).replace(0, 0.1).T
            for i in set(self.zone_list) - set(obs_share.columns.tolist()):
                obs_share.loc[:,i] = 0.1
            # 列を選択ゾーンの順番に並べ替え
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
                # 列を選択ゾーンの順番に並べ替え
                obs_share1 = obs_share1[self.zone_list]
                obs_share1 = obs_share1.values
                obs_share1 /= np.sum(obs_share1, axis=1, keepdims=True)
                obs_share2 = read_obs_share_data(2015, self.end_year, self.zoning, self.res_zone)
                # 2015年以降のデータはobs_share2を使用
                obs_share = obs_share1
                obs_share[2015 - self.start_year:] = obs_share2
        
        # processedフォルダに保存
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
        assert "los_bus" not in self.var_L_1_hetero or "los_all" not in self.var_L_1_hetero, "los_busとlos_allは同時に使用できません"
        N = len(df_year)  # Number of samples
        X1_hetero = np.zeros((N, self.J, self.L_1_hetero))
        X1_mean = np.zeros((N, self.J, self.L_1_mean))        
        # 距離行列を再利用（estimation時に計算済みの場合）または事前計算
        if not hasattr(self, '_distance_matrix'):
            self._distance_matrix = np.zeros((self.J, self.J))
            for i, zone_i in enumerate(self.zone_list):
                for j, zone_j in enumerate(self.zone_list):
                    self._distance_matrix[i, j] = self.dist_convert(zone_i, zone_j)
        
        # 前住地のインデックスを取得
        prev_zones = df_year["居住地_前_ゾーン"].values.astype(int)
        
        # X1_mean
        if "dist_prev" in self.var_L_1_mean:
            var_index = self.var_L_1_mean.index("dist_prev")
            X1_mean[prev_zones == -1, :, var_index] = 0.0
            for alt in range(self.J):
                X1_mean[prev_zones != -1, alt, var_index] = self._distance_matrix[prev_zones[prev_zones != -1], alt]
        
        # X1_hetero
        center_distances = self.center_distances
        if "dist_CBD" in self.var_L_1_hetero:
            var_index = self.var_L_1_hetero.index("dist_CBD")
            X1_hetero[:, :, var_index] = np.tile(center_distances, (N, 1))  # 中心部からの距離
        if "dist_CBD<threshold" in self.var_L_1_hetero:
            var_index = self.var_L_1_hetero.index("dist_CBD<threshold")
            X1_hetero[:, :, var_index] = np.tile(np.maximum(0, np.minimum(center_distances, self.dist_CBD_threshold)), (N, 1))  # 中心部からの距離（閾値以下）
        if "dist_CBD>threshold" in self.var_L_1_hetero:
            var_index = self.var_L_1_hetero.index("dist_CBD>threshold")
            X1_hetero[:, :, var_index] = np.tile(np.maximum(0, center_distances - self.dist_CBD_threshold), (N, 1))  # 中心部からの距離（閾値以上）
                
        price_array = np.array(price_lists)        
        # 投資データ
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
            X1_hetero[:, :, var_index] = np.tile(los_train, (N, 1))  # 鉄道LOS
        
        # 静的データを一括取得
        static_features = self.static_features[:, :7]
        
        if "flood_risk" in self.var_L_1_hetero:
            var_index = self.var_L_1_hetero.index("flood_risk")
            X1_hetero[:, :, var_index] = np.tile(static_features[:, 0] + static_features[:, 1], (N, 1))  # 浸水リスク  > 0).astype(float)
        if "school" in self.var_L_1_hetero:
            var_index = self.var_L_1_hetero.index("school")
            X1_hetero[:, :, var_index] = np.tile(static_features[:, 2], (N, 1))  # 学校距離
        if "child_care" in self.var_L_1_hetero:
            var_index = self.var_L_1_hetero.index("child_care")
            X1_hetero[:, :, var_index] = np.tile(np.log(1 + static_features[:, 3]), (N, 1))  # 子育て施設
        if "hospitals" in self.var_L_1_hetero:
            var_index = self.var_L_1_hetero.index("hospitals")
            X1_hetero[:, :, var_index] = np.tile(np.log(1 + static_features[:, 4]), (N, 1))  # 病院
        if "park_area" in self.var_L_1_hetero:
            var_index = self.var_L_1_hetero.index("park_area")
            X1_hetero[:, :,var_index] = np.tile(np.log(1 + 100 * static_features[:, 5]), (N, 1))  # 公園
        if "commercial_area" in self.var_L_1_hetero:
            var_index = self.var_L_1_hetero.index("commercial_area")
            X1_hetero[:, :, var_index] = np.tile(10 * static_features[:, 6], (N, 1))  # 商業地

        # x2をベクトル化計算
        x2 = np.zeros((self.J, self.L_2))
        if "dist_CBD" in self.var_L_2:
            var_index = self.var_L_2.index("dist_CBD")
            x2[:, var_index] = center_distances  # 中心部からの距離
        if "dist_CBD<threshold" in self.var_L_2:
            var_index = self.var_L_2.index("dist_CBD<threshold")
            x2[:, var_index] = np.maximum(0, np.minimum(center_distances, self.dist_CBD_threshold))  # 中心部からの距離（閾値以下）
        if "dist_CBD>threshold" in self.var_L_2:
            var_index = self.var_L_2.index("dist_CBD>threshold")
            x2[:, var_index] = np.maximum(0, center_distances - self.dist_CBD_threshold)  # 中心部からの距離（閾値以上）
        if "los_bus" in self.var_L_2:
            var_index = self.var_L_2.index("los_bus")
            x2[:, var_index] = self.los_convert(investment_Rzone)
        if "los_all" in self.var_L_2:
            var_index = self.var_L_2.index("los_all")
            x2[:, var_index] = self.los_convert(investment_Rzone)
        if "los_train" in self.var_L_2:
            var_index = self.var_L_2.index("los_train")
            x2[:, var_index] = los_train  # 鉄道LOS
        if "flood_risk" in self.var_L_2:
            var_index = self.var_L_2.index("flood_risk")
            x2[:, var_index] = static_features[:, 0] + static_features[:, 1]   # 浸水リスク > 0).astype(float)
        if "school" in self.var_L_2:
            var_index = self.var_L_2.index("school")
            x2[:, var_index] = static_features[:, 2]  # 学校距離
        if "child_care" in self.var_L_2:
            var_index = self.var_L_2.index("child_care")
            x2[:, var_index] = np.log(1 + static_features[:, 3])  # 子育て施設
        if "hospitals" in self.var_L_2:
            var_index = self.var_L_2.index("hospitals")
            x2[:, var_index] = np.log(1 + static_features[:, 4])  # 病院
        if "park_area" in self.var_L_2:
            var_index = self.var_L_2.index("park_area")
            x2[:, var_index] = np.log(1 + 100 * static_features[:, 5])  # 公園
        if "commercial_area" in self.var_L_2:
            var_index = self.var_L_2.index("commercial_area")
            x2[:, var_index] = 10 * static_features[:, 6]  # 商業地
        if "price" in self.var_L_2:
            var_index = self.var_L_2.index("price")
            x2[:, var_index] = price_array / self.scale_price  # 価格
        
        # 個人属性は移動しない選択のみ
        Z_dict = {"age": "世帯主年齢", "car": "自動車有無", "family": "子供有無", "income": "世帯年収", "move_in": "市外転入"}
        Z = df_year[[Z_dict[var] for var in self.var_Z]].copy()
        if "age" in self.var_Z:
            #Z["世帯主年齢"] = Z["世帯主年齢"] / 100
            Z["世帯主年齢"] = (Z["世帯主年齢"] >= 65).astype(float) # 65歳以上か
        if "income" in self.var_Z:
            Z["世帯年収"] = Z["世帯年収"] / 1000
        Z = Z.values
            
        base_path = Path(__file__).resolve().parent.parent  # model/ から1階層上へ
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
            # 転居なしの人からランダムにmove_ratio%を抽出
            df_year = df[df["転居年"]==0].sample(frac=move_ratio).copy()
        X1_hetero, X1_mean, x2, Z = self.generate_prediction_data_2step(df_year, investment_lists, price_lists, year)
        
        predicted_probs = self.model.predict(X1_hetero, X1_mean, x2, Z, self.XZ_mask, self.spatial_weight_matrix) # (N, J)
        predicted_move = predicted_probs * df_year["拡大係数"].values.reshape(-1, 1)  # (N, J)
        
        # 確率的に配分する
        plus_Rzone = predicted_move.sum(axis=0)
        minus_Rzone = np.array([df_year.loc[df_year["居住地_前_ゾーン"]==zone, "拡大係数"].sum() for zone in self.zone_list])
                
        if zoning_type == "dev_zone":
            plus_Dzone = np.sum(plus_Rzone[np.newaxis,:] * self.zone_conversion, axis=1)
            minus_Dzone = np.sum(minus_Rzone[np.newaxis,:] * self.zone_conversion, axis=1)
            predicted_population = np.array(population_lists) + plus_Dzone - minus_Dzone
        else:
            predicted_population = np.array(population_lists) + plus_Rzone - minus_Rzone
            
        # 0以上に制限
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
            # 転居なしの人からランダムに1.5%を抽出
            df_year = df[df["転居年"]==0].sample(frac=move_ratio).copy()
        X1_hetero, X1_mean, x2, Z = self.generate_prediction_data_2step(df_year, investment_lists, initial_price_list_R, year)
        # 繰り返し計算で価格を調整
        price_lists = np.array(initial_price_list_D.copy())
        price_lists_new = price_lists.copy()
        iteration = 0
        price_unit = self.config.adjust_price_unit  # 価格調整の単位
        margin = self.config.adjust_margin  # 収束性のため1000以内の変化は無視
        upper_limit_D = (1+self.config.change_limit) * np.array(initial_price_list_D.copy())
        lower_limit_D = (1-self.config.change_limit) * np.array(initial_price_list_D.copy())
        max_iterations = self.config.max_iteration  # 最大反復回数
                
        while True:
            # 人口予測
            predicted_probs = self.model.predict(X1_hetero, X1_mean, x2, Z, self.XZ_mask, self.spatial_weight_matrix) # (N, J)
            predicted_move = predicted_probs * df_year["拡大係数"].values.reshape(-1, 1)  # (N, J)
            
            # 確率的に配分する
            plus_Rzone = predicted_move.sum(axis=0)
            minus_Rzone = np.array([df_year.loc[df_year["居住地_前_ゾーン"]==zone, "拡大係数"].sum() for zone in self.zone_list])            
            plus_Dzone = np.sum(plus_Rzone[np.newaxis,:] * self.zone_conversion, axis=1)
            minus_Dzone = np.sum(minus_Rzone[np.newaxis,:] * self.zone_conversion, axis=1)
            
            if zoning_type == "dev_zone":
                predicted_population = np.array(population_lists) + plus_Dzone - minus_Dzone
                land_demand = plus_Dzone * self.land_demand_zone_D  # 土地需要量の計算。
            else:
                predicted_population = np.array(population_lists) + plus_Rzone - minus_Rzone
                land_demand = plus_Rzone * self.land_demand_zone_R 
            # 0以上に制限
            predicted_population = np.maximum(predicted_population, 0)
            # 供給可能量も考えることが可能
            
            # 需要量がdevelopment_lists (住宅開発量) を超える場合、価格を上げる。逆も。
            excess_demand = land_demand - np.array(development_lists[-1]) > margin # 収束性のため100以内の変化は無視
            deficit_demand = land_demand - np.array(development_lists[-1]) < -margin
            
            #print(iteration)
            #print(land_demand - np.array(development_lists[-1]))
            #print(price_lists)
            # 価格調整。大ゾーン
            price_lists_new = np.where(excess_demand, price_lists + price_unit, price_lists)
            price_lists_new = np.where(deficit_demand, price_lists_new - price_unit, price_lists_new)
            diff = price_lists_new - price_lists
            # 価格が変わらない場合は終了
            if np.max(abs(diff)) < margin or iteration >= max_iterations:
                break
            if np.all(diff > 0):
                #raise ValueError("価格がすべて上昇しているため、供給が需要を満たしていない可能性があります。")
                break
                
            # 価格を調整。小ゾーン
            Rzone_idx_excess_list = [np.where(self.zone_conversion[Dzone,:]>0)[0].tolist() for Dzone in self.Dzone_list if excess_demand[Dzone]]
            Rzone_idx_deficit_list = [np.where(self.zone_conversion[Dzone,:]>0)[0].tolist() for Dzone in self.Dzone_list if deficit_demand[Dzone]]
            threshold = 0 #np.percentile(plus_Rzone, 90) # 上位50%のゾーンを対象
            var_index = self.var_L_2.index("price")
            if len(Rzone_idx_excess_list) > 0:
                Rzone_idx_excess = np.unique([item for sublist in Rzone_idx_excess_list for item in sublist if plus_Rzone[item] > threshold])
                x2[Rzone_idx_excess, var_index] += price_unit / self.scale_price
            if len(Rzone_idx_deficit_list) > 0:
                Rzone_idx_deficit = np.unique([item for sublist in Rzone_idx_deficit_list for item in sublist if plus_Rzone[item] > threshold])
                x2[Rzone_idx_deficit, var_index] -= price_unit / self.scale_price
            
            iteration += 1
            x2[:, var_index] = np.clip(x2[:, var_index], 0, None)  # 価格は0以上に制限
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
        
        # 市外転入者むけの追加効用
        if "move_in" in self.var_Z:
            move_in_idx = self.var_Z.index("move_in")
            assert self.XZ_mask[move_in_idx, self.var_L_1_hetero.index("dist_CBD")] == 1, "move_in variable must interact with dist_CBD"
            beta_distance_CBD = self.model.beta_z[len(self.var_L_1_mean) + np.sum(self.XZ_mask[:move_in_idx, :]) + self.XZ_mask[move_in_idx, :self.var_L_1_hetero.index("dist_CBD")].sum()]
            #print("Beta distance_CBD for move_in:", beta_distance_CBD)
            self.distance_CBD_utility = beta_distance_CBD * self.center_distances
            
        # SAR計算のための行列
        self.SAR_cov_mat = np.linalg.inv(np.eye(self.J) - self.model.beta_z[-1] * self.spatial_weight_matrix)
        self.SAR_cov_mat_mean = np.dot(self.SAR_cov_mat, np.random.standard_normal((self.J, 100))).mean(axis=1)
        
        # 以前の居住地からの距離の効果
        if "dist_prev" in self.var_L_1_mean:
            var_index = self.var_L_1_mean.index("dist_prev")
            #print("Beta dist_prev:", self.model.beta_z[var_index])
            self.dist_prev_utility = self.model.beta_z[var_index] * self._distance_matrix
        
        # LOS効果
        investment_Rzone = np.maximum(
            np.dot(np.asarray(investment_lists, dtype=float), self.investment_distribution), 1.0
        )  # shape: (Dzone,) @ (Dzone, Rzone) -> (Rzone,)
        los_idx = 1 + self.var_L_2.index("los_all") # 1 +はconstant項
        los_year_idx = year - self.start_year if year <= self.end_year else self.end_year - self.start_year
        delta += self.model.beta_x[los_idx] * self.los_convert(investment_Rzone - self.investment_data_R[:, los_year_idx]) # LOSの影響を追加
        
        # price変化の効果
        price_year_idx = year - self.start_year if year <= self.end_year else self.end_year - self.start_year
        price_idx = 1 + self.var_L_2.index("price")
        # initial_price_list_Dをinitial_price_list_Rに変換
        initial_price_list_R = self.zone_conversion.T @ np.array(initial_price_list_D) # (Dzone,) -> (Rzone,)へ重み付き平均
        delta += self.model.beta_x[price_idx] * (initial_price_list_R - self.price_data_R[:,price_year_idx]) / self.scale_price
        
        population_Rzone = np.maximum(
            np.dot(np.asarray(population_lists, dtype=float), self.population_distribution),
            1.0
            )  # (Dzone,) @ (Dzone, Rzone) -> (Rzone,)
        minus_Rzone, exiting_Rzone = self.calculate_moving_out_population(year, population_Rzone)
        minus_Dzone = self.zone_conversion @ minus_Rzone # (Dzone, Rzone) @ (Rzone,) -> (Dzone,)
        exiting_Dzone = self.zone_conversion @ exiting_Rzone
        
        # 繰り返し計算で価格を調整
        price_D=np.array(initial_price_list_D)
        price_beta_idx = 1 + self.var_L_2.index("price")
        price_beta = self.model.beta_x[price_beta_idx]
        margin=self.config.adjust_margin # 収束性のためx以内の変化は無視
        price_unit=self.config.adjust_price_unit # 価格調整の単位
        lower_limit=(1-self.config.change_limit) * price_D # 価格の変化量の下限
        upper_limit=(1+self.config.change_limit) * price_D # 価格の変化量の上限
        
        # 1. 逐次計算で価格を調整する
        for iteration in range(self.config.max_iteration):
            # 人口移動
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
            
            # 需要量がdevelopment_lists (住宅開発量) を超える場合、価格を上げる。逆も。
            diff_demand = land_demand - np.asarray(development_lists[-1], dtype=float)  # (Dzone,)
            excess_mask = diff_demand > margin    # 需要超過
            deficit_mask = diff_demand < -margin  # 供給超過
            
            # 価格更新
            new_price_D = self.update_price(initial_price_D=price_D, price_unit=price_unit, excess_mask=excess_mask, deficit_mask=deficit_mask)
            if self.check_convergence(diff_demand, threshold=margin):
                price_D = new_price_D.copy()
                break
            
            # 小ゾーンの選択効用を更新
            delta = self.update_delta(delta, price_beta, price_unit, excess_mask, deficit_mask)
            
            # 次のループ用に価格を更新＆クリップ
            price_D = np.clip(new_price_D, lower_limit, upper_limit)
            
        """
        # 2. 最適化計算で価格を調整する
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
        # 入力人口と予測人口の総和が同じ
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
        # 外生的に各ゾーンの転入人口を与える場合
        if year < 2011:
            inflow_pop = self.inflow_pop[:,0]
        elif 2011 <= year <= 2024:
            year_index = year - 2011
            inflow_pop = self.inflow_pop[:,year_index]
        else:
            year_index = 2024 - 2011
            inflow_pop = self.inflow_pop[:,year_index]
        # 内生的に決める場合は以下を追加
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
            land_demand = plus_Dzone * self.land_demand_zone_D # 土地需要量の計算。
        else:
            predicted_population = np.array(population_lists) + plus_Rzone - minus_Rzone - exiting_Rzone
            land_demand = plus_Rzone * self.land_demand_zone_R
        # 0以上に制限
        predicted_population = np.maximum(predicted_population, 0.0)
        return predicted_population, land_demand
    
    def aggregate_migration(self, year, delta, population_lists, minus_Rzone, minus_Dzone, exiting_Rzone, exiting_Dzone, zoning_type):
        exp_delta = np.exp(delta[np.newaxis,:] + self.SAR_cov_mat_mean[np.newaxis,:] + self.dist_prev_utility) # (Rzone,Rzone)
        predicted_probs = exp_delta / np.sum(exp_delta, axis=1, keepdims=True)  # (Rzone,Rzone)
        plus_Rzone =(minus_Rzone[np.newaxis,:] @ predicted_probs).reshape(-1) + self.calculate_inflow_population(year, delta=delta) # (Rzone,)
        plus_Dzone = self.zone_conversion @ plus_Rzone # (Dzone,Rzone) @ (Rzone,) -> (Dzone,)
        
        if zoning_type == "dev_zone":
            predicted_population = np.array(population_lists) + plus_Dzone - minus_Dzone - exiting_Dzone
            land_demand = plus_Dzone * self.land_demand_zone_D # 土地需要量の計算。
        else:
            predicted_population = np.array(population_lists) + plus_Rzone - minus_Rzone - exiting_Rzone
            land_demand = plus_Rzone * self.land_demand_zone_R
        # 0以上に制限
        predicted_population = np.maximum(predicted_population, 0.0)
        return predicted_population, land_demand
    
    def update_price(self, initial_price_D, price_unit, excess_mask, deficit_mask):       
        # 価格更新（Dレベル）
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
        # 価格を調整。小ゾーン
        if np.any(excess_mask):
            #############
            # excessなDに属するRの価格を上げる
            r_flag = np.unique(np.concatenate([self.D_to_R_idx[d] for d in np.nonzero(excess_mask)[0]]))  # (Rzone,)
            #############
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
        
        # LOS効果
        investment_R = np.maximum(
            np.dot(np.asarray(investment_D, dtype=float), self.investment_distribution), 1.0
        )  # shape: (Dzone,) @ (Dzone, Rzone) -> (Rzone,)
        los_idx = 1 + self.var_L_2.index("los_all") # 1 +はconstant項
        los_year_idx = year - self.start_year if year <= self.end_year else self.end_year - self.start_year
        delta += self.model.beta_x[los_idx] * self.los_convert(investment_R - self.investment_data_R[:, los_year_idx]) # LOSの影響を追加
        
        # price変化の効果
        price_year_idx = year - self.start_year if year <= self.end_year else self.end_year - self.start_year
        price_idx = 1 + self.var_L_2.index("price")
        # initial_price_list_Dをinitial_price_list_Rに変換
        price_R = self.zone_conversion.T @ np.array(price_D) # (Dzone,) -> (Rzone,)へ重み付き平均
        delta += self.model.beta_x[price_idx] * (price_R - self.price_data_R[:,price_year_idx]) / self.scale_price
        
        # 以前の居住地からの距離の効果
        var_index = self.var_L_1_mean.index("dist_prev")
        dist_prev_utility = self.model.beta_z[var_index] * self._distance_matrix
        
        # 車を持っている人むけの追加効用
        if "car" in self.var_Z:
            car_idx = self.var_Z.index("car")
            beta_los_all = self.model.beta_z[len(self.var_L_1_mean) + np.sum(self.XZ_mask[:car_idx, :]) + self.XZ_mask[car_idx, :self.var_L_1_hetero.index("los_all")].sum()]
            los_all_utility = beta_los_all * self.investment_data_R[:, los_year_idx]
        
        # 車を持っている人の期待最大効用
        delta_car = delta + los_all_utility + dist_prev_utility
        expected_utility_car = self.expected_maximum_utility(delta_car)
        # 車を持っていない人の期待最大効用
        delta_no_car = delta + dist_prev_utility
        expected_utility_no_car = self.expected_maximum_utility(delta_no_car)
        
        # 金額換算
        welfare_car = expected_utility_car * (self.scale_price / abs(self.model.beta_x[price_idx]))
        welfare_no_car = expected_utility_no_car * (self.scale_price / abs(self.model.beta_x[price_idx]))
        
        return welfare_car, welfare_no_car
        

if __name__ == "__main__":
    # Generate estimation data
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
    
