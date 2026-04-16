"""
設定ファイルの読み込みと管理を行うモジュール
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


class ConfigManager:
    """設定値を一元管理するクラス"""
    
    # デフォルト値を一箇所に定義
    DEFAULT_CONFIG = {
        "model": {
            "start_year": 2009,
            "end_year": 2023,
            "ref_year": 1,
            "dev_zone": "19_zone",
            "res_zone": "pop_zone",
            "n_forward": 100,
            "n_perturb": 50,
            "seed": 42,
            "horizon": 15,
            "discount_factor": 0.99,
            "num_features_gov": 4,
            "num_features_dev": 4,
            "bootstrap": False,
            "n_bootstrap": 25,
        },
        "scaling": {
            "population_density": 1e3,
            "population": 1e4,
            "los": 1e2,
            "development_res": 1e4,
            "development_shop": 1e4,
            "price": 1e4,
            "land_demand": 1e4,
        },
        "ray": {},
        "counter_factual": {},
        "land_price_adjustment": {}
    }
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        設定ファイルを読み込んで初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config_path = self._resolve_config_path(config_path)
        self._raw_config = self._load_config()
        self._config = self._merge_with_defaults()
        self._validate_config()
        self._initialize_attributes()
    
    def _resolve_config_path(self, config_path: str) -> Path:
        """
        実行場所に依存せず config.yaml を見つけられるように探索する
        """
        path = Path(config_path)
        if path.is_absolute() and path.exists():
            return path

        # まずはカレントディレクトリ基準
        candidate = Path.cwd() / path
        if candidate.exists():
            return candidate

        # このモジュールの親ディレクトリを辿りながら探索
        for parent in Path(__file__).resolve().parents:
            candidate = parent / path
            if candidate.exists():
                return candidate

        # 最後のフォールバック（存在しなければ後で FileNotFoundError）
        return path
    
    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込む"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {self.config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    
    def _merge_with_defaults(self) -> Dict[str, Any]:
        """デフォルト値とマージ"""
        merged = {}
        for section, defaults in self.DEFAULT_CONFIG.items():
            merged[section] = {**defaults, **self._raw_config.get(section, {})}
        return merged
    
    def _validate_config(self):
        """設定値の妥当性をチェック"""
        model = self._config["model"]
        
        # 年に関するバリデーション
        if model["start_year"] >= model["end_year"]:
            raise ValueError(f"start_year ({model['start_year']}) は end_year ({model['end_year']}) より小さくする必要があります")
        
        if model["ref_year"] < 0:
            raise ValueError(f"ref_year ({model['ref_year']}) は0以上である必要があります")
        
        # 数値の範囲チェック
        if not 0 < model["discount_factor"] <= 1:
            raise ValueError(f"discount_factor ({model['discount_factor']}) は0より大きく1以下である必要があります")
        
        if model["n_forward"] <= 0:
            raise ValueError(f"n_forward ({model['n_forward']}) は正の値である必要があります")
        
        if model["n_perturb"] <= 0:
            raise ValueError(f"n_perturb ({model['n_perturb']}) は正の値である必要があります")
        
        if model["horizon"] <= 0:
            raise ValueError(f"horizon ({model['horizon']}) は正の値である必要があります")
        
        if model["move_ratio"] < 0 or model["move_ratio"] > 1:
            raise ValueError(f"move_ratio ({model['move_ratio']}) は0から1の間である必要があります")
        
        # スケーリング値のチェック
        for key, value in self._config["scaling"].items():
            if float(value) <= 0:
                raise ValueError(f"scaling.{key} ({value}) は正の値である必要があります")
    
    def _initialize_attributes(self):
        """設定値を属性として設定"""
        model = self._config["model"]
        scaling = self._config["scaling"]
        
        # モデル設定
        self.start_year = model["start_year"]
        self.end_year = model["end_year"]
        self.ref_year = model["ref_year"]
        self.dev_zone = model["dev_zone"]
        self.res_zone = model["res_zone"]
        self.move_ratio = model["move_ratio"]
        self.n_forward = model["n_forward"]
        self.n_perturb = model["n_perturb"]
        self.seed = model["seed"]
        self.horizon = model["horizon"]
        self.discount_factor = model["discount_factor"]
        self.num_features_gov = model["num_features_gov"]
        self.num_features_dev = model["num_features_dev"]
        self.bootstrap = model["bootstrap"]
        self.n_bootstrap = 1 if not model["bootstrap"] else model["n_bootstrap"]
        
        # スケーリング設定（float型に変換）
        self.scale_pop_dense = float(scaling["population_density"])
        self.scale_pop = float(scaling["population"])
        self.scale_los = float(scaling["los"])
        self.scale_dev_res = float(scaling["development_res"])
        self.scale_dev_shop = float(scaling["development_shop"])
        self.scale_price = float(scaling["price"])
        self.scale_land_demand = float(scaling["land_demand"])
        
        # Ray設定
        self.ray_config = self._config.get("ray", {})
        
        # counter_factual設定
        cf = self._config.get("counter_factual", {})
        self.theta_gov = cf.get("theta_gov", None)
        self.theta_dev = cf.get("theta_dev", None)
        self.target_zone = cf.get("target_zone", None)
        self.sim_start_year = cf.get("sim_start_year", None)
        
        land_price_adjustment = self._config.get("land_price_adjustment", {})
        self.new_house_ratio = land_price_adjustment.get("new_house_ratio", 0.616)  # 転入者のうち新築持ち家の割合
        self.per_person_land_demand = land_price_adjustment.get("per_person_land_demand", 13.414)  #1人あたり平均敷地面積[m2] 分譲マンションと戸建ての加重平均
        self.adjust_price_unit = land_price_adjustment.get("adjust_price_unit", 2000)  # 土地価格調整の単位価格（x円/m2ごとに土地価格を調整）
        self.adjust_margin = land_price_adjustment.get("adjust_margin", 2000)  # 収束性のためのマージン（x円/m2未満の調整は行わない）
        self.change_limit = land_price_adjustment.get("change_limit", 0.05)  # 土地価格の変化率の上限（±x*100 %以内に抑える）
        self.max_iteration = land_price_adjustment.get("max_iteration", 50)  # 土地価格調整の最大繰り返し回数
        
        # 乱数シードの設定
        np.random.seed(self.seed)
    
    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        設定値を取得
        
        Args:
            section: セクション名（例: "model", "scaling"）
            key: キー名（Noneの場合はセクション全体を返す）
            default: デフォルト値
        
        Returns:
            設定値
        """
        if key is None:
            return self._config.get(section, default)
        return self._config.get(section, {}).get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書として取得"""
        return self._config.copy()

