"""Configuration loading and validation utilities for Chapter 5."""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


class ConfigManager:
    """Centralized configuration manager."""
    
    # Default values used when a key is omitted from config.yaml.
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
        """Load the configuration file and initialize defaults."""
        self.config_path = self._resolve_config_path(config_path)
        self._raw_config = self._load_config()
        self._config = self._merge_with_defaults()
        self._validate_config()
        self._initialize_attributes()
    
    def _resolve_config_path(self, config_path: str) -> Path:
        """Resolve the config path without depending on the current working directory."""
        path = Path(config_path)
        if path.is_absolute() and path.exists():
            return path

        # First try the current working directory.
        candidate = Path.cwd() / path
        if candidate.exists():
            return candidate

        # Then walk up from this module's location.
        for parent in Path(__file__).resolve().parents:
            candidate = parent / path
            if candidate.exists():
                return candidate

        # Final fallback. A later read will raise if it still does not exist.
        return path
    
    def _load_config(self) -> Dict[str, Any]:
        """Read the YAML config file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    
    def _merge_with_defaults(self) -> Dict[str, Any]:
        """Merge the loaded config with the default settings."""
        merged = {}
        for section, defaults in self.DEFAULT_CONFIG.items():
            merged[section] = {**defaults, **self._raw_config.get(section, {})}
        return merged
    
    def _validate_config(self):
        """Validate the merged configuration."""
        model = self._config["model"]
        
        # Validate year settings.
        if model["start_year"] >= model["end_year"]:
            raise ValueError(f"start_year ({model['start_year']}) must be smaller than end_year ({model['end_year']})")
        
        if model["ref_year"] < 0:
            raise ValueError(f"ref_year ({model['ref_year']}) must be non-negative")
        
        # Validate numeric ranges.
        if not 0 < model["discount_factor"] <= 1:
            raise ValueError(f"discount_factor ({model['discount_factor']}) must be in (0, 1]")
        
        if model["n_forward"] <= 0:
            raise ValueError(f"n_forward ({model['n_forward']}) must be positive")
        
        if model["n_perturb"] <= 0:
            raise ValueError(f"n_perturb ({model['n_perturb']}) must be positive")
        
        if model["horizon"] <= 0:
            raise ValueError(f"horizon ({model['horizon']}) must be positive")
        
        if model["move_ratio"] < 0 or model["move_ratio"] > 1:
            raise ValueError(f"move_ratio ({model['move_ratio']}) must be between 0 and 1")
        
        # Validate scaling values.
        for key, value in self._config["scaling"].items():
            if float(value) <= 0:
                raise ValueError(f"scaling.{key} ({value}) must be positive")
    
    def _initialize_attributes(self):
        """Expose validated config values as attributes."""
        model = self._config["model"]
        scaling = self._config["scaling"]
        
        # Model settings.
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
        
        # Scaling settings converted to float.
        self.scale_pop_dense = float(scaling["population_density"])
        self.scale_pop = float(scaling["population"])
        self.scale_los = float(scaling["los"])
        self.scale_dev_res = float(scaling["development_res"])
        self.scale_dev_shop = float(scaling["development_shop"])
        self.scale_price = float(scaling["price"])
        self.scale_land_demand = float(scaling["land_demand"])
        
        # Ray settings.
        self.ray_config = self._config.get("ray", {})
        
        # Counterfactual settings.
        cf = self._config.get("counter_factual", {})
        self.theta_gov = cf.get("theta_gov", None)
        self.theta_dev = cf.get("theta_dev", None)
        self.target_zone = cf.get("target_zone", None)
        self.sim_start_year = cf.get("sim_start_year", None)
        
        land_price_adjustment = self._config.get("land_price_adjustment", {})
        self.new_house_ratio = land_price_adjustment.get("new_house_ratio", 0.616)  # Share of new detached housing among in-movers.
        self.per_person_land_demand = land_price_adjustment.get("per_person_land_demand", 13.414)  # Average land demand per person in m2.
        self.adjust_price_unit = land_price_adjustment.get("adjust_price_unit", 2000)  # Price adjustment step in yen/m2.
        self.adjust_margin = land_price_adjustment.get("adjust_margin", 2000)  # Ignore smaller changes to improve convergence.
        self.change_limit = land_price_adjustment.get("change_limit", 0.05)  # Maximum relative price change per iteration.
        self.max_iteration = land_price_adjustment.get("max_iteration", 50)  # Maximum number of price-adjustment iterations.
        
        # Random seed.
        np.random.seed(self.seed)
    
    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """Return a config value or a whole section."""
        if key is None:
            return self._config.get(section, default)
        return self._config.get(section, {}).get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return the merged configuration as a dictionary."""
        return self._config.copy()
