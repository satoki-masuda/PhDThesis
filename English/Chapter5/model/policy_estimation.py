"""政策関数の推定と、その摂動版モデルを扱うモジュール。"""

import joblib
import sys
sys.path.append("..")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from utils.config_manager import ConfigManager
from utils.data_loader import load_simulation_data

class HeteroMLEPolicyFunction:
    """
    平均と分散を別々の説明変数で表すヘテロ分散型の政策関数。
    """
    def __init__(self):
        self.beta_ = None
        self.gamma_ = None
        self.mean_features_ = None
        self.var_features_ = None
        self.y_scale = 1.0
        self.beta_se_ = None
        self.gamma_se_ = None
        self.beta_t_ = None
        self.gamma_t_ = None
    
    @staticmethod
    def _softplus(x):  # 数値安定版
        #return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
        return np.log1p(np.exp(x))

    @staticmethod
    def _neg_loglike(params, X_mean, X_var, y, sigma_min=1e-3):
        k1 = X_mean.shape[1]
        beta = params[:k1]
        gamma = params[k1:]
        mu = X_mean @ beta
        log_sigma = np.clip(X_var @ gamma, -10, 10)
        sigma = np.exp(log_sigma)
        #log_sigma = X_var @ gamma
        #sigma = HeteroMLEPolicyFunction._softplus(log_sigma) + sigma_min
        resid = y - mu
        ll_i = -0.5 * np.log(2 * np.pi) - log_sigma - 0.5 * (resid / sigma) ** 2
        #if np.random.rand() < 0.01:
        #    print(f"Current neg log-lik: {-np.sum(ll_i):.4f}")
        return -np.sum(ll_i)

    @staticmethod
    def _mape(y_true, y_pred):
        y_true = np.asarray(y_true, float)
        y_pred = np.asarray(y_pred, float)
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def estimate_policy_function_mle(self, X_mean, X_var, y, y_scale, output_path=None):
        if not isinstance(X_mean, pd.DataFrame):
            X_mean = pd.DataFrame(X_mean)
        if not isinstance(X_var, pd.DataFrame):
            X_var = pd.DataFrame(X_var)
        self.y_scale = y_scale
        y = np.asarray(y, float).ravel() / self.y_scale

        n, k1 = X_mean.shape
        _, k2 = X_var.shape

        x0 = np.zeros(k1+k2)
        res = minimize(
            self._neg_loglike,
            x0,
            args=(X_mean.values, X_var.values, y),
            method="BFGS",
        )
        #if not res.success:
        #    raise RuntimeError(f"MLE failed: {res.message}")

        params = res.x
        self.beta_ = params[:k1]
        self.gamma_ = params[k1:]
        self.mean_features_ = list(X_mean.columns)
        self.var_features_ = list(X_var.columns)
        self.all_features_ = self.mean_features_ + [feat for feat in self.var_features_ if feat not in self.mean_features_]

        # ===== 標準誤差とt値 =====
        cov = None
        try:
            H_inv = res.hess_inv  # BFGSの逆ヘッセ近似
            # scipyのバージョンによっては行列ライクオブジェクトなので素直にndarrayに
            if hasattr(H_inv, "todense"):
                H_inv = H_inv.todense()
            cov = np.asarray(H_inv, dtype=float)
            if cov.shape != (k1 + k2, k1 + k2):
                # 念のため形が変なら諦める
                cov = None
        except Exception:
            cov = None

        if cov is not None:
            var_params = np.diag(cov)
            # 負の分散が出たらNaNに飛ばす
            var_params = np.where(var_params > 0, var_params, np.nan)
            se_params = np.sqrt(var_params)
            self.beta_se_ = se_params[:k1]
            self.gamma_se_ = se_params[k1:]
            self.beta_t_ = self.beta_ / self.beta_se_
            self.gamma_t_ = self.gamma_ / self.gamma_se_
        else:
            print("[MLE policy] WARNING: could not compute Hessian-based standard errors.")
        
        # ===== in-sample fit =====    
        mu_hat = X_mean.values @ self.beta_
        mae = np.mean(np.abs(y - mu_hat)) * self.y_scale
        mape = self._mape(y, mu_hat)
        print(f"[MLE policy] n={n}")
        print(f"[MLE policy] MAE (vs μ): {mae:.4f}")
        print(f"[MLE policy] MAPE (vs μ): {mape:.2f}%")
        print(f"[MLE policy] final beta: {self.beta_}")
        print("[MLE policy] std. errors (beta):", self.beta_se_)
        print("[MLE policy] t-values   (beta):", self.beta_t_)
        print(f"[MLE policy] final gamma: {self.gamma_}")
        print("[MLE policy] std. errors (gamma):", self.gamma_se_)
        print("[MLE policy] t-values   (gamma):", self.gamma_t_)
        
        #log_sigma = np.clip(X_var.values @ self.gamma_, -10, 10)
        #sigma_hat = np.exp(log_sigma)
        #resid = y - mu_hat
        #ll_i = -0.5 * np.log(2 * np.pi) - log_sigma - 0.5 * (resid / sigma_hat) ** 2
        #avg_ll = np.mean(ll_i)
        #initial_ll = self._neg_loglike(x0, X_mean.values, X_var.values, y)
        #print(f"[MLE policy] initial log-lik: {initial_ll:.4f}, avg per obs: {initial_ll/n:.4f}")
        #print(f"[MLE policy] final log-lik: {np.sum(ll_i):.4f}, avg per obs: {avg_ll:.4f}")        
        
        neg_ll0 = self._neg_loglike(x0, X_mean.values, X_var.values, y)
        neg_ll_final = self._neg_loglike(np.r_[self.beta_, self.gamma_], X_mean.values, X_var.values, y)
        print(f"[MLE policy] initial neg log-lik: {neg_ll0:.4f}, avg per obs: {neg_ll0/n:.4f}")
        print(f"[MLE policy] final neg log-lik: {neg_ll_final:.4f}, avg per obs: {neg_ll_final/n:.4f}")
        print(f"[MLE policy] log-lik improvement: {-(neg_ll_final - neg_ll0):.4f}")
        
        # ===== CSV 用のパラメータ表 =====
        param_names = (
            [f"{name}" for name in self.mean_features_] +
            [f"{name}" for name in self.var_features_]
        )
        estimates = np.r_[np.round(self.beta_, 3), np.round(self.gamma_, 3)]
        std_errors = np.r_[np.round(self.beta_se_, 2), np.round(self.gamma_se_, 2)]
        t_values = np.r_[np.round(self.beta_t_, 2), np.round(self.gamma_t_, 2)]
        groups = np.r_[
            np.repeat("beta", k1),
            np.repeat("gamma", k2),
        ]

        summary_df = pd.DataFrame({
            "param": param_names,
            "group": groups,
            "estimate": estimates,
            "std_error": std_errors,
            "t_value": t_values,
        })

        # ===== 末尾にスカラー指標を追加 =====
        metrics_df = pd.DataFrame([
            {"param": "n",                  "group": "summary", "estimate": n,           "std_error": np.nan, "t_value": np.nan},
            {"param": "MAE",                "group": "summary", "estimate": mae,         "std_error": np.nan, "t_value": np.nan},
            {"param": "MAPE",               "group": "summary", "estimate": mape,        "std_error": np.nan, "t_value": np.nan},
            {"param": "initial_neg_loglik", "group": "summary", "estimate": neg_ll0, "std_error": np.nan, "t_value": np.nan},
            {"param": "final_neg_loglik",   "group": "summary", "estimate": neg_ll_final,   "std_error": np.nan, "t_value": np.nan},
        ])

        summary_all = pd.concat([summary_df, metrics_df], ignore_index=True)
        
        if output_path is not None:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            joblib.dump(
                {
                    "beta": self.beta_,
                    "gamma": self.gamma_,
                    "mean_features": self.mean_features_,
                    "var_features": self.var_features_,
                    "y_scale": self.y_scale,
                },
                output_path,
            )
            print(f"MLE policy saved to {output_path}")

            csv_path = Path(output_path).with_suffix(".csv")
            summary_all.to_csv(csv_path, index=False)
            print(f"MLE summary saved to {csv_path}")

    def load_model(self, path):
        obj = joblib.load(path)
        self.beta_ = obj["beta"]
        self.gamma_ = obj["gamma"]
        self.mean_features_ = obj["mean_features"]
        self.var_features_ = obj["var_features"]
        self.all_features_ = self.mean_features_ + [feat for feat in self.var_features_ if feat not in self.mean_features_]
        self.n_params_mean_ = len(self.mean_features_)
        self.n_params_var_ = len(self.var_features_)
        self.y_scale = obj.get("y_scale", 1.0)
    
    def predict(self, X, nu, t=None, actor=None, zones=None):
        if self.beta_ is None:
            raise ValueError("MLE policy not estimated/loaded.")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.all_features_)
        Xm = X[self.mean_features_].values
        Zv = X[self.var_features_].values
        mu = (Xm @ self.beta_) * self.y_scale
        log_sigma = np.clip(Zv @ self.gamma_, -10, 10)
        sigma = np.exp(log_sigma) * self.y_scale
        nu = np.asarray(nu, float)
        if Xm.shape[0] == 1 and nu.ndim == 1:
            return mu[0] + sigma[0] * nu
        if nu.shape == mu.shape:
            return mu + sigma * nu
        raise ValueError(f"Shape mismatch in predict. X shape: {X.values.shape}, nu shape: {nu.shape}, mu shape: {mu.shape}")
    
    def plot_fit(self, X_mean, X_var, y, y_scale=1.0, n_std=1.96):
        """
        真のyと μ(s) を比較し，μ±nσ のバンドも描く。
        """
        X_mean = pd.DataFrame(X_mean)
        X_var = pd.DataFrame(X_var)

        y_scaled = np.asarray(y, float)
        mu = (X_mean.values @ self.beta_) * y_scale
        log_sigma = np.clip(X_var.values @ self.gamma_, -10, 10)
        sigma = np.exp(log_sigma) * y_scale

        y_lower = mu - n_std * sigma
        y_upper = mu + n_std * sigma 

        plt.figure(figsize=(10, 6))
        plt.errorbar(
            y_scaled,
            mu,
            yerr=np.vstack([mu - y_lower, y_upper - mu]),
            fmt='o',
            alpha=0.7,
            ecolor='skyblue',
            mec='k',
            mfc='white',
            elinewidth=1.2,
            capsize=2,
            label=f"μ ± {n_std}σ interval"
        )

        plt.plot([y_scaled.min(), y_scaled.max()], [y_scaled.min(), y_scaled.max()],
                'r--', lw=2, label="Perfect prediction (y = ŷ)")
        plt.xlabel("True Values (Observed y)")
        plt.ylabel("Predicted Median (ŷ, q=0.5)")
        plt.title(f"True vs Predicted with Quantile Intervals (μ ± {n_std}σ)")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        avg_width = np.mean(y_upper - y_lower)
        print(f"Average interval width (μ ± {n_std}σ): {avg_width:.4f}")
        
        residuals = y_scaled - mu
        plt.figure(figsize=(10, 6))
        plt.scatter(mu, residuals, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted Values')
        plt.show()

    def plot_sensitivity(self, base_X, feature, fixed_dummies=None,
                        num_points=50, y_scale=1.0, n_std=1.96):
        assert isinstance(base_X, pd.DataFrame)
        x_min, x_max = base_X[feature].min(), base_X[feature].max()
        grid = np.linspace(x_min, x_max, num_points)
        base_row = base_X.median(numeric_only=True)

        # ダミーを固定する: 実データから1行拾ってもいい
        if fixed_dummies is not None:
            for col, val in fixed_dummies.items():
                base_row[col] = val
        # そうでなければ最初の観測を使う
        else:
            for col in base_X.columns:
                if base_X[col].isin([0, 1]).all():
                    base_row[col] = base_X.iloc[0][col]

        X_eval = pd.DataFrame(
            [base_row.copy() for _ in range(num_points)]
        )
        X_eval[feature] = grid

        mu = (X_eval[self.mean_features_].values @ self.beta_) * self.y_scale
        log_sigma = np.clip(X_eval[self.var_features_].values @ self.gamma_, -10, 10)
        sigma = np.exp(log_sigma) * self.y_scale

        plt.figure(figsize=(10, 5))
        plt.plot(grid, mu, label="μ(s)", lw=2)
        plt.fill_between(
            grid,
            np.maximum(mu - n_std * sigma, 0.0),
            mu + n_std * sigma,
            color="skyblue",
            alpha=0.3,
            label=f"μ ± {n_std}σ",
        )
        plt.ylim((0, None))
        plt.xlabel(feature)
        plt.ylabel("predicted action")
        plt.title(f"Sensitivity of μ,σ to {feature}")
        plt.legend()
        plt.tight_layout()
        plt.show()


class PerturbedModel:
    """推定済み政策関数に平均シフトや分散倍率を加えるラッパー。"""
    """
    MLEベースの方策を少しだけずらすためのラッパ。
    ・mean_shift: μ(s)に足す定数または係数
    ・sigma_scale: σ(s)をこの倍率で膨らませる
    どちらか一方だけでもよい。
    """
    def __init__(self, base_policy, mean_shift=0.0, sigma_scale=1.0):
        # base_policy は HeteroMLEPolicyFunction のインスタンス想定
        self.base = base_policy
        random_sign = np.random.choice([-1, 1])
        self.mean_shift = random_sign * float(mean_shift)
        self.sigma_scale = float(sigma_scale)

    def predict(self, X, nu, t=None, actor=None, zones=None):
        # baseでμ(s)+σ(s)νを作る
        if not hasattr(self.base, "beta_"):
            raise ValueError("PerturbedModel expects an MLE-based policy (has beta_).")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.base.all_features_)
        # 元のμとσを再計算する
        Xm = X[self.base.mean_features_].values
        Zv = X[self.base.var_features_].values
        mu = (Xm @ self.base.beta_) * self.base.y_scale
        log_sigma = Zv @ self.base.gamma_
        log_sigma = np.clip(log_sigma, -10, 10)
        sigma = np.exp(log_sigma) * self.base.y_scale

        nu = np.asarray(nu, dtype=float)
        # Xが1行でnuがベクトルのケース
        if Xm.shape[0] == 1 and nu.ndim == 1:
            mu_p = mu[0] + self.mean_shift
            sigma_p = sigma[0] * self.sigma_scale
            return mu_p + sigma_p * nu

        # 行ごとのnu
        if nu.shape == mu.shape:
            mu_p = mu + self.mean_shift
            sigma_p = sigma * self.sigma_scale
            return mu_p + sigma_p * nu

        raise ValueError("Shape mismatch in PerturbedModel.predict")


if __name__ == "__main__":
    config = ConfigManager("config.yaml")
    data_bundle = load_simulation_data(config)
    data = data_bundle.data
    Dzone_list = data_bundle.Dzone_list
    columns_zone_fe = data_bundle.columns_zone_fe
    columns_year_fe = data_bundle.columns_year_fe
    mean_features_gov = ['pop_dense_t', 'population_t', 'develop_res_t', 'distance_CBD', 'risk'] 
    mean_features_dev = ['pop_dense_t', 'population_t', 'los_t', 'land_price_t', 'distance_CBD', 'risk'] 
    var_features_gov = ['const'] + mean_features_gov
    var_features_dev = ['const'] + mean_features_dev

    scale_los = config.scale_los
    scale_dev_res = config.scale_dev_res

    # Step 1: モデル推定
    gov_mean_X = data.loc[:, mean_features_gov].copy()
    """
    # 既存の特徴量から2乗項を作成し、特徴量として追加
    for feat in mean_features_gov:
        sq_col = feat + "_sq"
        data[sq_col] = data.loc[:, feat] ** 2
        gov_mean_X[sq_col] = gov_mean_X.loc[:, feat] ** 2  
    
    # 既存特徴量から交互作用項を作成し、特徴量として追加
    feature_pairs = list(combinations(mean_features_gov, 2))
    for feat1, feat2 in feature_pairs:
        interaction_col = f"{feat1}_x_{feat2}"
        data[interaction_col] = data.loc[:, feat1] * data.loc[:, feat2]
        gov_mean_X[interaction_col] = gov_mean_X.loc[:, feat1] * gov_mean_X.loc[:, feat2]
    """
    # columns_year_fe + columns_zone_feを追加
    gov_mean_X = pd.concat([gov_mean_X, data[columns_year_fe + columns_zone_fe]], axis=1)
    gov_var_X = data[var_features_gov]
    
    dev_mean_X = data.loc[:, mean_features_dev]
    """
    # 既存の特徴量から2乗項を作成し、特徴量として追加
    for feat in mean_features_dev:
        sq_col = feat + "_sq"
        if sq_col in data.columns:
            dev_mean_X[sq_col] = data[sq_col]
        else:
            data[sq_col] = data.loc[:, feat] ** 2
            dev_mean_X[sq_col] = dev_mean_X.loc[:, feat] ** 2
    
    # 既存特徴量から交互作用項を作成し、特徴量として追加
    feature_pairs = list(combinations(mean_features_dev, 2))
    for feat1, feat2 in feature_pairs:
        interaction_col = feat1 + "_x_" + feat2
        if interaction_col in data.columns:
            dev_mean_X[interaction_col] = data[interaction_col]
        else:
            data[interaction_col] = data.loc[:, feat1] * data.loc[:, feat2]
            dev_mean_X[interaction_col] = dev_mean_X.loc[:, feat1] * dev_mean_X.loc[:, feat2]
    """
    # columns_year_fe + columns_zone_feを追加
    dev_mean_X = pd.concat([dev_mean_X, data[columns_year_fe + columns_zone_fe]], axis=1)
    dev_var_X = data[var_features_dev]
    y_gov = data['los_t_raw'].values
    y_dev = data['develop_res_t_raw'].values

    gov_model = HeteroMLEPolicyFunction()
    gov_model.estimate_policy_function_mle(
        gov_mean_X,
        gov_var_X,
        y_gov,
        y_scale=scale_los,
        output_path="output/estimates/gov_model_mle.joblib"
    )

    dev_model = HeteroMLEPolicyFunction()
    dev_model.estimate_policy_function_mle(
        dev_mean_X,
        dev_var_X,
        y_dev,
        y_scale=scale_dev_res,
        output_path="output/estimates/dev_model_mle.joblib"
    )

    gov_model.plot_fit(gov_mean_X, gov_var_X, y_gov, y_scale=gov_model.y_scale)
    X_sens = pd.concat([gov_mean_X, gov_var_X[[col for col in gov_var_X.columns if col not in gov_mean_X.columns]]], axis=1)
    for f in mean_features_gov:
        gov_model.plot_sensitivity(X_sens, feature=f, y_scale=gov_model.y_scale)

    dev_model.plot_fit(dev_mean_X, dev_var_X, y_dev, y_scale=dev_model.y_scale)
    X_sens = pd.concat([dev_mean_X, dev_var_X[[col for col in dev_var_X.columns if col not in dev_mean_X.columns]]], axis=1)
    for f in mean_features_dev:
        dev_model.plot_sensitivity(X_sens, feature=f, y_scale=dev_model.y_scale)

    s0 = data.iloc[0:5, :]
    nu = np.random.randn(len(s0))
    a = dev_model.predict(s0, nu=nu)
    print(a)
