"""多項ロジットモデルの推定実装。"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from statsmodels.sandbox.regression.gmm import IV2SLS
import scipy.stats as st
from tqdm import tqdm
import time
import json
import contextlib
import io
import logging
from pathlib import Path

# ログ設定
logging.basicConfig(level=logging.INFO)

class MNL:
    """標準的な多項ロジットモデル。"""
    def __init__(self):
        self.beta = None

    @staticmethod
    def softmax(utilities):
        assert np.all(np.sum(np.exp(utilities), axis=1) > 0), f"Sum of probs must be positive."
        return np.exp(utilities) / np.sum(np.exp(utilities), axis=1, keepdims=True)
    
    @staticmethod
    def utility(X, beta):
        """
        Compute the utility for each alternative given the features and parameters.
        
        Parameters:
            X (ndarray): Feature array of shape (n_samples, n_alternatives, n_features)
            beta (ndarray): Parameter array of shape (n_alternatives, n_features)
        
        Returns:
            v (ndarray): Utility array of shape (n_samples, n_alternatives)
        """
        # ベクトル化計算
        return np.einsum('ijk,k->ij', X, beta)
    
    @staticmethod
    def log_likelihood(beta, X, y):
        n_samples, _, _ = X.shape
        v = MNL.utility(X, beta)
        p = MNL.softmax(v)
        LL = np.sum(np.log(p[np.arange(n_samples), y]))
        return -LL
    
    def fit(self, X, y, asc_cols=None):
        logging.info("Fitting MNL model...")
        
        # 入力検証
        if X.size == 0 or y.size == 0:
            raise ValueError("Empty input data")
        if X.shape[0] != len(y):
            raise ValueError("X and y must have the same number of samples")
        
        initial_beta = np.zeros(X.shape[2])
        self.res = minimize(MNL.log_likelihood, initial_beta, args=(X, y), method='BFGS')
        if not self.res.success:
            logging.warning(f"Optimization failed: {self.res.message}")
            
        if np.any(np.isnan(self.res.x)):
            raise RuntimeError("Optimization resulted in NaN values.")
            
        self.beta = self.res.x
        self.res.tval = self.tval(1e-4, X, y)
        self.L0 = self.initial_log_likelihood(X, y, asc_cols=asc_cols)
        self.finalLL = self.res.fun
        self.res.likelihood = (self.L0 - self.finalLL) / self.L0
        self.res.adjusted_likelihood = (self.L0 - (self.finalLL - len(self.beta))) / self.L0
        
        # 弾性値の効率的計算
        self.mean_elasticity = self._compute_elasticity_efficient(X)
        base_path = Path(__file__).resolve().parent.parent  # model/ から1階層上へ
        self.save_result_json(n_sample=X.shape[0], filename=f"{base_path}/output/estimates/mnl_results.json")
        
        return self.res
    
    def _compute_elasticity_efficient(self, X):
        """弾性値の効率的計算"""
        p = self.predict(X)
        mean_elasticity = np.zeros(len(self.beta))
        
        # ベクトル化された計算
        for k in range(len(self.beta)):
            elasticity = p * X[:,:,k] * self.beta[k]
            mean_elasticity[k] = np.mean(np.sum(p * elasticity, axis=0) / np.sum(p, axis=0))
            
        return mean_elasticity
    
    def predict(self, X):
        if self.beta is None:
            raise ValueError("Model is not fitted yet. Please call fit() first.")
        v = MNL.utility(X, self.beta)
        p = MNL.softmax(v)
        return p
    
    def initial_log_likelihood(self, X, y, asc_cols: list = None):
        if asc_cols:
            initial_beta = np.zeros(len(asc_cols))
            res = minimize(MNL.log_likelihood, initial_beta, args=(X[:,:,asc_cols], y), method='BFGS')
            return res.fun
        else:
            initial_beta = np.zeros(X.shape[2])
            return MNL.log_likelihood(initial_beta, X, y)
    
    def hessian(self, h, X, y):
        x = self.res.x
        n = len(x)
        res = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                e_i, e_j = np.zeros(n), np.zeros(n)
                e_i[i] = 1
                e_j[j] = 1

                # mfは符号が逆転しているので数値微分も符号逆にする
                res[i][j] = (
                            - MNL.log_likelihood(x + h * e_i + h * e_j, X, y)
                            + MNL.log_likelihood(x + h * e_i - h * e_j, X, y)
                            + MNL.log_likelihood(x - h * e_i + h * e_j, X, y)
                            - MNL.log_likelihood(x - h * e_i - h * e_j, X, y)
                            ) / (4 * h * h)
        return res

    def tval(self, h, X, y):
        #print(self.hessian(h))
        return self.res.x / np.sqrt(-np.diag(np.linalg.inv(self.hessian(h, X, y))))
    
    def generate_random_data(self, n_samples, n_alternatives, n_features, beta):
        assert len(beta) == n_features, "Beta length must match the number of features."
        X = np.random.rand(n_samples, n_alternatives, n_features)
        v = MNL.utility(X, beta)
        p = MNL.softmax(v)
        y = np.array([np.random.choice(n_alternatives, p=p[i]) for i in range(n_samples)])
        return X, y
    
    def save_result_json(self, n_sample, filename):
        if self.res is None:
            raise ValueError("Model is not fitted yet. Please call fit() first.")
        # 推定結果をDataFrameにまとめる
        n_params = len(self.res.x)
        result = {
            'beta': self.res.x.tolist() if hasattr(self.res.x, 'tolist') else list(self.res.x),
            'tval': self.res.tval.tolist() if hasattr(self.res.tval, 'tolist') else list(self.res.tval),
            'elasticity': self.mean_elasticity.tolist() if hasattr(self.mean_elasticity, 'tolist') else list(self.mean_elasticity),
            'sample_size': n_sample,  # サンプルサイズ
            'initial_log_likelihood': self.L0,  # 初期対数尤度
            'final_log_likelihood': self.finalLL,  # 最終対数尤度
            'likelihood': self.res.likelihood,  # 尤度比
            'adjusted_likelihood': self.res.adjusted_likelihood  # 修正済み尤度比
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {filename}")


class MNL_2step():
    """Berry contraction と IV 推定を組み合わせた 2 段階推定版 MNL。"""
    def __init__(self, time_steps):
        self.beta_x = None
        self.beta_z = None
        self.delta = None
        self.berry_convergence = 1e-8
        self.time_steps = time_steps
        
    def softmax(self, utilities):
        assert np.all(np.sum(np.exp(utilities), axis=1) > 0), f"Sum of probs must be positive.{utilities}"
        return np.exp(utilities) / np.sum(np.exp(utilities), axis=1, keepdims=True)
    
    #@staticmethod
    def predict_market_share(self, utilities, market_index):
        """
        Predict market share for each alternative in each market given the utilities.
        
        Parameters:
            utilities (ndarray): Utility array of shape (n_samples, n_alternatives)
        
        Returns:
            shares (ndarray): Market share array of shape (n_markets, n_alternatives)
        """
        utilities[self.choice_set_mask == 0] = -1e10  # 選択肢集合に含まれない選択肢の効用を非常に小さな値に設定
        T = len(np.unique(market_index))
        prob = self.softmax(utilities)
        shares = np.array([
            np.sum(prob[market_index == t, :], axis=0) / np.sum(market_index == t)
            for t in range(T)
        ]) # (T, J)
        
        return shares
    
    #@staticmethod
    def berry_contraction(self, v_z):
        """
        Apply Berry contraction.
        
        Parameters:
            v_z (ndarray): Utility vector of shape (n_samples, n_alternatives)
            obs_share (ndarray): Observed shares of alternatives of shape (time_steps, n_alternatives)
            relocation_years (ndarray): Relocation years of shape (n_samples,)
        Returns:
            delta_v (ndarray): Alternative-specific vector of shape (n_samples, n_alternatives)
            delta (ndarray): zone-year specific constant of shape (time_steps, n_alternatives)
        """
        iteration = 0
        delta = np.log(self.obs_share) - np.log(self.obs_share[:,0][:,np.newaxis]) # 初期値
        while True:
            delta_v = np.array([delta[t, :] for t in self.relocation_years])
            delta_new = delta + np.log(self.obs_share) - np.log(self.predict_market_share(v_z + delta_v, self.relocation_years))
            if np.max(np.abs(delta_new - delta)) < self.berry_convergence:
                #print(iteration)
                break
            if iteration >= 1000:
                print("Berry contraction failed to converge.")
                break
            iteration += 1
            delta = delta_new.copy()
        return delta_v, delta
    
    #@staticmethod
    def calc_v_z(self, X1_hetero, X1_mean, Z, XZ_mask, beta_z):
        """
        個人属性Zから個人ごとの異質性を計算
        以下を計算している
        v = np.zeros((N, J))
        for n in range(N):
            for j in range(J):
                for l in range(L_1_hetero):
                    v[n, j] += beta_matrix[n, l] * X1_hetero[n, j, l]
        """
        _, _, L_1_hetero = X1_hetero.shape
        _, _, L_1_mean = X1_mean.shape
        K = Z.shape[1]  # 個人属性の数
        beta_z_hetero = beta_z[L_1_mean:]
        mask = np.cumsum(XZ_mask.flatten()) * XZ_mask.flatten()
        beta_z_hetero_extend = np.array([beta_z_hetero[i-1] if i > 0 else 0 for i in mask]).reshape(K, L_1_hetero)
        beta_matrix = Z @ beta_z_hetero_extend  # (N, K) @ (K, L_1_hetero) -> (N, L_1_hetero) # 現在の居住地からの距離の定数項部分を除く
        v_z_hetero = np.einsum('nl,njl->nj', beta_matrix, X1_hetero) # (N, J)
        v_z_mean = np.einsum('l,njl->nj', beta_z[:L_1_mean], X1_mean)  # (N, J)
        
        return v_z_hetero + v_z_mean
    
    #@staticmethod
    def random_sample(self, n_sample, n_alternative, y, choice_set_size):
        """
        Random sample for choice set generation.
        choice set = chosen alternative + random sample
        """
        N, J = n_sample, n_alternative
        # 各行について選択された列とランダムに選ばれた列のみ値を残し、それ以外は0
        choice_set_mask = np.zeros((N, J), dtype=int)
        choice_set_mask[np.arange(N), y] = 1

        # 各サンプルごとに、選択肢集合から選択されたもの以外からランダムに choice_set_size-1 個選ぶ
        all_choices = np.arange(J)
        mask = np.ones((N, J), dtype=bool)
        mask[np.arange(N), y] = 0
        available_choices = [all_choices[mask[i]] for i in range(N)]
        random_choices = np.array([
            np.random.choice(choices, size=choice_set_size-1, replace=False)
            for choices in available_choices
        ])
        # v_choice_setにランダム選択肢の値をセット
        for i in range(N):
            choice_set_mask[i, random_choices[i]] = 1
        return choice_set_mask
    
    #@staticmethod
    def utility(self, X1_hetero, X1_mean, Z, XZ_mask, y, beta_z):
        """
        Compute the utility for each alternative given the features and parameters.
        
        Parameters:
            X1_hetero (ndarray): Feature array of shape (n_samples, n_alternatives, n_features)
            X1_mean (ndarray): Mean feature array of shape (n_samples, n_alternatives, n_features)
            Z (ndarray): Personal attribute array of shape (n_samples, n_personal_attributes)
            beta_z (ndarray): Parameter array of shape (n_personal_attributes * n_features)
            obs_share (ndarray): Observed shares of alternatives of shape (n_samples, n_alternatives)
        Returns:
            v (ndarray): Utility array of shape (n_samples, n_alternatives)
            delta (ndarray): zone-year specific constant of shape (time_steps, n_alternatives)
        """
        
        v_z = self.calc_v_z(X1_hetero, X1_mean, Z, XZ_mask, beta_z)
        delta_v, delta = self.berry_contraction(v_z)
        v = delta_v + v_z
        v[self.choice_set_mask == 0] = -1e10  # 選択肢集合に含まれない選択肢の効用を非常に小さな値に設定
        
        return v, delta
    
    #@staticmethod
    def log_likelihood(self, beta_z, X1_hetero, X1_mean, Z, XZ_mask, y):
        N  = X1_hetero.shape[0]
        v, delta = self.utility(X1_hetero, X1_mean, Z, XZ_mask, y, beta_z)
        #self.delta = delta  # Save delta for later use
        p = self.softmax(v)
        LL = np.sum(np.log(p[np.arange(N), y]))
        print(-LL)
        return -LL
    
    def fit_step1(self, X1_hetero, X1_mean, Z, XZ_mask, y, obs_share, relocation_years):
        print("===========Estimation of the first step...===========")
        
        N = X1_hetero.shape[0]
        J = X1_hetero.shape[1]
        L_1_hetero = X1_hetero.shape[2]  # 異質性特徴量の数
        L_1_mean = X1_mean.shape[2]  # 平均特徴量の数
        K = Z.shape[1]  # 個人属性の数
        initial_beta_z = np.zeros(L_1_mean + np.sum(XZ_mask)) # 現在の居住地からの距離は定数項部分も第一段階で推定
        self.obs_share = obs_share
        self.relocation_years = relocation_years
        #self.delta = np.zeros((self.time_steps, J))  # zone-year specific constant
        if J > 50:
            self.choice_set_mask = self.random_sample(N, J, y, choice_set_size=50)
        else:
            #self.choice_set_mask = self.random_sample(N, J, y, choice_set_size=20)
            self.choice_set_mask = np.ones((N, J))  # 全ての選択肢を選択肢集合に含める
        
        start_time = time.time()
        #self.res1 = minimize(self.log_likelihood, initial_beta_z, args=(X1_hetero, X1_mean, Z, XZ_mask, y), method='Nelder-Mead')#, options={'maxiter': 10000})
        self.res1 = minimize(self.log_likelihood, initial_beta_z, args=(X1_hetero, X1_mean, Z, XZ_mask, y), method='BFGS')


        if not self.res1.success:
            #raise RuntimeError("Optimization failed: " + self.res1.message)
            print("Optimization failed: " + self.res1.message)
        if np.any(np.isnan(self.res1.x)):
            raise RuntimeError("Optimization resulted in NaN values.")
        print(f"Optimization step 1 completed in {time.time() - start_time:.2f} seconds.")
        
        self.beta_z = self.res1.x
        _, self.delta = self.utility(X1_hetero, X1_mean, Z, XZ_mask, y, self.beta_z)
        self.res1.tval = self.tval(1e-4, X1_hetero, X1_mean, Z, XZ_mask, y)
        #self.res.tval = np.zeros_like(self.res1.x)  # t値は後で計算する
        self.res1.L0 = self.initial_log_likelihood(X1_hetero, X1_mean, Z, XZ_mask, y)
        self.res1.finalLL = self.res1.fun
        self.res1.likelihood = (self.res1.L0 - self.res1.finalLL) / self.res1.L0
        self.res1.adjusted_likelihood = (self.res1.L0 - (self.res1.finalLL - len(self.beta_z))) / self.res1.L0
        
        base_path = Path(__file__).resolve().parent.parent  # model/ から1階層上へ
        self.save_result_step1(n_sample=X1_hetero.shape[0], filename=f"{base_path}/output/estimates/mnl_step1_results.json")
        
        return self.res1
    
    def fit_step2(self, X2, W=None, method='OLS'):
        """
        Args:
            X2 (ndarray): Zonal features of shape (time_step * n_alternatives, n_zonal_features)
        """
        print("===========Estimation of the second step...===========")
        if self.beta_z is None:
            raise ValueError("Step 1 must be fitted first. Please call fit_step1() first.")
        
        # 独立変数X2と回帰
        if method == 'OLS':
            model_lr = sm.OLS(self.delta.flatten(), sm.add_constant(X2))
            self.res2 = model_lr.fit()
            print(self.res2.summary())
            self.beta_x = self.res2.params
        
        elif method == 'IV':
            if W is None:
                raise ValueError("Instrumental variables W must be provided for IV estimation.")
            X2_const = sm.add_constant(X2)
            model_lr = sm.OLS(self.delta.flatten(), X2_const)
            res_lr = model_lr.fit()
                            
            # Instrumental Variable Regression
            Z_iv = sm.add_constant(np.hstack((X2[:,:-1], W)))  # Wを除くX2の全ての列とWを結合
            model_iv = IV2SLS(endog=self.delta.flatten(), exog=X2_const, instrument=Z_iv)
            self.res2 = model_iv.fit()
            print(self.res2.summary())
            self.beta_x = self.res2.params
            
            hausman_stat, p_value = self.hausman_test(res_lr, self.res2)

            print(f'Hausman検定統計量: {hausman_stat}')
            print(f'p値: {p_value}')
            # p値に基づく判定
            if p_value < 0.05:
                print("帰無仮説を棄却: OLS推定量はバイアスがあり、2SLSを使用すべきです。")
            else:
                print("帰無仮説を採択: OLS推定量にバイアスはなく、OLSを使用することが適切です。")
        
        base_path = Path(__file__).resolve().parent.parent  # model/ から1階層上へ
        self.save_result_step2(filename=f"{base_path}/output/estimates/mnl_step2_results.json")
        
        return self.res2
    
    # 2段階最小二乗法（2SLS）とOLSの差を比較するための関数を定義
    def hausman_test(self, ols_results, iv_results):
        """
        ols_results: OLS推定結果
        iv_results:  2SLS推定結果
        """
        # OLSとIVの推定量の差を計算
        b_ols = ols_results.params
        b_iv = iv_results.params
        diff = b_iv - b_ols
        # OLSの推定量の共分散行列
        cov_ols = ols_results.cov_params()
        # IVの推定量の共分散行列
        cov_iv = iv_results.cov_params()
        # 差の共分散行列を計算
        cov_diff = cov_iv - cov_ols
        # Hausman検定の統計量を計算
        stat = np.dot(np.dot(diff.T, np.linalg.inv(cov_diff)), diff)
        # 自由度
        df = len(b_ols) - 1
        # p値を計算
        p_value = st.chi2.sf(stat, df)
        return stat, p_value

        
    def predict(self, X1_hetero, X1_mean, x2, Z, XZ_mask):
        """
        x2: zonal features at the time of shape (n_alternatives, n_zonal_features)
        """
        if self.beta_z is None or self.beta_x is None:
            raise ValueError("Model is not fitted yet. Please call fit() first.")
        v_z = self.calc_v_z(X1_hetero, X1_mean, Z, XZ_mask, self.beta_z)
        x2_const = sm.add_constant(x2, has_constant='raise')
        zone_ASC = x2_const @ self.beta_x.T  # shape (n_alternatives)
        
        p = self.softmax(v_z + np.tile(zone_ASC, (v_z.shape[0], 1)))
        return p
    
    def initial_log_likelihood(self, X1_hetero, X1_mean, Z, XZ_mask, y):
        initial_beta = np.zeros(X1_mean.shape[2] + np.sum(XZ_mask)) # 現在の居住地からの距離は定数項部分も第一段階で推定
        return self.log_likelihood(initial_beta, X1_hetero, X1_mean, Z, XZ_mask, y)
    
    def hessian(self, h, X1_hetero, X1_mean, Z, XZ_mask, y):
        x = self.res1.x
        n = len(x)
        res = np.zeros((n, n))
        for i in tqdm(range(n), desc="Calculating Hessian"):
            for j in range(i, n):
                e_i, e_j = np.zeros(n), np.zeros(n)
                e_i[i] = 1
                e_j[j] = 1
                # mfは符号が逆転しているので数値微分も符号逆にする
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    val = (
                        - self.log_likelihood(x + h * e_i + h * e_j, X1_hetero, X1_mean, Z, XZ_mask, y)
                        + self.log_likelihood(x + h * e_i - h * e_j, X1_hetero, X1_mean, Z, XZ_mask, y)
                        + self.log_likelihood(x - h * e_i + h * e_j, X1_hetero, X1_mean, Z, XZ_mask, y)
                        - self.log_likelihood(x - h * e_i - h * e_j, X1_hetero, X1_mean, Z, XZ_mask, y)
                    ) / (4 * h * h)
                
                res[i][j] = val
                if i != j:
                    res[j][i] = val
                
        return res

    def tval(self, h, X1_hetero, X1_mean, Z, XZ_mask, y):
        #print(self.hessian(h))
        return self.res1.x / np.sqrt(-np.diag(np.linalg.inv(self.hessian(h, X1_hetero, X1_mean, Z, XZ_mask, y))))
    
    def generate_random_data(self, n_samples, n_alternatives, n_hetero_features, n_mean_features, n_zonal_features, n_personal_attributes, beta_z, XZ_mask,):
        X1_hetero = np.random.rand(n_samples, n_alternatives, n_hetero_features)
        X1_mean = np.random.rand(n_samples, n_alternatives, n_mean_features)  # 平均特徴量
        X2 = np.random.rand(n_alternatives * self.time_steps, n_zonal_features)
        Z = np.random.rand(n_samples, n_personal_attributes)
        obs_share = np.random.dirichlet(np.ones(n_alternatives), size=self.time_steps)
        v, _ = self.utility(X1_hetero, X1_mean, Z, XZ_mask, y, beta_z)
        p = self.softmax(v)
        y = np.array([np.random.choice(n_alternatives, p=p[i]) for i in range(n_samples)])
        return X1_hetero, X1_mean, X2, Z, y, obs_share
    
    def save_result_step1(self, n_sample, filename):
        if self.res1 is None:
            raise ValueError("Model is not fitted yet. Please call fit() first.")
        # 推定結果をjson形式で保存する
        result = {
            'beta_z': self.beta_z.tolist() if hasattr(self.beta_z, 'tolist') else list(self.beta_z),
            'tval': self.res1.tval.tolist() if hasattr(self.res1.tval, 'tolist') else list(self.res1.tval),
            'sample_size': n_sample,
            'initial_log_likelihood': self.res1.L0,
            'final_log_likelihood': self.res1.finalLL,
            'rho2': self.res1.likelihood,
            'adjusted_rho2': self.res1.adjusted_likelihood
        }
        if self.delta is not None:
            result["delta"] = self.delta.tolist() if hasattr(self.delta, 'tolist') else list(self.delta)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {filename}")
    
    def save_result_step2(self, filename):
        if self.res2 is None:
            raise ValueError("Model is not fitted yet. Please call fit() first.")
        # 推定結果をjson形式で保存する
        result = {
            'beta_x': self.beta_x.tolist() if hasattr(self.beta_x, 'tolist') else list(self.beta_x),
            'tval': self.res2.tvalues.tolist() if hasattr(self.res2.tvalues, 'tolist') else list(self.res2.tvalues),
            'pvalue': self.res2.pvalues.tolist() if hasattr(self.res2.pvalues, 'tolist') else list(self.res2.pvalues),
            'standard_error': self.res2.bse.tolist() if hasattr(self.res2.bse, 'tolist') else list(self.res2.bse),
            'sample_size': self.res2.nobs,
        }
        
        # AICとBICの計算を試行
        try:
            if hasattr(self.res2, 'aic'):
                result['AIC'] = self.res2.aic
        except (NotImplementedError, AttributeError):
            pass
        
        try:
            if hasattr(self.res2, 'bic'):
                result['BIC'] = self.res2.bic
        except (NotImplementedError, AttributeError):
            pass
        
        # F統計量の計算を試行
        try:
            if hasattr(self.res2, 'fvalue'):
                result['F-statistic'] = self.res2.fvalue
        except (NotImplementedError, AttributeError):
            pass
        
        # MSEの計算を試行
        try:
            if hasattr(self.res2, 'mse_model'):
                result['MSE'] = self.res2.mse_model
        except (NotImplementedError, AttributeError):
            pass
        
        # R-squaredの計算を試行
        try:
            if hasattr(self.res2, 'rsquared'):
                result['R-squared'] = self.res2.rsquared
        except (NotImplementedError, AttributeError):
            pass
        
        # Adjusted R-squaredの計算を試行
        try:
            if hasattr(self.res2, 'rsquared_adj'):
                result['Adjusted R-squared'] = self.res2.rsquared_adj
        except (NotImplementedError, AttributeError):
            pass
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {filename}")
    
    def load_step1(self, filename):
        """
        Load parameters from a JSON file.
        
        Parameters:
            filename (str): Path to the JSON file containing the parameters.
        """
        if filename is not None:
            with open(filename, "r") as f:
                params = json.load(f)
            self.beta_z = np.array(params["beta_z"])
            self.delta = np.array(params["delta"])
        
    def load_step2(self, filename):
        """
        Load parameters from a JSON file.
        
        Parameters:
            filename (str): Path to the JSON file containing the parameters.
        """
        if filename is not None:
            with open(filename, "r") as f:
                params = json.load(f)
            self.beta_x = np.array(params["beta_x"])
    
if __name__ == "__main__":
    # Example usage
    n_people = 200
    J = 20 # Number of alternatives
    L_1_hetero = 7  # Number of zonal features in X1_hetero
    L_1_mean = 1  # Number of zonal features in X1_mean
    L_2 = 8  # Number of zonal features in X2
    K = 3  # Number of personal attributes in Z
    T = 5 # Number of time steps
    n_samples = n_people * T
    beta_z = np.random.rand(L_1_mean + K * L_1_hetero)  # 個人属性ごとの異質性パラメータと分散共分散行列パラメータ
    relocation_years = np.random.randint(0, T, size=n_samples)
    XZ_mask = np.ones((K, L_1_hetero))  # 個人属性ごとの異質性を考慮するかどうかのマスク
    
    mnl = MNL_2step(time_steps=T)
    X1_hetero, X1_mean, X2, Z, y, obs_share = mnl.generate_random_data(n_samples, J, L_1_hetero, L_1_mean, L_2, K, beta_z, XZ_mask)
    # Fit the model
    res = mnl.fit_step1(X1_hetero, X1_mean, Z, y)
    print("Estimated Beta:", res.x)
    print("T-values:", res.tval)
    print("Likelihood Ratio:", res.likelihood)
    print("Adjusted Likelihood Ratio:", res.adjusted_likelihood)
    
    with open("output/estimates/mnl_step1_results.json", "r") as f:
        params = json.load(f)
    params = pd.DataFrame(params)
    mnl.beta_z = params["beta"].values
    mnl.delta = params["delta"].values
    
    mnl.fit_step2(X2)
    
    # Predict probabilities
    probabilities = mnl.predict(X1_hetero, X1_mean, X2, Z, relocation_years)
    print("Predicted Probabilities:", probabilities)
