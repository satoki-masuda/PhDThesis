"""混合ロジットモデルの推定実装。"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
import scipy.stats as st
from tqdm import tqdm
import time
import json
import contextlib
import io
from pathlib import Path
import logging
import ray
from scipy.stats import norm, qmc


# ログ設定
logging.basicConfig(level=logging.INFO)

# Ray actor for parallel log-likelihood computation
@ray.remote
class LLWorker:
    """MXL の対数尤度計算を並列化する Ray ワーカー。"""
    def __init__(self, obs_share, relocation_years, choice_set_mask, time_steps, berry_convergence):
        self.obs_share = obs_share
        self.relocation_years = relocation_years
        self.choice_set_mask = choice_set_mask
        self.time_steps = time_steps
        self.berry_convergence = berry_convergence
    
    def softmax(self, utilities):
        return np.exp(utilities) / np.sum(np.exp(utilities), axis=1, keepdims=True)

    def predict_market_share(self, utilities, market_index):
        utilities[self.choice_set_mask == 0] = -1e10  # 選択肢集合に含まれない選択肢の効用を非常に小さな値に設定
        T = len(np.unique(market_index))
        prob = self.softmax(utilities)
        shares = np.array([
            np.sum(prob[market_index == t, :], axis=0) / np.sum(market_index == t)
            for t in range(T)
        ])
        return shares

    def berry_contraction(self, v_z):
        iteration = 0
        delta = np.log(self.obs_share) - np.log(self.obs_share[:,0][:,np.newaxis])
        while True:
            delta_v = np.array([delta[t, :] for t in self.relocation_years])
            delta_new = delta + np.log(self.obs_share) - np.log(self.predict_market_share(v_z + delta_v, self.relocation_years))
            if np.max(np.abs(delta_new - delta)) < self.berry_convergence:
                break
            if iteration >= 1000:
                break
            iteration += 1
            delta = delta_new.copy()
        return delta_v, delta

    def calc_v_z(self, X1_hetero, X1_mean, Z, XZ_mask, SAR_cov_mat, random_num, beta_z):
        _, _, L_1_hetero = X1_hetero.shape
        _, _, L_1_mean = X1_mean.shape
        K = Z.shape[1]
        SAR = np.dot(SAR_cov_mat, random_num) # (J, J) @ (J, N) -> (J, N)
        
        beta_z_hetero = beta_z[L_1_mean:(L_1_mean + int(np.sum(XZ_mask)))]
        mask = np.cumsum(XZ_mask.flatten()) * XZ_mask.flatten()
        beta_z_hetero_extend = np.array([beta_z_hetero[i-1] if i > 0 else 0 for i in mask]).reshape(K, L_1_hetero)
        beta_matrix = Z @ beta_z_hetero_extend
        v_z_hetero = np.einsum('nl,njl->nj', beta_matrix, X1_hetero)
        v_z_mean = np.einsum('nl,njl->nj', beta_z[:L_1_mean][np.newaxis,:], X1_mean)  # (N, J)
        return v_z_hetero + v_z_mean + SAR.T  # (N, J)

    def utility(self, X1_hetero, X1_mean, Z, XZ_mask, SAR_cov_mat, random_num, beta_z):
        v_z = self.calc_v_z(X1_hetero, X1_mean, Z, XZ_mask, SAR_cov_mat, random_num, beta_z)
        delta_v, delta = self.berry_contraction(v_z)
        v = delta_v + v_z
        v[self.choice_set_mask == 0] = -1e10  # 選択肢集合に含まれない選択肢の効用を非常に小さな値に設定
        return v, delta

    def loglik_chunk(self, N, beta_z, X1_hetero, X1_mean, Z, XZ_mask, SAR_cov_mat, random_draws, y, start_draw, end_draw):
        ll_sum = 0.0
        for draw in range(start_draw, end_draw):
            random_num = random_draws[:, :, draw]
            v, _ = self.utility(X1_hetero, X1_mean, Z, XZ_mask, SAR_cov_mat, random_num, beta_z)
            p = self.softmax(v)
            ll_sum += p[np.arange(N), y]
        return ll_sum


class MXL():
    """Chapter 5 の居住地選択に使う混合ロジットモデル。"""
    def __init__(self, time_steps):
        self.beta_x = None
        self.beta_z = None
        self.delta = None
        self.time_steps = time_steps
        self.n_draws = 500  # モンテカルロサンプリングの数
        self.num_cpus = 20  # 並列計算に使用するCPUコア数
        self.berry_convergence = 1e-8
        if not ray.is_initialized():
            ray.init(num_cpus=self.num_cpus, include_dashboard=False)
        self.workers = None
        
    def softmax(self, utilities):
        assert np.all(np.sum(np.exp(utilities), axis=1) > 0), f"Sum of probs must be positive.{np.sum(np.exp(utilities), axis=1)}"
        return np.exp(utilities) / np.sum(np.exp(utilities), axis=1, keepdims=True)
    
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
        #delta = np.zeros_like(self.obs_share)
        delta = np.log(self.obs_share) - np.log(self.obs_share[:,0][:,np.newaxis]) # 初期値
        while True:
            delta_v = np.array([delta[t, :] for t in self.relocation_years])
            delta_new = delta + np.log(self.obs_share) - np.log(self.predict_market_share(v_z + delta_v, self.relocation_years))
            if np.max(np.abs(delta_new - delta)) < self.berry_convergence:
                #print(iteration)
                break
            if iteration >= 1000:
                break
            iteration += 1
            delta = delta_new.copy()
        return delta_v, delta
    
    def calc_v_z(self, X1_hetero, X1_mean, Z, XZ_mask, SAR_cov_mat, random_num, beta_z):
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
        SAR = np.dot(SAR_cov_mat, random_num) # (J, J) @ (J, N) -> (J, N)
            
        beta_z_hetero = beta_z[L_1_mean:(L_1_mean + int(np.sum(XZ_mask)))]  # 個人属性ごとの異質性パラメータ
        mask = np.cumsum(XZ_mask.flatten()) * XZ_mask.flatten()
        beta_z_hetero_extend = np.array([beta_z_hetero[i-1] if i > 0 else 0 for i in mask]).reshape(K, L_1_hetero)
        beta_matrix = Z @ beta_z_hetero_extend  # (N, K) @ (K, L_1_hetero) -> (N, L_1_hetero) # 現在の居住地からの距離の定数項部分を除く
        
        v_z_hetero = np.einsum('nl,njl->nj', beta_matrix, X1_hetero) # (N, J)
        v_z_mean = np.einsum('nl,njl->nj', beta_z[:L_1_mean][np.newaxis,:], X1_mean)  # (N, J)
        
        return v_z_hetero + v_z_mean + SAR.T  # (N, J)
    
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
    
    def utility(self, X1_hetero, X1_mean, Z, XZ_mask, SAR_cov_mat, random_num, beta_z):
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
        
        v_z = self.calc_v_z(X1_hetero, X1_mean, Z, XZ_mask, SAR_cov_mat, random_num, beta_z)
        delta_v, delta = self.berry_contraction(v_z)
        v = delta_v + v_z 
        v[self.choice_set_mask == 0] = -1e10  # 選択肢集合に含まれない選択肢の効用を非常に小さな値に設定
        
        return v, delta
    
    
    def log_likelihood(self, beta_z, X1_hetero, X1_mean, Z, XZ_mask, spatial_weight_matrix, random_draws, y):
        if self.workers is None:
            raise RuntimeError("Workers are not initialized. Call fit_step1() first to set up Ray workers.")
        N = X1_hetero.shape[0]
        SAR_cov_mat = np.linalg.inv(np.eye(X1_hetero.shape[1]) - beta_z[-1] * spatial_weight_matrix)
        # chunking
        chunk_size = max(1, self.n_draws // self.num_cpus)
        tasks = []
        w = 0
        for i in range(0, self.n_draws, chunk_size):
            start_draw = i
            end_draw = min(i + chunk_size, self.n_draws)
            tasks.append(
                #workers[w % len(workers)].loglik_chunk.remote(
                self.workers[w % len(self.workers)].loglik_chunk.remote(
                    N, beta_z, X1_hetero, X1_mean, Z, XZ_mask, SAR_cov_mat, random_draws, y, start_draw, end_draw
                )
            )
            w += 1
        simulated_prob = ray.get(tasks)
        prob = np.sum(simulated_prob, axis=0)    # R 個分をまとめる
        prob /= self.n_draws                     # 平均
        LL = np.sum(np.log(prob))
        #print(beta_z)
        print(-LL)
        return -LL
    
    """
    def log_likelihood(self, beta_z, X1_hetero, X1_mean, Z, XZ_mask, spatial_weight_matrix, random_draws, y):
        N = X1_hetero.shape[0]
        SAR_cov_mat = np.linalg.inv(np.eye(X1_hetero.shape[1]) - beta_z[-1] * spatial_weight_matrix)
        ll_sum = 0.0
        for draw in range(self.n_draws):
            random_num = random_draws[:, :, draw]
            v, _ = self.utility(X1_hetero, X1_mean, Z, XZ_mask, SAR_cov_mat, random_num, beta_z)
            p = self.softmax(v)
            ll_sum += p[np.arange(N), y]
        LL = ll_sum / self.n_draws
        LL = np.sum(np.log(LL))
        print(-LL)
        return -LL
    """
    def fit_step1(self, X1_hetero, X1_mean, Z, XZ_mask, spatial_weight_matrix, y, obs_share, relocation_years):
        print("===========Estimation of the first step...===========")
        
        N = X1_hetero.shape[0]
        J = X1_hetero.shape[1]
        K = Z.shape[1]  # 個人属性の数
        initial_beta_z = np.zeros(X1_mean.shape[2] + np.sum(XZ_mask) + 1) # var_mean, var_hetero, sigma, SAR
        self.obs_share = obs_share
        self.relocation_years = relocation_years
        # Sobol sequence
        sampler = qmc.Sobol(d=J, scramble=True)
        sample = sampler.random_base2(m=int(np.ceil(np.log2(N * self.n_draws))))
        sample = (sample[:N * self.n_draws, :]).reshape(N, self.n_draws, J).transpose(2, 0, 1)  # (J, N, n_draws)
        random_draws = norm.ppf(sample)
        
        if J > 50:
            self.choice_set_mask = self.random_sample(N, J, y, choice_set_size=50)
        else:
            #self.choice_set_mask = self.random_sample(N, J, y, choice_set_size=20)
            self.choice_set_mask = np.ones((N, J))  # 全ての選択肢を選択肢集合に含める
        
        # Initialize Ray workers once (reuse during optimization)
        self.workers = [
            LLWorker.remote(self.obs_share, self.relocation_years, self.choice_set_mask, self.time_steps, self.berry_convergence)
            for _ in range(self.num_cpus)
        ]
        
        start_time = time.time()
        self.res1 = minimize(self.log_likelihood, initial_beta_z, args=(X1_hetero, X1_mean, Z, XZ_mask, spatial_weight_matrix, random_draws, y), method='BFGS')
        
        if not self.res1.success:
            print("Optimization failed: " + self.res1.message)
        if np.any(np.isnan(self.res1.x)):
            raise RuntimeError("Optimization resulted in NaN values.")
        print(f"Optimization step 1 completed in {time.time() - start_time:.2f} seconds.")
        print(self.res1.x)
        self.beta_z = self.res1.x
        
        self.delta = np.zeros_like(obs_share)
        for draw in range(self.n_draws):
            random_num = random_draws[:, :, draw]
            _, delta = self.utility(X1_hetero, X1_mean, Z, XZ_mask, spatial_weight_matrix, random_num, self.beta_z)
            self.delta += delta
        self.delta /= self.n_draws  # モンテカルロサンプリングの平均を取る
        
        if hasattr(self.res1, "hess_inv") and self.res1.hess_inv is not None:
            try:
                print("Calculating tval using the inverse Hessian from optimization result...")
                self.res1.tval = self.tval(1e-4, X1_hetero, X1_mean, Z, XZ_mask, spatial_weight_matrix, random_draws, y, hessian_inv=self.res1.hess_inv)
                print(self.res1.tval)
            except:
                print("Failed to compute t-values using the inverse Hessian from optimization result.")
        
        h = 1e-4
        print("Calculating tval with h=" + str(h))
        try:
            self.res1.tval = self.tval(h, X1_hetero, X1_mean, Z, XZ_mask, spatial_weight_matrix, random_draws, y)
            print(self.res1.tval)
        except:
            print("Failed to compute t-values using numerical Hessian with h=" + str(h))
        
        self.res1.L0 = self.initial_log_likelihood(X1_hetero, X1_mean, Z, XZ_mask, spatial_weight_matrix, random_draws, y)
        self.res1.finalLL = self.res1.fun
        self.res1.likelihood = (self.res1.L0 - self.res1.finalLL) / self.res1.L0
        self.res1.adjusted_likelihood = (self.res1.L0 - (self.res1.finalLL - len(self.beta_z))) / self.res1.L0
        base_path = Path(__file__).resolve().parent.parent  # model/ から1階層上へ
        self.save_result_step1(n_sample=X1_hetero.shape[0], filename=f"{base_path}/output/estimates/mxl_step1_results.json")
        
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
        self.save_result_step2(filename=f"{base_path}/output/estimates/mxl_step2_results.json")
        
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

        
    def predict(self, X1_hetero, X1_mean, x2, Z, XZ_mask, spatial_weight_matrix):  
        """
        x2: zonal features at the time of shape (n_alternatives, n_zonal_features)
        """
        if self.beta_z is None or self.beta_x is None:
            raise ValueError("Model is not fitted yet. Please call fit() first.")
        random_vec = np.random.standard_normal(size=(X1_hetero.shape[1], X1_hetero.shape[0]))
        SAR_cov_mat = np.linalg.inv(np.eye(X1_hetero.shape[1]) - self.beta_z[-1] * spatial_weight_matrix)
        v_z = self.calc_v_z(X1_hetero, X1_mean, Z, XZ_mask, SAR_cov_mat, random_vec, self.beta_z)
        x2_const = sm.add_constant(x2, has_constant='raise')
        zone_ASC = x2_const @ self.beta_x.T  # shape (n_alternatives)
        
        p = self.softmax(v_z + np.tile(zone_ASC, (v_z.shape[0], 1)))
        return p
    
    def initial_log_likelihood(self, X1_hetero, X1_mean, Z, XZ_mask, spatial_weight_matrix, random_draws, y):
        initial_beta = np.zeros(X1_mean.shape[2] + np.sum(XZ_mask) + 1)
        return self.log_likelihood(initial_beta, X1_hetero, X1_mean, Z, XZ_mask, spatial_weight_matrix, random_draws, y)
    
    def hessian(self, x, h, X1_hetero, X1_mean, Z, XZ_mask, spatial_weight_matrix, random_draws, y):
        n = len(x)
        H = np.zeros((n, n))
        for i in tqdm(range(n), desc="Calculating Hessian"):
            for j in range(i, n):
                e_i, e_j = np.zeros(n), np.zeros(n)
                e_i[i] = 1
                e_j[j] = 1
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    val = (
                        self.log_likelihood(x + h * e_i + h * e_j, X1_hetero, X1_mean, Z, XZ_mask, spatial_weight_matrix, random_draws, y)
                        - self.log_likelihood(x + h * e_i - h * e_j, X1_hetero, X1_mean, Z, XZ_mask, spatial_weight_matrix, random_draws, y)
                        - self.log_likelihood(x - h * e_i + h * e_j, X1_hetero, X1_mean, Z, XZ_mask, spatial_weight_matrix, random_draws, y)
                        + self.log_likelihood(x - h * e_i - h * e_j, X1_hetero, X1_mean, Z, XZ_mask, spatial_weight_matrix, random_draws, y)
                    ) / (4 * h * h)
                
                H[i, j] = val
                if i != j:
                    H[j, i] = val
                
        return H

    def tval(self, h, X1_hetero, X1_mean, Z, XZ_mask, spatial_weight_matrix, random_draws, y, hessian_inv=None):
        if hessian_inv is not None:
            return self.res1.x / np.sqrt(np.diag(hessian_inv))
        else:
            self.numeric_hessian = self.hessian(self.res1.x, h, X1_hetero, X1_mean, Z, XZ_mask, spatial_weight_matrix, random_draws, y)
            return self.res1.x / np.sqrt(np.diag(np.linalg.inv(self.numeric_hessian)))
    
    def generate_random_data(self, n_samples, n_alternatives, n_hetero_features, n_mean_features, n_zonal_features, n_personal_attributes, beta_z, XZ_mask, spatial_weight_matrix):
        X1_hetero = np.random.rand(n_samples, n_alternatives, n_hetero_features)
        X1_mean = np.random.rand(n_samples, n_alternatives, n_mean_features)  # 平均特徴量
        X2 = np.random.rand(n_alternatives * self.time_steps, n_zonal_features)
        Z = np.random.rand(n_samples, n_personal_attributes)
        obs_share = np.random.dirichlet(np.ones(n_alternatives), size=self.time_steps)
        self.obs_share = obs_share
        relocation_years = np.random.randint(0, self.time_steps, size=n_samples)
        self.relocation_years = relocation_years
        self.choice_set_mask = np.ones((n_samples, n_alternatives))  # 全ての選択肢を選択肢集合に含める
        random_num = np.random.standard_normal(size=(n_alternatives, n_samples)) 
        v, _ = self.utility(X1_hetero, X1_mean, Z, XZ_mask, spatial_weight_matrix, random_num, beta_z)
        p = self.softmax(v)
        y = np.array([np.random.choice(n_alternatives, p=p[i]) for i in range(n_samples)])
        return X1_hetero, X1_mean, X2, Z, y, obs_share, relocation_years
    
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
    beta_z = np.random.rand(L_1_mean + K * L_1_hetero + L_1_hetero * (L_1_hetero + 1) // 2 + L_1_mean * (L_1_mean + 1) // 2)  # Randomly initialize beta_z
    XZ_mask = np.ones((K, L_1_hetero), dtype=int)  # 個人属性ごとの異質性を考慮するかどうかのマスク
    spatial_weight_matrix = np.random.rand(J, J)
    
    mxl = MXL(time_steps=T)
    X1_hetero, X1_mean, X2, Z, y, obs_share, relocation_years = mxl.generate_random_data(n_samples, J, L_1_hetero, L_1_mean, L_2, K, beta_z, XZ_mask, spatial_weight_matrix)
    # Fit the model
    
    res = mxl.fit_step1(X1_hetero, X1_mean, Z, XZ_mask, spatial_weight_matrix, y, obs_share, relocation_years)
    print("Estimated Beta:", res.x)
    print("T-values:", res.tval)
    print("Likelihood Ratio:", res.likelihood)
    print("Adjusted Likelihood Ratio:", res.adjusted_likelihood)
    
    with open("output/estimates/mxl_step1_results.json", "r") as f:
        params = json.load(f)
    params = pd.DataFrame(params)
    mxl.beta_z = params["beta_z"].values
    mxl.delta = params["delta"].values
    
    mxl.fit_step2(X2)
    
    # Predict probabilities
    probabilities = mxl.predict(X1_hetero, X1_mean, X2[:J, :], Z, XZ_mask, spatial_weight_matrix)
    print("Predicted Probabilities:", probabilities)
