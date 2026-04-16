"""BBL 型の構造推定で使う目的関数。"""

import numpy as np

EPS = 1e-8

def Q(theta, W, W_perturb):
    """観測方策と摂動方策の利得差から BBL の損失を計算する。"""
    theta = np.concatenate(([1.0], theta))
    gap = np.dot(W, theta.reshape(-1,1)) - np.einsum('ijk,j->ik', W_perturb, theta) # (n_sample,) - (n_sample, n_pertub)
    #sd = np.std(gap)
    #scale = sd if sd > EPS else 1.0
    scale = 1.0
    loss = np.sum(np.minimum(gap / scale, 0) ** 2)
    
    return loss / W.shape[0]  # サンプル数で割る
