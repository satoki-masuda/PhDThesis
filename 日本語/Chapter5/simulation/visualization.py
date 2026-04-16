"""シミュレーション結果と観測データを可視化する関数群。"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def _ensure_1d(x):
    if x is None:
        return None
    arr = np.asarray(x)
    return float(np.squeeze(arr))

def _years_from_start(start_year: int, length: int):
    return np.arange(start_year, start_year + length)

def _get_series(paths: dict, key: str, zone: int):
    """
    paths[key]: shape (T, Z) の配列を想定
    zone は列インデックス（int）
    """
    arr = np.asarray(paths[key])
    if arr.ndim != 2:
        raise ValueError(f"{key} must be a 2D array (T, Z). Got shape {arr.shape}.")
    if not (0 <= zone < arr.shape[1]):
        raise IndexError(f"zone index {zone} is out of bounds for {key} with Z={arr.shape[1]}.")
    return arr[:, zone]

def plot_simulation_paths(
    paths: dict,
    target_zone: int,
    start_year: int,
    init_paths: dict | None = None,
    data: pd.DataFrame | None = None,
    zone_col: str = "zone",
    year_col: str = "year",
    observed_cols: dict | None = None,
    savepath_prefix: str | None = None,
):
    """
    時系列パス（population, investment, development, price）と価値 Vg/Vd を描画。

    Parameters
    ----------
    paths : dict
        {"population_path": (T,Z), "investment_path": (T,Z),
         "development_path": (T,Z), "price_path": (T,Z)}
    target_zone : int
        描画対象ゾーン（列インデックス）
    start_year : int
        シミュレーション開始年（x軸に使用）
    Vg, Vd : float or array-like
        政府・デベロッパーの割引利得（スカラー想定）。配列でもOK。
    init_paths : dict, optional
        観測方策での初期パス（同じキー構成・同じ形状）。
        渡した場合は同じプロットに破線で重ねる。
    data : pd.DataFrame, optional
        観測データをプロットしたい場合に渡す。`observed_cols` で列名を指定。
        例: {"investment":"los_t_raw", "development":"develop_res_t_raw", "price":"land_price_t", "population":"population_t"}
    zone_col, year_col : str
        `data` 内のゾーン・年の列名。
    observed_cols : dict, optional
        観測の列名マップ。上の例参照。指定が無ければ観測は描かない。
    savepath_prefix : str, optional
        例: "output/figs/market_z1_2010" を渡すと "{prefix}_investment.png" などに保存。
        指定しなければ表示のみ（plt.show()）。
    """
    # --- 取り出し ---
    series_keys = [
        ("investment_path", "Investment (gov)"),
        ("development_path", "Development (dev)"),
        ("price_path", "Land Price"),
        ("population_path", "Population"),
    ]
    T = np.asarray(paths["investment_path"]).shape[0]
    years = _years_from_start(start_year, T)

    # --- 観測データの準備（オプション） ---
    obs = {}
    if data is not None and observed_cols:
        df_zone = data.loc[data[zone_col] == target_zone]
        if not df_zone.empty:
            for logical_name, col in observed_cols.items():
                if col in df_zone.columns:
                    s = df_zone[[year_col, col]].dropna()
                    # 同じ年の範囲に合わせて並べ替え（存在すればプロット）
                    obs[logical_name] = (s[year_col].to_numpy(), s[col].to_numpy())

    # --- 図の作成（1図1系列） ---
    for key, title in series_keys:
        y = _get_series(paths, key, target_zone)
        fig, ax = plt.subplots()
        ax.plot(years, y, label="Equilibrium")
        # 初期パスを重ね書き（破線）
        if init_paths is not None and key in init_paths:
            y0 = _get_series(init_paths, key, target_zone)
            ax.plot(years, y0, linestyle="--", label="Init path")
        # 観測を重ね書き
        if key == "investment_path" and "investment" in obs:
            ax.plot(obs["investment"][0], obs["investment"][1], linestyle=":", label="Observed")
        if key == "development_path" and "development" in obs:
            ax.plot(obs["development"][0], obs["development"][1], linestyle=":", label="Observed")
        if key == "price_path" and "price" in obs:
            ax.plot(obs["price"][0], obs["price"][1], linestyle=":", label="Observed")
        if key == "population_path" and "population" in obs:
            ax.plot(obs["population"][0], obs["population"][1], linestyle=":", label="Observed")

        MAE = None
        MAPE = None
        key_map = {
            "investment_path": "investment",
            "development_path": "development",
            "price_path": "price",
            "population_path": "population",
        }
        obs_key = key_map.get(key)
        if obs_key is not None and obs_key in obs:
            obs_years, obs_values = obs[obs_key]
            # モデルの年と観測の年は基本的に一致しているので、単純に合致する年だけ値を取り出す
            y_at_obs = np.full_like(obs_values, np.nan, dtype=float)
            for i, yy in enumerate(obs_years):
                if yy in years:
                    idx = np.where(years == yy)[0][0]
                    y_at_obs[i] = y[idx]
            errors = y_at_obs - obs_values
            MAE = np.mean(np.abs(errors))
            # MAPE（観測値が0の場合は除外）
            nonzero_idx = obs_values != 0
            if np.any(nonzero_idx):
                MAPE = np.mean(np.abs(errors[nonzero_idx] / obs_values[nonzero_idx])) * 100
            else:
                MAPE = None
            # Annotation表示
            mae_text = f"MAE={MAE:.2f}" if MAE is not None else "MAE=N/A"
            mape_text = f"MAPE={MAPE:.1f}%" if MAPE is not None else "MAPE=N/A"
            ax.text(0.98, 0.02, f"{mae_text}\n{mape_text}", transform=ax.transAxes,
                    va="bottom", ha="right", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
        
        ax.set_xlabel("Year")
        ax.set_ylabel(title)
        ax.set_title(f"{title} | zone={target_zone}")
        ax.legend(loc="best")
        #ax.grid(True, which="both", linestyle=":")
        fig.tight_layout()
        if savepath_prefix:
            if not os.path.exists(os.path.dirname(savepath_prefix)):
                os.makedirs(os.path.dirname(savepath_prefix), exist_ok=True)
            fig.savefig(f"{savepath_prefix}_{key.replace('_path','')}.png", dpi=300)
            plt.close(fig)
        else:
            plt.show()

def _years_from_start(start_year: int, length: int):
    return np.arange(start_year, start_year + length)

def _to_2d(a, name):
    arr = np.asarray(a)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D (T, Z). Got {arr.shape}.")
    return arr

def plot_all_zones_lines(
    paths: dict,
    start_year: int,
    zone_labels: list | None = None,
    var_keys: list[str] | None = None,
    legend_cols: int = 4,
    savepath_prefix: str | None = None,
):
    """
    均衡パスの全ゾーン・年次推移をスパゲッティ図で描画（凡例あり）。

    Parameters
    ----------
    paths : dict
        {"population_path": (T,Z), "investment_path": (T,Z),
         "development_path": (T,Z), "price_path": (T,Z)}
    start_year : int
        x軸の開始年
    zone_labels : list | None
        ゾーンの凡例ラベル（Z 個）。None なら 0..Z-1。
    var_keys : list[str] | None
        描くキー。None なら主要4変数。
    legend_cols : int
        凡例の列数（多ゾーンで横に畳む）
    savepath_prefix : str | None
        指定すれば "{prefix}_{name}_lines.png" に保存
    """
    title_map = {
        "investment_path": "Investment (gov)",
        "development_path": "Development (dev)",
        "price_path": "Land Price",
        "population_path": "Population",
    }
    if var_keys is None:
        var_keys = ["investment_path","development_path","price_path","population_path"]

    for key in var_keys:
        A = _to_2d(paths[key], key)   # (T, Z)
        T, Z = A.shape
        years = _years_from_start(start_year, T)
        labels = zone_labels if (zone_labels is not None and len(zone_labels)==Z) else list(range(Z))

        fig, ax = plt.subplots()
        for z in range(Z):
            ax.plot(years, A[:, z], label=str(labels[z]))
        ax.set_xlabel("Year")
        ax.set_ylabel(title_map.get(key, key))
        ax.set_title(f"{title_map.get(key, key)} — all zones (Z={Z})")
        #ax.grid(True, which="both", linestyle=":")

        # 凡例は図の下にまとめて配置（多ゾーン対応）
        leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                        ncol=legend_cols, frameon=False)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.22)  # 凡例分の余白

        if savepath_prefix:
            if not os.path.exists(os.path.dirname(savepath_prefix)):
                os.makedirs(os.path.dirname(savepath_prefix), exist_ok=True)
            fig.savefig(f"{savepath_prefix}_{key}_lines.png", dpi=300)
            plt.close(fig)
        else:
            plt.show()

def plot_all_zones_observed_lines(
    data: pd.DataFrame,
    zone_col: str,
    year_col: str,
    observed_cols: dict,
    legend_cols: int = 4,
    zone_order: list | None = None,
    savepath_prefix: str | None = None,
):
    """
    観測データを全ゾーンのスパゲッティ図で描画（凡例あり）。

    Parameters
    ----------
    data : DataFrame
        観測データ（year, zone 列を含む）
    zone_col, year_col : str
        ゾーン・年の列名
    observed_cols : dict
        例: {"investment":"los_t_raw", "development":"develop_res_t_raw",
             "price":"land_price_t", "population":"population_t"}
    legend_cols : int
        凡例の列数
    zone_order : list | None
        凡例の表示順を固定したいときに与える（ゾーンIDの並び）
    savepath_prefix : str | None
        指定すれば "{prefix}_observed_{name}_lines.png" に保存
    """
    title_map = {
        "investment": "Investment (gov) — observed",
        "development": "Development (dev) — observed",
        "price": "Land Price — observed",
        "population": "Population — observed",
    }

    for logical_name, col in observed_cols.items():
        if col not in data.columns:
            continue

        # Year×Zone の行列化（欠損は NaN）
        pivot = data.pivot(index=year_col, columns=zone_col, values=col).sort_index()
        years = pivot.index.to_numpy()
        zones = pivot.columns.to_list()
        if zone_order is not None:
            # 与えられた順序に並べ替え（存在しないものは無視）
            zones = [z for z in zone_order if z in pivot.columns]
            pivot = pivot.reindex(columns=zones)

        fig, ax = plt.subplots()
        for z in pivot.columns:
            s = pivot[z].to_numpy()
            ax.plot(years, s, label=str(z))
        ax.set_xlabel("Year")
        ax.set_ylabel(logical_name.capitalize())
        ax.set_title(title_map.get(logical_name, f"{logical_name} — observed"))
        #ax.grid(True, which="both", linestyle=":")

        leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                        ncol=legend_cols, frameon=False)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.22)

        if savepath_prefix:
            if not os.path.exists(os.path.dirname(savepath_prefix)):
                os.makedirs(os.path.dirname(savepath_prefix), exist_ok=True)
            fig.savefig(f"{savepath_prefix}_observed_{logical_name}_lines.png", dpi=300)
            plt.close(fig)
        else:
            plt.show()
