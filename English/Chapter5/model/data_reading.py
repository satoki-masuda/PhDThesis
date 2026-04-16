"""Helpers for converting raw Chapter 5 data into model-ready datasets."""

import unicodedata
import os
import json
from pathlib import Path
import yaml

import numpy as np
import pandas as pd
import geopandas as gpd

def read_zone_code(zoning_type):
    """Return zoning definitions and PT zone-code mappings."""
    base_path = Path(__file__).resolve().parent.parent
    zone_code_in = pd.read_csv(base_path / "data/raw/PT/R05_zone_code.csv", encoding="utf-8")
    zone_code_in = zone_code_in[zone_code_in["市町村名"]=="松山市"].dropna(subset="R05Zone_CD")
    zone_code_in = zone_code_in.reset_index(drop=True)
    zone_code_all = pd.read_csv(base_path / "data/raw/PT/R05_zone_code_all.csv", encoding="utf-8")
    zone_code_all = zone_code_all[zone_code_all["市町村名"]=="松山市"].dropna(subset="R05Zone_CD")
    zone_code_all = zone_code_all.reset_index(drop=True)
    zone_code_in["町丁字名"] = zone_code_in["町丁字名"].apply(lambda x: unicodedata.normalize("NFKC", str(x)))
    zone_code_in["町名"] = zone_code_in["町丁字名_漢字"].str.replace(r'(.*?)([一二三四五六七八九十]+丁目)$', r'\1', regex=True)
    zoning = gpd.read_file(base_path / f"data/raw/zoning/{zoning_type}.geojson", encoding="utf-8")
    zoning = zoning.dropna(subset=["ゾーン名"])
    zoning = zoning.reset_index(drop=True)
    zoning = zoning.to_crs("EPSG:6668")
    # Read the zone-selection dictionary from YAML.
    with open(base_path / f"data/raw/zoning/{zoning_type}.yaml", "r", encoding="utf-8") as f:
        zone_dict = yaml.safe_load(f)
    # Invert the mapping for one-to-one lookup.
    zone_mapping = {v[i]: k for k,v in zone_dict.items() for i in range(len(v))}
    zone_code_in["選択ゾーン"] = zone_code_in["町丁字名"].map(zone_mapping).map(zoning.set_index("ゾーン名")["選択ゾーン"])
    #if zone_code_in["選択ゾーン"].isna().any():
    #    print("以下の町名が選択ゾーンにマッピングできませんでした:")
    #    print(sorted(zone_code_in[zone_code_in["選択ゾーン"].isna()]["町丁字名_漢字"].unique().tolist()))
        
    return zone_code_in, zone_code_all, zoning

def read_pt_data(start_year, end_year, zoning_type):
    """Build or reuse a household-level relocation panel from PT microdata."""
    base_path = Path(__file__).resolve().parent.parent
    folder_path = os.path.join(base_path, f"data/processed/{zoning_type}")
    if os.path.exists(folder_path + f"/df_pool_{start_year}_{end_year}.csv"):
        df_pool = pd.read_csv(folder_path + f"/df_pool_{start_year}_{end_year}.csv", encoding="utf-8")
        print(f"Number of relocation observations: {len(df_pool[df_pool['転居有無']==1])}")
        return df_pool
    
    zone_code_in, zone_code_all, _ = read_zone_code(zoning_type)
    df1 = pd.read_csv(base_path / "data/raw/PT/2023PT_setai_with_kakudai.csv", encoding="utf-8")
    df2 = pd.read_csv(base_path / "data/raw/PT/2023PT_setai_kojin.csv", encoding="utf-8")
    df1 = df1[(df1["39_■5_居住年数_①回答区分"]!=9) & (df1["41_■6_以前のお住まい_①回答区分"]!=2) & (df1["41_■6_以前のお住まい_①回答区分"]!=9)].reset_index(drop=True) # Exclude unknown tenure length, unknown previous address, and foreign locations.
    df1 = df1[(df1["37_■5_所有関係_①回答区分"]==1)].reset_index(drop=True) # Restrict to owner-occupied households.
    df1.loc[pd.isna(df1["63_■6_転居年月_平成：1，令和：2"]), "転居年"] = 0
    # Recover relocation year from the survey coding.
    conditions = {
        9: lambda x: 2023 - x["40_■5_居住年数_②●年間居住"],
        1: lambda x: 1988 + x["64_■6_転居年月_年"],
        2: lambda x: 2018 + x["64_■6_転居年月_年"]
    }

    for code, calc in conditions.items():
        mask = df1["63_■6_転居年月_平成：1，令和：2"] == code
        df1.loc[mask, "転居年"] = calc(df1[mask])

    # Map household-head age from the person file.
    age_df = df2[df2["20_■3_世帯主との続柄"]==1].groupby("5_整理番号_市町村・ロット・SEQ")["22_■3_年齢"].first()
    df1["世帯主年齢"] = df1["5_整理番号_市町村・ロット・SEQ"].map(age_df)
    # Drop households whose head cannot be matched.
    not_found = df1[df1["世帯主年齢"].isna()]["5_整理番号_市町村・ロット・SEQ"].unique()
    if len(not_found) > 0:
        print(f"Household IDs not found in the person file: {not_found}")
        df1 = df1[~df1["5_整理番号_市町村・ロット・SEQ"].isin(not_found)]

    years = end_year - start_year + 1
    df_pool = df1.loc[df1.index.repeat(years)].copy()
    df_pool["年"] = list(range(start_year, end_year+1)) * len(df1)
    df_pool["居住地_前"] = df_pool["6_■1_現住所_住所"]
    df_pool["居住地_後"] = df_pool["6_■1_現住所_住所"]
    mask = df_pool["年"] <= df_pool["転居年"]
    df_pool.loc[mask, "居住地_前"] = df_pool.loc[mask, "42_■6_以前のお住まい_②以前の住所"]
    df_pool.loc[mask, "居住地_後"] = df_pool.loc[mask, "42_■6_以前のお住まい_②以前の住所"]
    df_pool.loc[df_pool["年"] == df_pool["転居年"], "居住地_後"] = df_pool.loc[df_pool["年"] == df_pool["転居年"], "6_■1_現住所_住所"]

    df_pool["転居有無"] = 0
    df_pool.loc[df_pool["転居年"] == df_pool["年"], "転居有無"] = 1
    df_pool = df_pool[["5_整理番号_市町村・ロット・SEQ", "年", "居住地_前", "居住地_後", "転居有無", "転居年", "世帯主年齢"]]
    df_pool.rename(columns={
        "5_整理番号_市町村・ロット・SEQ": "世帯ID"
    }, inplace=True)
    
    zone_mapping = zone_code_all.dropna(subset="R05Zone_CD").set_index("R05Zone_CD")["市町村名"]
    """
    # 松山市の内内移動に絞る
    df_pool = df_pool[(df_pool["居住地_前"].map(zone_mapping) == "松山市") & (df_pool["居住地_後"].map(zone_mapping) == "松山市")].reset_index(drop=True)
    """
    # 松山市内の移動と市外から市内への移動に絞る
    #df_pool = df_pool[(df_pool["居住地_後"].map(zone_mapping) == "松山市")].reset_index(drop=True)
    # 町以下不明を一丁目に変換
    mask = df_pool["居住地_前"].apply(lambda x: str(x).endswith("99"))
    df_pool.loc[mask, "居住地_前"] = df_pool.loc[mask, "居住地_前"].apply(lambda x: int(str(x)[:4] + "01"))
    mask = df_pool["居住地_後"].apply(lambda x: str(x).endswith("99"))
    df_pool.loc[mask, "居住地_後"] = df_pool.loc[mask, "居住地_後"].apply(lambda x: int(str(x)[:4]+"01"))
    df_pool["居住地_前_ゾーン"] = -1 # 松山市外
    df_pool["居住地_後_ゾーン"] = -1 # 松山市外
    zone_mapping = zone_code_in.dropna(subset="R05Zone_CD").set_index("R05Zone_CD")["選択ゾーン"]
    mask = df_pool["居住地_前"].isin(zone_mapping.index)
    df_pool.loc[mask, "居住地_前_ゾーン"] = df_pool.loc[mask, "居住地_前"].map(zone_mapping)
    mask = df_pool["居住地_後"].isin(zone_mapping.index)
    df_pool.loc[mask, "居住地_後_ゾーン"] = df_pool.loc[mask, "居住地_後"].map(zone_mapping)
    df_pool.dropna(subset=["居住地_前_ゾーン", "居住地_後_ゾーン"], inplace=True)
    # 現在の居住地が松山市内ではない世帯を削除
    df_pool = df_pool[~df_pool["世帯ID"].isin(
        df_pool.loc[(df_pool["居住地_後_ゾーン"] == -1) & (df_pool["年"] == end_year), "世帯ID"]
    )].reset_index(drop=True)
    df_pool.reset_index(drop=True, inplace=True)

    # 世帯主の年齢
    for n, data in df_pool.groupby("世帯ID", as_index=False):
        df_pool.loc[df_pool["世帯ID"]==n, "世帯主年齢"] = np.arange(data["世帯主年齢"].iloc[0] - len(data)+1, data["世帯主年齢"].iloc[0]+1)
        if len(data) < years:
            df_pool.drop(df_pool[df_pool["世帯ID"]==n].index, inplace=True)
    df_pool.reset_index(drop=True, inplace=True)
    
    """
    # 世帯年収
    income_dict = {1: 100, 2: 300, 3: 500, 4: 800, 5: 1250, 6: 1750, 7: np.nan}
    for n, data in df_pool.groupby("世帯ID", as_index=False):
        df_pool.loc[df_pool["世帯ID"]==n, "世帯年収"] = df1.loc[df1["5_整理番号_市町村・ロット・SEQ"]==n, "70_■8_世帯年収"].apply(lambda x: income_dict.get(x, np.nan)).values[0]
    df_pool["世帯年収"] = df_pool["世帯年収"].astype(float)
    # 年収無回答の世帯は削除
    df_pool.dropna(subset=["世帯年収"], inplace=True)
    df_pool.reset_index(drop=True, inplace=True)
    
    #年収無回答の世帯は中央値で補完
    #median_income = df_pool["世帯年収"].median()
    #df_pool["世帯年収"] = df_pool["世帯年収"].fillna(median_income)
    """
    # 自動車の有無 (転居前と転居後)
    df1[["10_■2_保有車両_軽乗用・乗用車_ガソリン・ディーゼル車", "11_■2_保有車両_軽乗用・乗用車_ＥＶ", "12_■2_保有車両_軽乗用・乗用車_ＨＢＤ", 
         "13_■2_保有車両_軽貨物・普通貨物車_ガソリン・ディーゼル車", "14_■2_保有車両_軽貨物・普通貨物車_ＥＶ", "15_■2_保有車両_軽貨物・普通貨物車_ＨＢＤ", 
         "66_■6_転居直前の自動車等の保有台数_自動車"]] = df1[["10_■2_保有車両_軽乗用・乗用車_ガソリン・ディーゼル車", "11_■2_保有車両_軽乗用・乗用車_ＥＶ", "12_■2_保有車両_軽乗用・乗用車_ＨＢＤ", 
                                             "13_■2_保有車両_軽貨物・普通貨物車_ガソリン・ディーゼル車", "14_■2_保有車両_軽貨物・普通貨物車_ＥＶ", "15_■2_保有車両_軽貨物・普通貨物車_ＨＢＤ", 
                                             "66_■6_転居直前の自動車等の保有台数_自動車"]].replace({99: 0})
    for n, data in df_pool.groupby("世帯ID", as_index=False):
        df1_mask = df1.loc[df1["5_整理番号_市町村・ロット・SEQ"]== n]
        df_pool.loc[(df_pool["世帯ID"]==n) & (df_pool["年"] <= data["転居年"].iloc[0]), "自動車有無"] = df1_mask["66_■6_転居直前の自動車等の保有台数_自動車"].iloc[0]
        df_pool.loc[(df_pool["世帯ID"]==n) & (df_pool["年"] > data["転居年"].iloc[0]), "自動車有無"] = df1_mask["10_■2_保有車両_軽乗用・乗用車_ガソリン・ディーゼル車"].iloc[0] + df1_mask["11_■2_保有車両_軽乗用・乗用車_ＥＶ"].iloc[0] + df1_mask["12_■2_保有車両_軽乗用・乗用車_ＨＢＤ"].iloc[0] + \
            df1_mask["13_■2_保有車両_軽貨物・普通貨物車_ガソリン・ディーゼル車"].iloc[0] + df1_mask["14_■2_保有車両_軽貨物・普通貨物車_ＥＶ"].iloc[0] + df1_mask["15_■2_保有車両_軽貨物・普通貨物車_ＨＢＤ"].iloc[0]
    # 二値変数に変換
    df_pool["自動車有無"] = df_pool["自動車有無"].apply(lambda x: 1 if x > 0 else 0)
    
    # 子供の有無 (転居当時ではなく現在の状態を使用)
    df_pool["子供有無"] = 0
    for n, data in df_pool.groupby("世帯ID", as_index=False):
        df_child = df2.loc[df2["5_整理番号_市町村・ロット・SEQ"]==n, "22_■3_年齢"]
        if len(df_child) > 0 and df_child.min() <= end_year - start_year: # start_yearとend_yearの間に子供が誕生
            ref_year = df_child[df_child <= end_year - start_year].max() + 1
            df_pool.loc[df_pool["世帯ID"]==n, "子供有無"] = [0] * (years - ref_year) + [1] * ref_year
        elif len(df_child) > 0 and end_year - start_year < df_child.min() <= 18: # 全期間18歳以下の子供あり
            df_pool.loc[df_pool["世帯ID"]==n, "子供有無"] = 1
        elif len(df_child) > 0 and 18 < df_child.min() <= 18 + (end_year - start_year): # 期間中に18歳を超える子供がいる
            ref_year = 18 + (end_year - start_year) - df_child.min() + 1
            df_pool.loc[df_pool["世帯ID"]==n, "子供有無"] = [1] * ref_year + [0] * (years - ref_year)
    
    # 市外転入
    df_pool["市外転入"] = 0
    for n, data in df_pool.groupby("世帯ID", as_index=False):
        if data["居住地_前_ゾーン"].iloc[0] == -1 and data["居住地_後_ゾーン"].iloc[-1] != -1:
            df_pool.loc[df_pool["世帯ID"]==n, "市外転入"] = 1
    
    # 拡大係数
    kakudai_mapping = df1.set_index("5_整理番号_市町村・ロット・SEQ")["拡大係数"]
    df_pool["拡大係数"] = df_pool["世帯ID"].map(kakudai_mapping)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    df_pool.to_csv(folder_path + f"/df_pool_{start_year}_{end_year}.csv", index=False, encoding="utf-8-sig")
    print(f"データ数: {len(df_pool[df_pool['転居有無']==1])}")
    
    return df_pool

def read_pop_data(start_year, end_year, zoning_type):
    """Build or reuse yearly zone-level population dictionaries."""
    base_path = Path(__file__).resolve().parent.parent
    folder_path = os.path.join(base_path, f"data/processed/{zoning_type}")
    if os.path.exists(folder_path + f"/pop_data_{start_year}_{end_year}.json"):
        with open(folder_path + f"/pop_data_{start_year}_{end_year}.json", "r") as f:
            json_pop_dict = json.load(f)
        pop_dict = {}
        for year in json_pop_dict:
            pop_dict[int(year)] = {}
            for index in json_pop_dict[year]:
                pop_dict[int(year)][index] = {}
                for zone, data in json_pop_dict[year][index].items():
                    pop_dict[int(year)][index][int(zone)] = data
        return pop_dict
    
    zone_code_in, _, zoning = read_zone_code(zoning_type)
    # Build population totals from the resident registry files.
    pop_dict = {}
    na_list = []  # マッピングできなかった町名のリスト
    for year in range(start_year, end_year+1):
        usecols = [0, 1, 2, 3] if year < 2015 else [0, 2, 3, 4]
        if year == 2018:
            pop = pd.read_excel(base_path / f"data/raw/population_by_chome/2018.xls", header=2, usecols=usecols, sheet_name=11)
        elif year >= 2023:
            pop = pd.read_excel(base_path / f"data/raw/population_by_chome/{year}.xlsx", header=2, usecols=usecols, sheet_name="1月1日")
        else:
            pop = pd.read_excel(base_path / f"data/raw/population_by_chome/{year}.xls", header=2, usecols=usecols, sheet_name="1月1日")
            
        pop["町名"] = pop["町名"].apply(lambda x: unicodedata.normalize("NFKC", str(x)).replace('　', '').replace(' ', ''))
        pop["地区名"] = pop["地区名"].apply(lambda x: unicodedata.normalize("NFKC", str(x)).replace('　', '').replace(' ', ''))
        pop = pop.dropna(subset=["地区名", "町名", "人口", "世帯数"])
        pop.reset_index(drop=True, inplace=True)
        pop = pop.loc[(~pop["町名"].isin(zone_code_in.loc[zone_code_in["R05大ゾーン"] == "松山市28区", "町丁字名"].to_list())) & (~pop["町名"].str.contains("計")) & (~pop["地区名"].str.contains("計"))].reset_index(drop=True)
        
        with open(base_path / f"data/raw/zoning/{zoning_type}.yaml", "r", encoding="utf-8") as f:
            zone_dict = yaml.safe_load(f)
        # Swap key and value for one-to-one lookup.
        zone_mapping = {v[i]: k for k,v in zone_dict.items() for i in range(len(v))}
        pop["選択ゾーン"] = pop["町名"].map(zone_mapping).map(zoning.set_index("ゾーン名")["選択ゾーン"])
        na_list += pop[pop["選択ゾーン"].isna()]["町名"].unique().tolist()
        pop = pop.dropna(subset=["選択ゾーン"])
        pop = pop.reset_index(drop=True)
        pop["選択ゾーン"] = pop["選択ゾーン"].astype(int)
        pop_dict[year] = pop.groupby("選択ゾーン").agg({"人口": "sum", "世帯数": "sum"}).to_dict()
        zero_zones = set(zoning['選択ゾーン'].unique().tolist()) - set(pop_dict[year]['人口'].keys())
        print(f"Missing zones in {year}: {sorted(list(zero_zones))}")
        for zone in zero_zones:
            pop_dict[year]['人口'][zone] = 0
            pop_dict[year]['世帯数'][zone] = 0
        
    
    print(f"Unmapped town names: {sorted(list(set(na_list)))}")
    print(f"Number of zones: {[len(pop_dict[year]['人口']) for year in range(start_year, end_year+1)]}")
    # Save as JSON.
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(folder_path + f"/pop_data_{start_year}_{end_year}.json", "w", encoding='utf-8') as f:
        json.dump(pop_dict, f, ensure_ascii=False)
        
    return pop_dict

def read_building_data(start_year, end_year, zoning_type):
    """Build or reuse yearly zone-level new-building area data."""
    base_path = Path(__file__).resolve().parent.parent  # model/ から1階層上へ
    folder_path = os.path.join(base_path, f"data/processed/{zoning_type}")
    if os.path.exists(folder_path + f"/building_data_{start_year}_{end_year}.json"):
        with open(folder_path + f"/building_data_{start_year}_{end_year}.json", "r") as f:
            json_building_dict = json.load(f)
        building_dict = {}
        for year in json_building_dict:
            building_dict[int(year)] = {}
            for index in json_building_dict[year]:
                building_dict[int(year)][index] = {}
                for string, data in json_building_dict[year][index].items():
                    key = tuple(map(int, string.split("_")))  # 文字列キーをタプルに変換
                    building_dict[int(year)][index][key] = data
        return building_dict
    
    _, _, zoning = read_zone_code(zoning_type)
    af1 = gpd.read_file(base_path / "data/raw/建築確認/2008_2013/新築動向_2008-2013.geojson", encoding="utf-8")
    # Keep only observations overlapping the zoning polygons.
    af1 = af1[af1.geometry.intersects(zoning.union_all())]
    af1 = point_to_zone(af1, zoning)
    af1["用途コード"] = af1["用途コード"].astype(int)
    af1["選択ゾーン"] = af1["選択ゾーン"].astype(int)
    af1 = af1.loc[(af1["種別"]=="新築"), :].reset_index(drop=True)  # Keep only new housing.
    
    af2 = gpd.read_file(base_path / "data/raw/建築確認/2014_2018/新築動向_2014-2018.geojson", encoding="utf-8")
    af2 = af2[af2.geometry.intersects(zoning.union_all())]
    af2 = point_to_zone(af2, zoning)
    af2["用途コード"] = af2["用途コード"].astype(int)
    af2["選択ゾーン"] = af2["選択ゾーン"].astype(int)
    af2 = af2.loc[(af2["工事種別"]=="新築"), :].reset_index(drop=True)  # Keep only new housing.
    
    af3 = gpd.read_file(base_path / "data/raw/建築確認/2019_2023/新築動向_2019-2023.geojson", encoding="utf-8")
    af3 = point_to_zone(af3, zoning)
    af3["年"] = af3["年"].astype(int)
    af3["用途コード"] = af3["用途コード"].astype(int)
    af3["選択ゾーン"] = af3["選択ゾーン"].astype(int)
    
    # Aggregate area by zone and land-use code.
    building_dict = {}
    for year in range(start_year, end_year+1):
        if year <= 2013:
            df = af1
            df = df[df["建築許可日"].dt.year == year]
        elif year <= 2018:
            df = af2
            df = df[df["申請年月日"].dt.year == year]
        else:
            df = af3
            df = df[df["年"] == year]
        
        df = df.rename(columns={"敷地面積": "面積"})
        # Aggregate floor area and counts by zone and use code.
        building_dict[year] = df.groupby(["選択ゾーン", "用途コード"]).agg({
            "ID": "count",
            "面積": "sum"
        }).to_dict()
    
    # Convert tuple keys into JSON-safe strings.
    json_building_dict = {}
    for year in building_dict:
        json_building_dict[year] = {}
        for index in building_dict[year]:
            json_building_dict[year][index] = {}
            for (zone, purpose), data in building_dict[year][index].items():
                key = f"{zone}_{purpose}"  # tupleキーを文字列に変換
                json_building_dict[year][index][key] = data
            
    # Save as JSON.
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(folder_path + f"/building_data_{start_year}_{end_year}.json", "w", encoding='utf-8') as f:
        json.dump(json_building_dict, f, ensure_ascii=False)
    
    return building_dict
    
def point_to_zone(gdf:gpd.GeoDataFrame, zoning):
    """Attach the matching `選択ゾーン` value to each point observation."""
    # Identify which polygon contains each point.
    gdf["選択ゾーン"] = gdf.apply(lambda row: zoning.loc[zoning.geometry.contains(row.geometry), "選択ゾーン"].values[0] if not zoning[zoning.geometry.contains(row.geometry)].empty else None, axis=1)
    
    # Slightly perturb unmatched points to recover boundary cases.
    for idx, row in gdf.loc[pd.isna(gdf["選択ゾーン"])].iterrows():
        for xoff, yoff in [(0.00005, 0), (0, 0.00005), (-0.00005, 0), (0, -0.00005)]:
            new_geometry = gpd.GeoSeries(gdf.loc[idx, "geometry"]).translate(xoff, yoff).values[0]
            if len(zoning[zoning.geometry.contains(new_geometry)]) > 0:
                # Use the shifted geometry if it falls inside a polygon.
                gdf.loc[idx, "geometry"] = new_geometry
                gdf.loc[idx, "選択ゾーン"] = zoning.loc[zoning.geometry.contains(new_geometry), "選択ゾーン"].values[0]
                break
        
    print(f"Unmapped observations: {sum(pd.isna(gdf['選択ゾーン']))} / {len(gdf)}")
    gdf = gdf.dropna(subset=["選択ゾーン"])
    gdf = gdf.reset_index(drop=True)
    
    return gdf

def convert_jp_year(x):
    """Convert Japanese era dates into Gregorian `YYYYMMDD` strings."""
    if pd.isna(x):
        return x
    x = str(x)
    if x.startswith('H'):
        year = int(x[1:].split('.')[0]) + 1988
        month = x.split('.')[1].zfill(2)
        day = x.split('.')[2].zfill(2)
        return f"{year}{month}{day}"
    elif x.startswith('R'):
        year = int(x[1:].split('.')[0]) + 2018
        month = x.split('.')[1].zfill(2)
        day = x.split('.')[2].zfill(2)
        return f"{year}{month}{day}"
    return x

def read_los_data(start_year, end_year, zoning_type):
    """Return public transport LOS data as yearly zone-level dictionaries."""
    base_path = Path(__file__).resolve().parent.parent  # model/ から1階層上へ
    folder_path = os.path.join(base_path, f"data/processed/{zoning_type}")
    if os.path.exists(folder_path + f"/los_data_{start_year}_{end_year}.json"):
        with open(folder_path + f"/los_data_{start_year}_{end_year}.json", "r") as f:
            json_los_dict = json.load(f)
        los_dict = {}
        for year in json_los_dict:
            los_dict[int(year)] = {}
            for zone in json_los_dict[year]:
                los_dict[int(year)][int(zone)] = {}
                for mode, data in json_los_dict[year][zone].items():
                    los_dict[int(year)][int(zone)][mode] = data
        return los_dict
    
    _, _, zoning = read_zone_code(zoning_type)
    zone_list = zoning["選択ゾーン"].astype(int).tolist()
    
    bus = pd.read_excel(base_path / "data/raw/public_transp_los.xlsx", sheet_name="路線バス")
    bus[zoning_type] = bus[zoning_type].astype(str)
    shinai = pd.read_excel(base_path / "data/raw/public_transp_los.xlsx", sheet_name="市内電車")
    shinai[zoning_type] = shinai[zoning_type].astype(str)
    kougai = pd.read_excel(base_path / "data/raw/public_transp_los.xlsx", sheet_name="郊外電車")
    kougai[zoning_type] = kougai[zoning_type].astype(str)
    bus.columns = bus.columns.astype(str)
    shinai.columns = shinai.columns.astype(str)
    kougai.columns = kougai.columns.astype(str)

    df_bus = pd.DataFrame(columns=[f"{str(i)}" for i in range(start_year, end_year + 1)], index=zone_list)
    for zone in zone_list:
        df_bus.loc[zone, :] = bus.loc[[zone in [int(s) for s in bus[zoning_type].values[j].split(',') if bus[zoning_type].values[j]!='nan'] for j in range(len(bus))], f"{start_year}":f"{end_year}"].sum(axis=0)

    df_shinai = df_bus.copy()
    for zone in zone_list:
        df_shinai.loc[zone, :] = shinai.loc[[zone in [int(s) for s in shinai[zoning_type].values[j].split(',') if shinai[zoning_type].values[j]!='nan'] for j in range(len(shinai))], f"{start_year}":f"{end_year}"].sum(axis=0)

    df_kougai = df_bus.copy()
    for zone in zone_list:
        df_kougai.loc[zone, :] = kougai.loc[[zone in [int(s) for s in kougai[zoning_type].values[j].split(',') if kougai[zoning_type].values[j]!='nan'] for j in range(len(kougai))], f"{start_year}":f"{end_year}"].sum(axis=0)

    # df_total stores the sum of bus, tram, and suburban rail LOS.
    df_total = df_bus.copy()
    for zone in zone_list:
        df_total.loc[zone, :] = df_bus.loc[zone, :] + df_shinai.loc[zone, :] + df_kougai.loc[zone, :]
    
    dict_bus = df_bus.to_dict(orient="dict")
    dict_shinai = df_shinai.to_dict(orient="dict")
    dict_kougai = df_kougai.to_dict(orient="dict")
    dict_total = df_total.to_dict(orient="dict")
    
    # Merge the mode-specific dictionaries.
    dict_los = {}
    for year in range(start_year, end_year + 1):
        dict_los[year] = {}
        for zone in dict_bus[str(year)]:
            dict_los[year][zone] = {
                "bus": int(dict_bus[str(year)][zone]),
                "shinai": int(dict_shinai[str(year)][zone]),
                "kougai": int(dict_kougai[str(year)][zone]),
                "total": int(dict_total[str(year)][zone])
            }
    
    # Save as JSON.
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(folder_path + f"/los_data_{start_year}_{end_year}.json", "w", encoding='utf-8') as f:
        json.dump(dict_los, f, ensure_ascii=False)
    return dict_los

def read_obs_share_data(start_year, end_year, zoning, zoning_type):
    """Construct observed destination-zone shares."""
    assert zoning_type == "pop_zone", "obs_share_dataはpop_zoneのみ対応"
    assert start_year >= 2015 and end_year <= 2023, "obs_share_dataは2015年から2023年まで対応"
    base_path = Path(__file__).resolve().parent.parent
    data_inside = pd.read_excel(base_path / "data/raw/district_social_mobility/data.xlsx", sheet_name="地区内転居入")
    data_move_in = pd.read_excel(base_path / "data/raw/district_social_mobility/data.xlsx", sheet_name="地区外転入")
    # Keep only zones used in the current zoning scheme.
    data_inside = data_inside.loc[data_inside["ゾーン"].isin(zoning["ゾーン名"].astype(str).tolist()), :]
    data_move_in = data_move_in.loc[data_move_in["ゾーン"].isin(zoning["ゾーン名"].astype(str).tolist()), :]
    obs_share = pd.DataFrame(0, index=range(start_year, end_year+1), columns=zoning["選択ゾーン"].astype(int).tolist())
    for year in range(start_year, end_year+1):
        obs_share.loc[year, :] = data_inside[year].values + data_move_in[year].values
    obs_share = obs_share.fillna(0).astype(float)
    obs_share = obs_share.values
    obs_share /= np.sum(obs_share, axis=1, keepdims=True)
    
    return obs_share

def distance_matrix(choice_zone, zoning_type):
    """Build or reuse a centroid-to-centroid distance matrix."""
    base_path = Path(__file__).resolve().parent.parent
    folder_path = os.path.join(base_path, f"data/processed/{zoning_type}")
    if os.path.exists(folder_path + "/distance_matrix.csv"):
        return pd.read_csv(folder_path + "/distance_matrix.csv", encoding="utf-8")
    choice_zone["選択ゾーン"] = choice_zone["選択ゾーン"].astype(int)
    choice_zone = choice_zone.to_crs(epsg=6672)

    # Compute centroids.
    choice_zone["centroid"] = choice_zone.geometry.centroid

    # Compute pairwise distances.
    distance_records = []
    for i, row_i in choice_zone.iterrows():
        for j, row_j in choice_zone.iterrows():
            dist = row_i["centroid"].distance(row_j["centroid"])
            distance_records.append({
                "zone_1": row_i["選択ゾーン"],
                "zone_2": row_j["選択ゾーン"],
                "distance_km": round(dist, 2)/1000,  # Convert from meters to km.
            })

    # Save to CSV.
    df_dist = pd.DataFrame(distance_records)
    df_dist.to_csv(folder_path + "/distance_matrix.csv", index=False, encoding="utf-8")
    
    return df_dist

def read_move_data(zoning):
    """Load move rates, out-migration rates, and inflow counts by zone."""
    base_path = Path(__file__).resolve().parent.parent
    move_ratio = pd.read_excel(base_path / "data/raw/district_social_mobility/data.xlsx", sheet_name="地区内転居出率")
    move_ratio = move_ratio.loc[move_ratio["ゾーン"].isin(zoning["ゾーン名"].astype(str).tolist()), :].drop(columns=["ゾーン", "期間中平均移動率"]).values
    exiting_ratio = pd.read_excel(base_path / "data/raw/district_social_mobility/data.xlsx", sheet_name="地区外転出率")
    exiting_ratio = exiting_ratio.loc[exiting_ratio["ゾーン"].isin(zoning["ゾーン名"].astype(str).tolist()), :].drop(columns=["ゾーン", "期間中平均移動率"]).values
    inflow_pop = pd.read_excel(base_path / "data/raw/district_social_mobility/data.xlsx", sheet_name="地区外転入")
    inflow_pop = inflow_pop.loc[inflow_pop["ゾーン"].isin(zoning["ゾーン名"].astype(str).tolist()), :].drop(columns=["ゾーン", "期間中増減", "期間中増減率"]).values
    
    return move_ratio, exiting_ratio, inflow_pop


if __name__ == "__main__":
    # Quick test entry point.
    #zone_code_in, zone_code_all, zoning = read_zone_code("chou_zone")
    read_pt_data(start_year=2015, end_year=2023, zoning_type="pop_zone")
    #read_pop_data(start_year=2008, end_year=2023, zoning_type="19_zone")
    #read_building_data(start_year=2008, end_year=2023, zoning_type="19_zone")
    #read_los_data(start_year=2008, end_year=2023, zoning_type="chou_zone")
    #distance_matrix(choice_zone)
