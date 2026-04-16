# Chapter 5

Chapter 5 のコードは、土地利用と居住地選択の動学モデルを推定し、均衡シミュレーションと反実仮想分析を行うための実装です。  
コードは大きく `データ読み込み・前処理`、`モデル推定`、`forward simulation / 局所均衡計算`、`反実仮想分析` に分かれています。

## このディレクトリでできること

- 行動方策関数と居住地選択モデルの推定
- 推定済みモデルを使った forward simulation
- 特定ゾーンに対する政策変更の反実仮想分析
- 推定結果やシミュレーション結果の保存・可視化

## 主要ファイル

- `main.py`
  Chapter 5 のメイン推定スクリプトです。`config.yaml` を読み込み、共通データを生成し、方策関数推定、遷移モデル推定、forward simulation に基づく特徴量生成、構造パラメータ推定までを実行します。
  Bootstrapで推定誤差を計算する場合、計算サーバーのCPU 47並列で1晩くらい計算時間がかかります。

- `counterfactual.py`
  推定済みモデルを読み込み、対象ゾーンでの政策変更を与えた反実仮想シミュレーションを実行します。均衡経路の再計算と図の保存もここで行います。
  計算サーバーのCPU 47並列で数十分かかります。

- `demand_estimation.py`
  遷移モデルのうち居住地選択モデルの推定を単独で試すためのスクリプトです。
  MXLだと計算サーバーのCPU 47並列で数十分、MNLだと普通のPCでも一瞬で終わります。

- `config.yaml`
  推定期間、ゾーニング、割引率、forward simulation 回数、反実仮想の対象ゾーンなどをまとめた設定ファイルです。
  rayは並列計算の設定で、PCで回す際にnum_cpuに大きな数を設定すると固まるので注意。
  modelは推定モデルの設定。
  counter_factualは反実仮想シミュレーションの設定
  land_price_adjustmentは住宅価格調整パラメータ

- `pyproject.toml`
  依存ライブラリとパッケージ設定です。

## ディレクトリ構成

```text
Chapter5/
├── main.py                   # 推定のメイン入口
├── counterfactual.py         # 反実仮想シミュレーションの入口
├── demand_estimation.py      # 居住地選択モデルの単体推定
├── config.yaml               # 共通設定
├── pyproject.toml            # 依存関係
├── model/                    # 推定モデル・遷移モデル・forward simulation
├── simulation/               # 局所 MPE 計算・政策上書き・可視化
├── utils/                    # 設定管理・共通データ読込・補助関数
├── data/
│   ├── raw/                  # 元データ
│   └── processed/            # 前処理済みデータ
└── notebook/                 # 補助分析用ノートブック
```

## サブディレクトリの役割

### `model/`

モデル本体を置くディレクトリです。

- `transition_model.py`
  居住地選択と人口遷移の中心となるクラスです。ゾーン定義、距離行列、空間重み行列、推定用データの組み立てを担当します。

- `data_reading.py`
  生データから人口、開発、LOS、PT データ、ゾーン対応表などを読み込んで辞書や表に整形します。

- `policy_estimation.py`
  政府投資や開発量の方策関数を推定・保存・読込するためのコードです。

- `mnl.py`, `mxl.py`
  居住地選択モデルの推定実装です。`MXL` が現在のメイン推定に使われています。

- `forward_simulation.py`
  推定済み方策関数と遷移モデルを使って、人口・投資・開発・地価の将来パスをシミュレートします。

- `payoff_structure.py`
  構造推定で使う利得特徴量を定義します。

- `bbl_objective_function.py`
  BBL 型の目的関数を計算します。

### `simulation/`

反実仮想時の均衡計算と図の出力を担います。

- `mpe_local_driver.py`
  単一市場で各期の最適応答を順に解き、局所的な均衡経路を構成します。

- `best_response_mpe.py`
  forward simulation を繰り返し、各主体の best response を数値最適化で求めます。

- `policy_override.py`
  特定ゾーン・特定期の政策や上限制約を既存モデルに上書きするためのラッパーです。

- `visualization.py`
  観測系列とシミュレーション系列を比較する図を保存します。

### `utils/`

共通処理をまとめた補助モジュールです。

- `data_loader.py`
  `main.py` と `counterfactual.py` で共通しているデータ準備処理を一つにまとめています。Chapter 5 の実質的なデータ読み込み入口です。

- `config_manager.py`
  `config.yaml` を読み込み、デフォルト値やバリデーションを含めて設定を一元管理します。

- `helpers.py`
  JSON 保存や簡単な計測などの補助関数です。

### `data/`

- `data/raw/`
  人口、建築確認、PT、ゾーニング定義などの元データです。現行の本流コードで未使用でも、`notebook/` から参照される補助データが一部含まれています。

- `data/processed/`
  ゾーン単位に整形済みの CSV、JSON、NumPy 配列を保存しています。推定と simulation の多くはこのディレクトリを参照します。

## 実行の流れ

1. `config.yaml` で対象期間やゾーン設定を確認する
2. `main.py` で推定と構造パラメータ推定を行う
3. 推定済み結果を使って `counterfactual.py` で反実仮想分析を行う
4. 必要な推定結果や図は実行時に生成する

