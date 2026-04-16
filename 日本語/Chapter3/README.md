# Chapter 3

Chapter 3 では、避難時の道路ネットワークをゾーン単位に集約し、MFD ベースの簡略化ネットワーク上で contraflow / network reconfiguration の探索と比較実験を行います。

## どこから見るか

最初に見るファイルは次の 5 つです。

- `code/network_partitioning/network_partitioning.py`
  SUMO ネットワークとリンク密度からクラスタ分割を作る中心実装です。ゾーン分割、クラスタごとの MFD 推定、出力保存までをまとめて扱います。
- `code/sumo/simulation_parking.py`
  ゾーン分割と需要ファイルを受けて、ベースラインの SUMO シミュレーションを回す入口です。
- `code/network_partitioning/sigma_i.py`
  `NetworkPartitioning` を呼び出して、`sigma_i` を変えながら分割結果を比較する実行スクリプトです。
- `code/reconfigration/src/model/collect_table5_metrics.py`
  論文の比較表に近い形で、複数シナリオ・複数 policy の評価をまとめて走らせる入口です。
- `code/reconfigration/src/model/run_table5_batch.sh`
  `collect_table5_metrics.py` をシナリオごとにまとめて回すバッチスクリプトです。

`reconfigration` はディレクトリ名に typo がありますが、既存コードの import / 相対パスに合わせてそのまま残しています。

## ディレクトリ構成

```text
Chapter3/
├── code/
│   ├── network_partitioning/
│   │   ├── network_partitioning.py
│   │   ├── sigma_i.py
│   │   ├── zoning.csv
│   │   ├── network_partitioning.ipynb
│   │   └── log_norm/
│   ├── sumo/
│   │   ├── network/
│   │   ├── demand/
│   │   ├── evac_shelter/
│   │   ├── simulation_parking.py
│   │   └── simulation_parking_contraflow.py
│   └── reconfigration/
│       ├── data/
│       │   ├── raw/
│       │   └── processed/
│       ├── output/
│       └── src/
│           ├── data/
│           └── model/
├── pyproject.toml
└── poetry.lock
```

## `network_partitioning/` の役割

- `network_partitioning.py`
  ネットワークの読み込み、リンク密度の集約、line graph の構築、類似度行列の作成、recursive bipartition、クラスタ可視化、MFD / 回帰結果の保存を担います。
- `sigma_i.py`
  パラメータ感度を見るための小さな実行スクリプトです。現在は `sigma_i = 2, 3, 4` を順に試す設定になっています。
- `zoning.csv`
  分割済みクラスタを edge id の集合として保存したファイルです。
- `log_norm/`
  `sigma_i` やクラスタ数ごとの MFD, 回帰結果, 図などの生成物です。GitHub 公開時には再生成物として扱うのが自然です。
- `network_partitioning.ipynb`
  補助的な検証用 notebook です。再現の本流は `.py` 側です。

## `reconfigration/` の役割

### `data/`

- `data/raw/`
  ネットワーク、ゾーニング、避難所一覧などの入力データです。
- `data/processed/`
  OD、境界容量、平均トリップ長、ゾーンポリゴン、経路辞書など、シミュレーション前処理済みのデータです。

### `src/data/`

- `mfd_zoning.py`
  ゾーン分割を読み込み、隣接関係・経路列挙・ゾーン可視化・MFD 推定用の前処理をまとめたユーティリティです。
- `od_generate.py`
  SUMO の trip / OD XML から、平常時・避難時の OD CSV を作る前処理スクリプトです。

### `src/model/`

中核モジュール

- `parameters_ndp.py`
  データ読込とパラメータ管理の中心です。MFD パラメータ、OD、経路選択確率、容量制約などをここでまとめます。
- `mfd_dynamics.py`
  集約ネットワーク上のシミュレーション本体です。
- `reconf_shortest.py`
  初期グラフから目標グラフへの「最短ステップ」再構成を扱う実装です。
- `reconf_horizon.py`
  一定 horizon 内で到達可能な再構成系列を扱う実装です。step change limit を含む感度分析系の基盤にもなっています。
- `cross-entropy.py`
  再構成系列の探索に cross-entropy 法を使う最適化実装です。
- `discrete_mpc.py`
  再構成を逐次的に選ぶ MPC 系の比較実装です。
- `value_iteration.py`
  離散状態上での価値反復による方策評価・探索を試す実装です。
- `contraflow_ndp.py`
  静的な contraflow パターンを探索する旧めの最適化実装です。

比較・集計・可視化スクリプト

- `collect_table5_metrics.py`
  シナリオ別に複数 policy を評価して CSV にまとめる集計入口です。
- `run_table5_batch.sh`
  Table 5 相当のシナリオ一括実行バッチです。
- `run_optimizer_comparison.sh`
  CEM+ZDD と GA+ZDD の比較実験をまとめて回します。
- `plot_optimizer_comparison.py`
  最適化履歴を読み込んで比較図を作ります。
- `plot_transition_congestion_comparison.py`
  再構成系列ごとの混雑推移を比較可視化します。
- `plot_pareto_frontier.py`
  多目的比較のフロンティアを描きます。
- `compare_macro_micro_validation.py`
  マクロ近似とマイクロシミュレーションの整合性確認用です。
- `compare_sampling_efficiency.py`
  サンプリング効率の比較を行います。
- `analyze_step_change_limit_sensitivity.py`
  step-change-limit の感度分析をまとめます。
- `collect_table5_metrics.py`, `format_step_change_limit_table.py`
  論文表の作成や整形に使う集計スクリプトです。

補助スクリプト

- `make_constraint.py`
  実行前に使う制約グラフの生成・確認用です。
- `memory_calculation_shortest.py`, `memory_calculation_horizon.py`
  探索時のメモリ計測用です。
- `run_cem_sequence_baseline.py`, `genetic_sequence_baseline.py`
  ベースライン実験の実行用です。
- `logger_writer.py`, `cost_function.py`
  ログ出力やコスト計算の補助です。
- `calculation_reconf.ipynb`
  モデル検討用 notebook です。現在の再現入口は `.py` 側を優先して読むのがわかりやすいです。

### `output/`

- `output/`
  MFD 推定結果、再構成系列、図、比較表、動画フレームなどの生成物置き場です。
- `output/mfd_dynamics/`, `output/reconfiguration/`, `output/optimizer_baselines/` などは、実験を回すと大きくなりやすいので GitHub では再生成前提にするのが扱いやすいです。
- ただし `output/ScaledMFD/fitted_scale_factors.csv` と `output/ParkingSuccessRate/zone_speed_penalty/estimated_params.csv` は、現状の `parameters_ndp.py` から入力として読まれています。ここは単純な削除候補ではなく、「別の保存場所へ移す」か「再生成手順を README に書く」前提で整理するのが安全です。

## `sumo/` の役割

`code/sumo/` は、Chapter 3 の集約モデルに入る前のミクロシミュレーション入力や検証用データを作る層です。

### `network/`

- `network.py`
  OSM からベースの SUMO ネットワークを作り、`output.net.xml` と `output.geojson` を出力します。
- `network_partitioned.py`
  `network_partitioning/zoning.csv` を使って、ゾーン別の小ネットワークを作る補助スクリプトです。
- `network/data/`
  SUMO ネットワーク本体、geojson、contraflow 定義 CSV などの入力・中間成果物です。

### `demand/`

- `demand/od_koto/od_demand.py`
  Koto 地域の OD 表から通常の SUMO trip / route を作ります。
- `demand/od_koto/od_demand_parking.py`
  避難先候補や parking 情報を含む需要シナリオを作る本流スクリプトです。
- `demand/random/random_demand.py`, `random_DUE.py`
  ランダム需要を使った小規模な検証・感度確認用です。
- `demand/od_koto/OD_matrix/`, `pop_synthesis/`
  OD 表や合成人口の作成に使う前処理群です。

### `evac_shelter/`

- `FLOOD_API.py`
  緯度経度から浸水深・継続時間を取得する補助関数です。
- `evac_flood_koto.csv`, `evac_flood_koto_depth.csv`
  避難所候補と浸水情報を組み合わせた入力データです。

### シミュレーション実行

- `simulation_parking.py`
  基本ケースの SUMO シミュレーションを実行し、edge data, tripinfo, vehroute を出力します。
- `simulation_parking_contraflow.py`
  `contraflow_simulation/*/best_sequence.csv` を読んで、時間変化する contraflow を SUMO 上で再現します。
- `contraflow_simulation/`
  マクロ側最適化で得た `best_sequence.csv` を SUMO 側へ渡すための入力置き場です。

## 実行の流れ

典型的には次の順で追うと理解しやすいです。

1. `code/sumo/network/network.py` でベースの SUMO ネットワークを作る
2. `code/network_partitioning/network_partitioning.py` でゾーン分割を作る
3. `code/sumo/demand/od_koto/od_demand_parking.py` や `demand/random/*.py` で SUMO 需要を作る
4. `code/sumo/simulation_parking.py` / `simulation_parking_contraflow.py` でミクロシミュレーションを回す
5. `code/reconfigration/src/data/od_generate.py` と `mfd_zoning.py` で集約モデル向けデータを整える
6. `code/reconfigration/src/model/parameters_ndp.py` で入力データを束ねる
7. `code/reconfigration/src/model/mfd_dynamics.py` で集約シミュレーションを回す
8. `code/reconfigration/src/model/reconf_shortest.py` / `reconf_horizon.py` / `cross-entropy.py` で再構成を探索する
9. `code/reconfigration/src/model/collect_table5_metrics.py` や各 plot スクリプトで結果を集計する

