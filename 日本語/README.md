# D論コードまとめ

このリポジトリは、各章ごとの分析コードや実験コードをまとめた作業ディレクトリです。  
GitHub で共有しやすいように、各章ごとに README を整備しながら、入口スクリプトと生成物を切り分けています。

## ディレクトリ一覧

```text
.
├── Chapter1/   # 初期検証・基礎実験
├── Chapter3/   # Chapter 3 関連コード
└── Chapter5/   # 推定・simulation・反実仮想分析の中心実装
```

## Chapter 3 について

`Chapter3/` には、道路ネットワークのゾーン分割、MFD ベースの集約交通ダイナミクス、避難時の network reconfiguration / contraflow の探索コードが含まれています。

主な入口は次の 3 つです。

- `Chapter3/code/network_partitioning/network_partitioning.py`: ゾーン分割と MFD 推定の中心実装
- `Chapter3/code/reconfigration/src/model/collect_table5_metrics.py`: 複数 policy・複数シナリオの集計入口
- `Chapter3/code/reconfigration/src/model/run_table5_batch.sh`: Table 5 相当の一括実行バッチ


## Chapter 5 について

`Chapter5/` には、土地利用と居住地選択の動学モデルに関する推定コード、forward simulation、局所均衡計算、反実仮想分析のコードがまとまっています。

主な入口は次の 3 つです。

- `Chapter5/main.py`: 推定のメインスクリプト
- `Chapter5/counterfactual.py`: 反実仮想分析の実行スクリプト
- `Chapter5/demand_estimation.py`: 居住地選択モデルの単体推定

