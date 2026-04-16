
"""小さな補助関数をまとめたユーティリティモジュール。"""


import os
import json
import numpy as np
import matplotlib.pyplot as plt
import time
from contextlib import contextmanager

def ensure_dir(path):
    """存在しないディレクトリを作成する。"""
    if not os.path.exists(path):
        os.makedirs(path)

def save_json(obj, filepath):
    """辞書やリストを JSON ファイルとして保存する。"""
    with open(filepath, 'w') as f:
        json.dump(obj, f, indent=2)

def load_json(filepath):
    """JSON ファイルを読み込んで Python オブジェクトとして返す。"""
    with open(filepath, 'r') as f:
        return json.load(f)

@contextmanager
def timeit(name="Block"):
    """コードブロックの実行時間を表示する context manager。"""
    start = time.time()
    yield
    end = time.time()
    print(f"[{name}] Elapsed time: {end - start:.4f} seconds")
