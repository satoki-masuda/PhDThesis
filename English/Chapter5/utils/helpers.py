
"""Small utility helpers used across Chapter 5."""


import os
import json
import numpy as np
import matplotlib.pyplot as plt
import time
from contextlib import contextmanager

def ensure_dir(path):
    """Create a directory if it does not already exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def save_json(obj, filepath):
    """Save a Python object to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(obj, f, indent=2)

def load_json(filepath):
    """Load a JSON file and return the parsed Python object."""
    with open(filepath, 'r') as f:
        return json.load(f)

@contextmanager
def timeit(name="Block"):
    """Context manager that prints elapsed execution time."""
    start = time.time()
    yield
    end = time.time()
    print(f"[{name}] Elapsed time: {end - start:.4f} seconds")
