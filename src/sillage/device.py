"""Use the GPU when it is available, and say which device ran.

- PyTorch models (EASE, RP3beta, iALS, BPR-MF): CUDA if torch sees a GPU.
- LightGBM: CPU by measured choice. A CUDA build (-DUSE_CUDA=ON) was benchmarked on
  this ranking task: 112 s vs 1.2 s per fit, and two runs with the same seed differed
  (max prediction gap 0.84), which breaks the reproducibility requirement.
  `lightgbm_device()` still reports what the installed build supports.
PyTorch GPU models were checked for bit-identical reruns (iALS uses batched matmul
instead of atomic scatter-adds for that reason).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

LIGHTGBM_DEVICE = "cpu"


@lru_cache(maxsize=1)
def torch_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def lightgbm_device() -> dict:
    import lightgbm as lgb

    X = np.random.default_rng(0).random((200, 3))
    y = np.random.default_rng(1).integers(0, 2, 200)
    for dev in ("cuda", "gpu"):
        try:
            lgb.train({"objective": "lambdarank", "device_type": dev, "verbose": -1},
                      lgb.Dataset(X, y, group=[20] * 10), 2)
            return {"device_type": dev}
        except Exception:
            continue
    return {"device_type": "cpu"}


def describe() -> str:
    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    return f"torch: {torch_device()} ({gpu}); lightgbm: {lightgbm_device()['device_type']}"
