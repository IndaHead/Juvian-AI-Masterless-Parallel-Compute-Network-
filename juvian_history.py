# Juvian Grid
# Copyright 2026 Jason M. Vajler & Dwayne Aubery
# Lead Author: Jason M. Vajler | Co-Author: Dwayne Aubery
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
==============================================================================
JUVIAN GRID :: HISTORY LEDGER
Memory-mapped 3D tensor of validated reductions. Zero RAM growth -- writes
stream straight to disk blocks via np.memmap.

Layout  H in R^(E x F x M):
    E  epoch  (rolling window, MAX_EPOCHS)
    F  feature slots (top latent modes)
    M  metric vector [timestamp, energy, valid_nodes, purged_nodes]

Fix vs original: log_validated_reduction always calls initialize_ledger first,
so it can never crash on a missing file.
==============================================================================
"""

import os
import time
import threading
import numpy as np

HISTORY_FILE = "juvian_history.npy"
MAX_EPOCHS, FEATURE_SLOTS, METRIC_DIM = 10000, 8, 4
_SHAPE = (MAX_EPOCHS, FEATURE_SLOTS, METRIC_DIM)
_lock = threading.Lock()


class JuvianHistoryLedger:

    @staticmethod
    def initialize(path: str = HISTORY_FILE):
        with _lock:
            if not os.path.exists(path):
                fp = np.memmap(path, dtype="float64", mode="w+", shape=_SHAPE)
                fp[:] = 0.0
                fp[0, 0, 0] = time.time()
                fp.flush()
                del fp

    @classmethod
    def log(cls, energy: float, dominant_features, total_nodes: int,
            purged: int, path: str = HISTORY_FILE):
        cls.initialize(path)
        with _lock:
            fp = np.memmap(path, dtype="float64", mode="r+", shape=_SHAPE)
            active = np.where(fp[:, 0, 0] > 0)[0]
            nxt = (active[-1] + 1) % MAX_EPOCHS if len(active) else 0
            now = time.time()
            feats = list(dominant_features)[:FEATURE_SLOTS]
            for i in range(FEATURE_SLOTS):
                if i < len(feats):
                    fp[nxt, i, 0] = now
                    fp[nxt, i, 1] = energy
                    fp[nxt, i, 2] = float(total_nodes - purged)
                    fp[nxt, i, 3] = float(purged)
                else:
                    fp[nxt, i, :] = 0.0
            fp.flush()
            del fp
            return nxt

    @classmethod
    def recent(cls, n: int = 20, path: str = HISTORY_FILE):
        if not os.path.exists(path):
            return []
        with _lock:
            fp = np.memmap(path, dtype="float64", mode="r", shape=_SHAPE)
            active = np.where(fp[:, 0, 0] > 0)[0]
            out = []
            for idx in active[-n:]:
                out.append({
                    "epoch": int(idx),
                    "timestamp": float(fp[idx, 0, 0]),
                    "energy": float(fp[idx, 0, 1]),
                    "valid_nodes": int(fp[idx, 0, 2]),
                    "purged_nodes": int(fp[idx, 0, 3]),
                })
            del fp
            return out
