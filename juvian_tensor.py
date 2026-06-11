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
JUVIAN GRID :: TENSOR ENGINE
Distributed map/reduce over high-dimensional arrays with:
  * coordinate-wise MAD Byzantine filter (drop |Z| > 4.5)
  * weighted consensus
  * SVD energy extraction for dominant latent modes
==============================================================================
"""

import numpy as np
from typing import Dict, List, Tuple


MAD_Z_THRESHOLD = 4.5
EPS = 1e-6
ENERGY_MODE_FRACTION = 0.15   # a mode is "dominant" if S^2 > 15% of total


class TensorMapWorker:
    """The map phase that runs on an edge node. Projects a tensor slice
    through a projection matrix, throttling when hardware weight is low."""

    @staticmethod
    def map_slice(tensor_slice: np.ndarray, projection: np.ndarray,
                  hw_weight: float = 1.0) -> np.ndarray:
        if hw_weight <= 0.0:
            return np.zeros((projection.shape[0],) + tensor_slice.shape[1:])
        # mode-1 product: project along the first axis
        I, J, K = tensor_slice.shape
        R = projection.shape[0]
        out = np.zeros((R, J, K))
        for j in range(J):
            for k in range(K):
                out[:, j, k] = projection @ tensor_slice[:, j, k]
        return out


class TensorReducer:
    """The reduce phase that runs on an anchor. Filters Byzantine yields with
    a coordinate-wise MAD gate, then forms a weighted consensus and extracts
    SVD energy."""

    @staticmethod
    def mad_filter(yields: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return a boolean mask of non-Byzantine rows and the peak per-node
        score. yields: (n_nodes, ...) stack.

        IMPORTANT correction vs the original Juvian design: we aggregate each
        node's coordinate-wise Z scores with the MEDIAN, not the MAX. Taking
        the max across thousands of coordinates trips the threshold for honest
        nodes by chance (a multiple-comparisons problem) and purges them. A
        Byzantine node corrupts a large *fraction* of its coordinates, so the
        median Z separates honest from poisoned cleanly."""
        n = yields.shape[0]
        if n < 3:
            return np.ones(n, dtype=bool), 0.0
        median = np.median(yields, axis=0)
        mad = np.median(np.abs(yields - median), axis=0) + EPS
        z = np.abs(yields - median) / mad
        axes = tuple(range(1, z.ndim))
        node_score = np.median(z, axis=axes)   # robust per-node aggregate
        mask = node_score <= MAD_Z_THRESHOLD
        peak = float(np.max(node_score)) if node_score.size else 0.0
        return mask, peak

    @classmethod
    def reduce(cls, node_yields: Dict[str, np.ndarray],
               weights: Dict[str, float]) -> dict:
        if not node_yields:
            return {"status": "EMPTY", "energy": 0.0, "report": "no inputs"}

        ids = list(node_yields.keys())
        stack = np.array([node_yields[i] for i in ids], dtype=np.float64)

        mask, peak_z = cls.mad_filter(stack)
        purged = [ids[i] for i in range(len(ids)) if not mask[i]]
        valid_ids = [ids[i] for i in range(len(ids)) if mask[i]]

        if not valid_ids:
            return {"status": "COMPROMISED", "energy": 0.0,
                    "purged": purged, "peak_z": peak_z,
                    "report": "all nodes failed MAD gate"}

        # weighted consensus
        wsum = sum(weights.get(i, 1.0) for i in valid_ids) + EPS
        consensus = np.zeros_like(stack[0])
        for i in valid_ids:
            consensus += node_yields[i] * weights.get(i, 1.0)
        consensus /= wsum

        # SVD energy on the mode-1 unfolding (sum across last axis)
        mat = np.sum(consensus, axis=tuple(range(2, consensus.ndim))) \
            if consensus.ndim > 2 else consensus
        U, Svals, Vt = np.linalg.svd(np.atleast_2d(mat), full_matrices=False)
        energy = float(np.sum(Svals ** 2))
        dominant = np.where(Svals ** 2 > energy * ENERGY_MODE_FRACTION)[0].tolist()

        return {
            "status": "SUCCESS",
            "energy": energy,
            "singular_values": Svals.tolist(),
            "dominant_modes": dominant,
            "valid_ids": valid_ids,
            "purged": purged,
            "peak_z": peak_z,
            "consensus_shape": list(consensus.shape),
            "report": f"converged | energy={energy:.2f} | "
                      f"valid={len(valid_ids)}/{len(ids)} | purged={len(purged)}",
        }
