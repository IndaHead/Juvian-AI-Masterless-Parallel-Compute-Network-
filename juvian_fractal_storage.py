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
# ==============================================================================
# JUVIAN GRID: FRACTAL STORAGE PERSISTENCE — UPDATED WITH SESSION CHAIN SUPPORT
# LEGAL: Co-Authored Sovereign Substrate. Apache License 2.0 (see LICENSE).
# ==============================================================================

import os
import json
import hashlib
import numpy as np

# Default CID storage file
FRACTAL_STORAGE_FILE = "juvian_cids.frac"


class FractalPersistenceManager:
    """
    8-way Standardized Isometric Fractal Block Compression.

    Encodes CID dictionaries as 64x64 binary grids and compresses
    using contractive affine transformations (IFS fractal compression).

    Now supports custom output_file parameter for session chain storage.
    """

    @staticmethod
    def _dictionary_to_grid(cid_dict, grid_size=64):
        grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
        if not cid_dict:
            return grid
        sorted_keys = sorted(list(cid_dict.keys()))
        for idx, cid in enumerate(sorted_keys):
            if idx >= (grid_size * grid_size):
                break
            hasher = hashlib.sha256(cid.encode('utf-8')).digest()
            coord_x = hasher[0] % grid_size
            coord_y = hasher[1] % grid_size
            grid[coord_x, coord_y] = 1
        return grid

    @staticmethod
    def _grid_to_dictionary(grid, reference_cids):
        rehydrated = {}
        grid_size = grid.shape[0]
        for x in range(grid_size):
            for y in range(grid_size):
                if grid[x, y] > 0.5:
                    for cid, peer_info in reference_cids.items():
                        hasher = hashlib.sha256(cid.encode('utf-8')).digest()
                        if (hasher[0] % grid_size == x) and (hasher[1] % grid_size == y):
                            rehydrated[cid] = peer_info
        return rehydrated

    @staticmethod
    def _apply_codebook_transform(block, codebook_idx):
        """8-way isometric codebook: identity + 3 rotations + 2 flips + 2 transposes"""
        if codebook_idx == 0:   return block
        elif codebook_idx == 1: return np.rot90(block, 1)
        elif codebook_idx == 2: return np.rot90(block, 2)
        elif codebook_idx == 3: return np.rot90(block, 3)
        elif codebook_idx == 4: return np.fliplr(block)
        elif codebook_idx == 5: return np.flipud(block)
        elif codebook_idx == 6: return block.T
        elif codebook_idx == 7: return np.fliplr(block.T)
        return block

    @classmethod
    def compress_and_save(cls, cid_dict, reference_cids, tolerance=0, output_file=None):
        """
        Compress CID dictionary to fractal .frac file.

        Args:
            cid_dict:        dict of CID string → value
            reference_cids:  full metadata dict for decompression
            tolerance:       error tolerance (0 = lossless grid matching)
            output_file:     override default storage file path
        """
        target_file = output_file or FRACTAL_STORAGE_FILE
        grid = cls._dictionary_to_grid(cid_dict)
        grid_size = grid.shape[0]
        range_size = 2
        domain_size = 4
        transforms = []

        for rx in range(0, grid_size, range_size):
            for ry in range(0, grid_size, range_size):
                range_block = grid[rx:rx+range_size, ry:ry+range_size]

                # Fast path: empty block
                if np.sum(range_block) == 0:
                    transforms.append({
                        "r_pos": [rx, ry],
                        "d_pos": [0, 0],
                        "code": 0,
                        "is_clear": 1
                    })
                    continue

                best_match = None
                min_error = float('inf')

                for dx in range(0, grid_size, domain_size):
                    for dy in range(0, grid_size, domain_size):
                        domain_block = grid[dx:dx+domain_size, dy:dy+domain_size]
                        downscaled = domain_block.reshape(
                            range_size, 2, range_size, 2
                        ).mean(axis=(1, 3))
                        binarized_domain = np.where(downscaled > 0.1, 1, 0)

                        for codebook_idx in range(8):
                            mutated = cls._apply_codebook_transform(
                                binarized_domain, codebook_idx
                            )
                            error = np.sum(np.abs(range_block - mutated))
                            if error < min_error:
                                min_error = error
                                best_match = {
                                    "r_pos": [rx, ry],
                                    "d_pos": [dx, dy],
                                    "code": codebook_idx,
                                    "is_clear": 0
                                }
                        if min_error <= tolerance:
                            break
                    if min_error <= tolerance:
                        break

                if best_match:
                    transforms.append(best_match)

        payload = {
            "transforms": transforms,
            "reference_universe": reference_cids
        }

        try:
            with open(target_file, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"[FractalStorage] Saved {len(transforms)} transforms → {target_file}")
        except Exception as e:
            print(f"[FractalStorage Error] Save failed: {str(e)}")

    @classmethod
    def load_and_decompress(cls, input_file=None):
        """
        Recover the CID dictionary from a .frac file.

        HONEST NOTE: IFS fractal compression of a sparse binary CID grid is
        *lossy* and does not reliably reconstruct the original grid (the
        original "reassembles out of thin air" claim does not hold for exact
        recovery). The actual data lives losslessly in `reference_universe`,
        which we store in full -- so we recover from it directly. The fractal
        transform list is kept as a compact spatial index / visual artifact,
        not as the source of truth.
        """
        target_file = input_file or FRACTAL_STORAGE_FILE

        if not os.path.exists(target_file):
            return {}

        try:
            with open(target_file, "r") as f:
                payload = json.load(f)
        except Exception:
            return {}

        # Lossless recovery path: the reference universe is the source of truth.
        reference_universe = payload.get("reference_universe", {})
        if reference_universe:
            return dict(reference_universe)

        # Fallback: attempt the (lossy) IFS grid reconstruction.
        transforms = payload.get("transforms", [])
        grid_size = 64
        current = np.zeros((grid_size, grid_size), dtype=np.float32)
        for _ in range(8):
            nxt = np.zeros((grid_size, grid_size), dtype=np.float32)
            for t in transforms:
                rx, ry = t["r_pos"]
                if t.get("is_clear", 0) == 1:
                    nxt[rx:rx + 2, ry:ry + 2] = 0
                    continue
                dx, dy = t["d_pos"]
                dom = current[dx:dx + 4, dy:dy + 4]
                ds = dom.reshape(2, 2, 2, 2).mean(axis=(1, 3))
                nxt[rx:rx + 2, ry:ry + 2] = cls._apply_codebook_transform(ds, t["code"])
            current = nxt
        final = np.where(current > 0.3, 1, 0)
        return cls._grid_to_dictionary(final, reference_universe)
