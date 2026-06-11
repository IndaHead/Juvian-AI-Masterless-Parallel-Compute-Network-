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
JUVIAN GRID :: KADEMLIA ROUTING TABLE
160-bit XOR namespace, 160 k-buckets. Standard Kademlia bucketing with LRU
refresh.

Bucket capacity (k=48) is deliberately sized for the system's stated operating
envelope -- a ZONE of <= ~40-50 nodes -- so that every member's view of the
zone is COMPLETE (the far-half bucket holds ~N/2 peers; k=48 keeps that
complete up to roughly N~96). Complete zone views are load-bearing well beyond
routing: verifier selection, quorum attestation counting, and verifiable
sequencer rotation all assume members agree on the zone membership. The
classic k=20 is tuned for million-node DHTs where buckets MUST be partial;
at that scale this design federates across zones rather than growing one flat
mesh, and partial views are the documented boundary, not a silent failure.
Used both for routing and for deterministic verifier selection.
==============================================================================
"""

import time
import hashlib
import threading
from typing import Dict, List, Optional


def node_id_from_seed(seed: str) -> str:
    """Deterministic 160-bit (40 hex char) node id from any seed string."""
    return hashlib.sha1(seed.encode()).hexdigest()


class KademliaRoutingTable:
    def __init__(self, local_id: str, k: int = 48):
        self.local_id = local_id
        self.local_int = int(local_id[:40], 16)
        self.k = k
        self.buckets: List[List[dict]] = [[] for _ in range(160)]
        self._lock = threading.Lock()

    def _bucket_index(self, peer_id: str) -> int:
        xor = self.local_int ^ int(peer_id[:40], 16)
        return 0 if xor == 0 else xor.bit_length() - 1

    def update(self, peer_id: str, address: str,
               nat_type: str = "UNKNOWN", weight: float = 1.0,
               device_type: str = "UNKNOWN", geo_zone: str = "UNKNOWN"):
        if peer_id == self.local_id:
            return
        idx = self._bucket_index(peer_id)
        with self._lock:
            b = self.buckets[idx]
            existing = next((p for p in b if p["id"] == peer_id), None)
            if existing:
                b.remove(existing)
                existing.update(address=address, nat_type=nat_type,
                                weight=weight, device_type=device_type,
                                geo_zone=geo_zone, last_seen=time.time())
                b.append(existing)
            elif len(b) < self.k:
                b.append({"id": peer_id, "address": address,
                          "nat_type": nat_type, "weight": weight,
                          "device_type": device_type, "geo_zone": geo_zone,
                          "last_seen": time.time()})
            else:
                b.sort(key=lambda p: p["last_seen"])
                if time.time() - b[0]["last_seen"] > 60:
                    b.pop(0)
                    b.append({"id": peer_id, "address": address,
                              "nat_type": nat_type, "weight": weight,
                              "device_type": device_type, "geo_zone": geo_zone,
                              "last_seen": time.time()})

    def set_weight(self, peer_id: str, weight: float) -> bool:
        """Adjust a known peer's weight in place (preserving its other fields).
        Used for cooperative thermal/battery down-weighting."""
        with self._lock:
            for b in self.buckets:
                for p in b:
                    if p["id"] == peer_id:
                        p["weight"] = weight
                        return True
        return False

    def touch(self, peer_id: str) -> bool:
        """Refresh a known peer's last_seen. Called for every authenticated
        message a node processes, so any live protocol traffic -- not only
        beacons -- counts as evidence of liveness. Returns False if unknown."""
        now = time.time()
        with self._lock:
            for b in self.buckets:
                for p in b:
                    if p["id"] == peer_id:
                        p["last_seen"] = now
                        return True
        return False

    def remove(self, peer_id: str) -> bool:
        """Physically remove a peer from the table (e.g. proven dead)."""
        with self._lock:
            for b in self.buckets:
                for p in b:
                    if p["id"] == peer_id:
                        b.remove(p)
                        return True
        return False

    def prune(self, max_age_s: float) -> List[str]:
        """Liveness-based view maintenance: drop every peer not heard from
        (beacon or any authenticated message) within `max_age_s`. This is what
        makes membership reflect reality -- a departed node leaves every honest
        member's view within roughly one expiry window, so roles derived from
        membership (the rotating sequencer above all) recompute to a live
        member instead of stalling on a ghost. Returns the removed ids."""
        cutoff = time.time() - max_age_s
        removed: List[str] = []
        with self._lock:
            for b in self.buckets:
                stale = [p for p in b if p["last_seen"] < cutoff]
                for p in stale:
                    b.remove(p)
                    removed.append(p["id"])
        return removed

    def all_peers(self) -> Dict[str, dict]:
        with self._lock:
            return {p["id"]: dict(p) for b in self.buckets for p in b}

    def peer_ids(self) -> List[str]:
        with self._lock:
            return [p["id"] for b in self.buckets for p in b]

    def closest(self, target_hex: str, count: int = 3) -> List[dict]:
        target = int(target_hex[:40], 16)
        with self._lock:
            peers = [p for b in self.buckets for p in b]
        peers.sort(key=lambda p: target ^ int(p["id"][:40], 16))
        return peers[:count]

    def address_of(self, peer_id: str) -> Optional[str]:
        with self._lock:
            for b in self.buckets:
                for p in b:
                    if p["id"] == peer_id:
                        return p["address"]
        return None

    def count(self) -> int:
        with self._lock:
            return sum(len(b) for b in self.buckets)
