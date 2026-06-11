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
JUVIAN GRID :: LOCAL END-TO-END DEMO
Spins up several nodes in ONE process over an in-memory bus (no sockets) and
drives the full protocol:
    1. discovery
    2. strict 3-of-3 pi-Mandelbrot key verification (with chaining)
    3. distributed tensor map/reduce with a Byzantine attacker
This proves the whole stack works end to end. The same JuvianNode code runs
over real UDP via run_node.py -- only the transport changes.
==============================================================================
"""

import asyncio
import base64
import types
import numpy as np

from juvian_transport import InMemoryBus, InMemoryTransport
from juvian_node import JuvianNode
from juvian_tensor import TensorMapWorker


SECRET = "demo-session-secret"


async def main():
    bus = InMemoryBus()

    # build 5 nodes: 2 anchors + 3 mobile workers
    specs = [
        ("anchor-1", "10.0.0.1:8000", "ANCHOR", 5.0),
        ("anchor-2", "10.0.0.2:8000", "ANCHOR", 5.0),
        ("mobile-1", "10.0.0.11:8000", "MOBILE", 2.5),
        ("mobile-2", "10.0.0.12:8000", "MOBILE", 2.5),
        ("mobile-3", "10.0.0.13:8000", "MOBILE", 2.5),
    ]
    nodes = []
    for name, addr, dtype, w in specs:
        node = JuvianNode(
            name, addr, InMemoryTransport(addr, bus),
            session_secret=SECRET, device_type=dtype, hw_weight=w,
            mandelbrot_iter=400,
            history_path=f"/tmp/{name}_hist.npy",
            chain_path=f"/tmp/{name}_chain.frac",
        )
        nodes.append(node)

    for n in nodes:
        await n.start()

    print("=" * 70)
    print("PHASE 1 :: DISCOVERY")
    print("=" * 70)
    # let beacons propagate
    await asyncio.sleep(0.2)
    # nudge a round of beacons
    for n in nodes:
        await n.transport.broadcast({
            "type": "BEACON", "from": n.node_id, "from_addr": n.address,
            "name": n.name, "device_type": n.device_type, "weight": n.hw_weight,
        })
    await asyncio.sleep(0.2)
    for n in nodes:
        print(f"  {n.name:10s} sees {n.routing.count()} peers")

    print()
    print("=" * 70)
    print("PHASE 2 :: STRICT 3-OF-3 KEY VERIFICATION (chained)")
    print("=" * 70)
    initiator = nodes[0]
    for i in range(4):
        payload = f"compute-request-{i}".encode()
        result = await initiator.submit_request(payload, timeout=3.0)
        status = result.get("status")
        fp = (result.get("fingerprint") or "")[:16]
        depth = result.get("chain_depth")
        verifiers = result.get("verifiers", [])
        vshort = [v[:8] for v in verifiers]
        print(f"  request {i}: {status:9s} | fp={fp} | chain_depth={depth} | "
              f"verifiers={vshort}")
        await asyncio.sleep(0.05)

    print()
    print(f"  initiator chain depth: {initiator.chain.depth()}")
    print(f"  keys verified: {initiator.stats['keys_verified']} | "
          f"rejected: {initiator.stats['keys_rejected']}")

    print()
    print("=" * 70)
    print("PHASE 3 :: DISTRIBUTED TENSOR REDUCE (with Byzantine attacker)")
    print("=" * 70)

    # make mobile-3 Byzantine: it returns poisoned yields. It is a fully
    # legitimate, signed member -- it just computes garbage. We override its
    # tensor-task handler so the poisoned yield still goes out through the
    # node's signed send path (otherwise message auth would drop it).
    evil = nodes[4]
    rng = np.random.default_rng(7)

    async def evil_tensor_task(self, msg):
        proj_shape = msg["proj_shape"]
        tensor_shape = msg["tensor_shape"]
        shape = tuple(proj_shape[:1]) + tuple(tensor_shape[1:])
        poisoned = rng.standard_normal(shape) * 100.0
        await self._send(msg["from_addr"], {
            "type": "TENSOR_YIELD", "from": self.node_id,
            "from_addr": self.address, "job_id": msg["job_id"],
            "yield_b64": base64.b64encode(poisoned.astype(np.float64).tobytes()).decode(),
            "yield_shape": list(poisoned.shape), "weight": self.hw_weight,
        })

    evil._handle_tensor_task = types.MethodType(evil_tensor_task, evil)
    print(f"  {evil.name} is now BYZANTINE (returns signed but poisoned yields)")

    tensor = rng.standard_normal((8, 6, 6))
    result = await initiator.run_tensor_job(tensor, rank=4, timeout=3.0)
    print(f"  status        : {result['status']}")
    print(f"  Frobenius E   : {result['energy']:.2f}")
    print(f"  dominant modes: {result['dominant_modes']}")
    print(f"  valid nodes   : {[v[:8] for v in result['valid_ids']]}")
    print(f"  purged nodes  : {[p[:8] for p in result['purged']]}")
    print(f"  peak Z-score  : {result['peak_z']:.1f}")
    print(f"  report        : {result['report']}")

    evil_purged = evil.node_id in result["purged"]
    print()
    print(f"  >>> Byzantine attacker purged: {evil_purged}")

    print()
    print("=" * 70)
    print("PHASE 4 :: HISTORY LEDGER + CHAIN PERSISTENCE")
    print("=" * 70)
    from juvian_history import JuvianHistoryLedger
    recent = JuvianHistoryLedger.recent(path=initiator.history_path)
    print(f"  history ledger entries: {len(recent)}")
    if recent:
        print(f"  latest: energy={recent[-1]['energy']:.1f} "
              f"valid={recent[-1]['valid_nodes']} purged={recent[-1]['purged_nodes']}")

    initiator.persist_chain()
    from juvian_fractal_storage import FractalPersistenceManager
    recovered = FractalPersistenceManager.load_and_decompress(
        input_file=initiator.chain_path)
    print(f"  chain persisted + recovered: {len(recovered)} entries")

    print()
    print("=" * 70)
    print("NODE SNAPSHOTS")
    print("=" * 70)
    for n in nodes:
        s = n.snapshot()
        print(f"  {s['name']:10s} | peers={s['peers']} | "
              f"chain={s['chain_depth']} | verified={s['stats']['keys_verified']} | "
              f"fp={s['latest_fingerprint']}")

    for n in nodes:
        await n.stop()

    print()
    print("END-TO-END DEMO COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
