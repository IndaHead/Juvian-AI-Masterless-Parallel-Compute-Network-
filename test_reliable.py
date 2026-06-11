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
JUVIAN GRID :: RELIABLE-TRANSPORT TEST SUITE
==============================================================================
Proves the chunking + ARQ layer (audit 2.4 / 2.5) actually works: large
messages survive fragmentation, and messages survive heavy datagram loss,
duplication, and reordering via retransmission -- including the full node
tensor path end-to-end over a lossy network.

Run: python3 test_reliable.py
==============================================================================
"""

import asyncio
from collections import Counter
import numpy as np

from juvian_reliable import (
    ReliableDatagram, LossyDatagramBus, ReliableInMemoryTransport,
)

PASS, FAIL = "  PASS", "  ** FAIL **"


def check(label, cond):
    print((PASS if cond else FAIL), label)
    assert cond, label


def _pair(bus, **kw):
    """Two reliable endpoints 'A' and 'B' over the given bus."""
    recv = {"A": [], "B": []}
    rdts = {}
    for name in ("A", "B"):
        def mk(n):
            return lambda addr, data: recv[n].append(data)
        rdt = ReliableDatagram(bus.send_raw_for(name), mk(name), **kw)
        bus.register(name, rdt.feed)
        rdt.start()
        rdts[name] = rdt
    return rdts, recv


async def _wait(cond, timeout=15.0, step=0.02):
    t = 0.0
    while t < timeout:
        if cond():
            return True
        await asyncio.sleep(step)
        t += step
    return cond()


# ---------------------------------------------------------------------------
async def test_large_message_no_loss():
    print("[2.4] large message is fragmented and reassembled")
    bus = LossyDatagramBus(loss=0.0)
    rdts, recv = _pair(bus, rto=0.05)
    payload = bytes(np.random.RandomState(1).bytes(100_000))   # 100 KB
    await rdts["A"].send("B", payload)
    ok = await _wait(lambda: len(recv["B"]) == 1)
    check("100 KB message delivered (exceeds any single datagram)", ok)
    check("reassembled bytes are identical", recv["B"][0] == payload)
    for r in rdts.values():
        await r.stop()


async def test_survives_heavy_loss():
    print("[2.5] retransmission recovers from heavy loss")
    bus = LossyDatagramBus(loss=0.30, seed=42)
    rdts, recv = _pair(bus, rto=0.04, max_retries=60)
    payload = bytes(np.random.RandomState(2).bytes(40_000))    # ~37 chunks
    await rdts["A"].send("B", payload)
    ok = await _wait(lambda: len(recv["B"]) == 1, timeout=20.0)
    check("message delivered intact despite 30% datagram loss", ok)
    check("content correct after retransmits", recv["B"] and recv["B"][0] == payload)
    print(f"     (network dropped {bus.dropped}/{bus.sent} datagrams)")
    for r in rdts.values():
        await r.stop()


async def test_dedup_and_reorder():
    print("[2.5] duplicates/reordering -> each message delivered exactly once")
    bus = LossyDatagramBus(loss=0.1, dup=0.25, reorder=0.4, seed=7)
    rdts, recv = _pair(bus, rto=0.04, max_retries=60)
    msgs = [f"message-number-{i}".encode() * 50 for i in range(20)]
    for m in msgs:
        await rdts["A"].send("B", m)
    ok = await _wait(lambda: len(recv["B"]) == 20, timeout=20.0)
    check("all 20 messages delivered", ok)
    check("no duplicate deliveries (dedup works)", len(recv["B"]) == 20)
    check("every message arrived intact", sorted(recv["B"]) == sorted(msgs))
    for r in rdts.values():
        await r.stop()


async def test_broadcast_chunking():
    print("[2.4] best-effort broadcast is chunked and reassembled (no loss)")
    bus = LossyDatagramBus(loss=0.0)
    rdts, recv = _pair(bus, rto=0.05)
    payload = bytes(np.random.RandomState(3).bytes(8_000))     # multi-chunk
    await rdts["A"].send("B", payload, reliable=False)         # NOACK path
    ok = await _wait(lambda: len(recv["B"]) == 1)
    check("multi-chunk broadcast reassembled", ok and recv["B"][0] == payload)
    for r in rdts.values():
        await r.stop()


async def test_tensor_over_lossy_network():
    print("[2.4/2.5] full node tensor job over a 20%-loss network")
    from juvian_node import JuvianNode
    bus = LossyDatagramBus(loss=0.20, dup=0.05, reorder=0.15, seed=11)
    a_addr, b_addr = "10.0.0.1:9000", "10.0.0.2:9000"
    A = JuvianNode("A", a_addr,
                   ReliableInMemoryTransport(a_addr, bus, rto=0.04, max_retries=80),
                   history_path="/tmp/relA.npy", chain_path="/tmp/relA.frac")
    B = JuvianNode("B", b_addr,
                   ReliableInMemoryTransport(b_addr, bus, rto=0.04, max_retries=80),
                   history_path="/tmp/relB.npy", chain_path="/tmp/relB.frac")
    for n in (A, B):
        await n.start()
    # wire routing directly so the test isolates the bulk-transport behaviour
    A.routing.update(B.node_id, B.address)
    B.routing.update(A.node_id, A.address)

    tensor = np.random.RandomState(0).standard_normal((48, 25, 8))  # 3D, ~77 KB payload
    res = await A.run_tensor_job(tensor, rank=4, timeout=25.0)
    check("distributed tensor job completed over a lossy network",
          res.get("status") == "SUCCESS")
    check("worker B actually contributed (2 yields reduced: local + remote)",
          len(res.get("valid_ids", [])) + len(res.get("purged", [])) >= 2)
    print(f"     (network dropped {bus.dropped}/{bus.sent} datagrams; job still succeeded)")
    for n in (A, B):
        await n.stop()


async def test_kex_over_lossy_network():
    print("[reliable-bcast] group key agreement (KEX) survives a 20%-loss network")
    from juvian_node import JuvianNode
    bus = LossyDatagramBus(loss=0.20, dup=0.05, reorder=0.15, seed=23)
    addrs = [f"10.0.7.{i+1}:9000" for i in range(3)]
    nodes = [JuvianNode(f"k{i}", addrs[i],
                        ReliableInMemoryTransport(addrs[i], bus, rto=0.04,
                                                  max_retries=120),
                        mandelbrot_iter=80, pow_difficulty=0,
                        history_path=f"/tmp/rk{i}.npy", chain_path=f"/tmp/rk{i}.frac")
             for i in range(3)]
    for n in nodes:
        await n.start()
    # wire the roster directly so the test isolates reliable-broadcast behaviour
    for a in nodes:
        for b in nodes:
            if a is not b:
                a.routing.update(b.node_id, b.address)

    initiator = min(nodes, key=lambda n: n.node_id)
    await initiator.establish_session(timeout=25.0)
    await asyncio.sleep(1.0)        # let every member install the group salt

    sources = {n.name: n.genesis_source for n in nodes}
    salts = {n.chain.current_salt().hex() for n in nodes}
    check("every member upgraded to a group-DH genesis salt despite loss",
          all(s == "GROUP_DH" for s in sources.values()))
    check("every member computed the SAME secret genesis salt (no split)",
          len(salts) == 1)
    print(f"     (network dropped {bus.dropped}/{bus.sent} datagrams; KEX still agreed)")
    for n in nodes:
        await n.stop()


async def test_verify_lockstep_over_lossy_network():
    print("[reliable-bcast] verify rounds keep all members in lockstep over 30% loss")
    from juvian_node import JuvianNode
    bus = LossyDatagramBus(loss=0.30, dup=0.05, reorder=0.15, seed=29)
    addrs = [f"10.0.8.{i+1}:9000" for i in range(3)]
    nodes = [JuvianNode(f"v{i}", addrs[i],
                        ReliableInMemoryTransport(addrs[i], bus, rto=0.04,
                                                  max_retries=160),
                        mandelbrot_iter=80, pow_difficulty=0,
                        history_path=f"/tmp/rv{i}.npy", chain_path=f"/tmp/rv{i}.frac")
             for i in range(3)]
    for n in nodes:
        await n.start()
    for a in nodes:
        for b in nodes:
            if a is not b:
                a.routing.update(b.node_id, b.address)

    _ctr = Counter(n._current_sequencer() for n in nodes)
    _sid, _votes = _ctr.most_common(1)[0]
    assert _votes >= (2 * len(nodes)) // 3, f"sequencer split: {_ctr}"
    seq = next(n for n in nodes if n.node_id == _sid)
    ROUNDS = 3
    verified = 0
    for r in range(ROUNDS):
        res = await seq.submit_request(f"lockstep-{r}".encode(), timeout=25.0)
        verified += 1 if res.get("status") == "VERIFIED" else 0
    await asyncio.sleep(1.0)        # let the last VERIFY_RESULT advance every chain

    check(f"all {ROUNDS} requests VERIFIED despite loss (quorum reached reliably)",
          verified == ROUNDS)
    depths = {n.chain.depth() for n in nodes}
    heads = {n.chain.current_salt().hex() for n in nodes}
    check("all members advanced to the SAME chain depth (lockstep, no desync)",
          len(depths) == 1 and depths.pop() == ROUNDS)
    check("all members hold the SAME chain head (adopted the same keys)",
          len(heads) == 1)
    print(f"     (network dropped {bus.dropped}/{bus.sent} datagrams; "
          f"chain stayed in lockstep across {ROUNDS} rounds)")
    for n in nodes:
        await n.stop()


def main():
    print("=" * 60)
    print("JUVIAN GRID :: RELIABLE-TRANSPORT TEST SUITE")
    print("=" * 60)
    asyncio.run(test_large_message_no_loss())
    asyncio.run(test_survives_heavy_loss())
    asyncio.run(test_dedup_and_reorder())
    asyncio.run(test_broadcast_chunking())
    asyncio.run(test_tensor_over_lossy_network())
    asyncio.run(test_kex_over_lossy_network())
    asyncio.run(test_verify_lockstep_over_lossy_network())
    print("=" * 60)
    print("ALL RELIABLE-TRANSPORT TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
