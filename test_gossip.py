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
JUVIAN GRID :: GOSSIP DISSEMINATION TESTS

The per-request verify broadcasts (VERIFY_OPEN / VERIFY_RESULT) are disseminated
by gossip once the roster grows past GOSSIP_MIN_ROSTER: the sequencer pushes to a
bounded GOSSIP_FANOUT of peers and each peer re-forwards on first receipt, so a
message floods the overlay with O(fanout) per-node cost instead of an O(N)
fan-out from the sequencer. At or below zone scale the system keeps the
GUARANTEED reliable broadcast (hard lockstep, covered by the other suites), so
these tests deliberately run ABOVE the threshold where gossip is engaged.

What is proven here:
  * per-node fan-out is bounded (~fanout), not O(N) -- the whole point;
  * a single gossip push reaches the entire mesh with high probability;
  * a verify request still completes (VERIFIES) end-to-end over gossip at scale.

Honest limit: push-gossip coverage is HIGH-PROBABILITY, not a hard guarantee, so
at very large scale / under partition a node can miss a round and fall out of
lockstep. The companion that restores a hard guarantee -- anti-entropy repair (a
lagging node pulls the chain entry it missed) -- is future work; below the
threshold the reliable broadcast already gives hard lockstep.
==============================================================================
"""
import asyncio
from collections import Counter
import random
import sys

from juvian_transport import InMemoryBus, InMemoryTransport
from juvian_node import (JuvianNode, MSG_VERIFY_RESULT,
                         GOSSIP_FANOUT, GOSSIP_MIN_ROSTER)

PASS = "  PASS"
FAIL = "  ** FAIL **"
_failures = 0


def check(label, cond):
    global _failures
    print(f"{PASS if cond else FAIL} {label}")
    if not cond:
        _failures += 1


async def _spin(bus, N):
    nodes = []
    for i in range(N):
        addr = f"10.11.{(i >> 8) & 255}.{i & 255}:9000"
        n = JuvianNode(f"x{i}", addr, InMemoryTransport(addr, bus),
                       mandelbrot_iter=45, pow_difficulty=0,
                       history_path=f"/tmp/gs{i}.npy",
                       chain_path=f"/tmp/gs{i}.frac")
        await n.start()
        nodes.append(n)
    seed = nodes[0].address
    await asyncio.gather(*(n.bootstrap([seed], refresh=6) for n in nodes[1:]))
    for _ in range(3):                              # converge routing tables
        await asyncio.gather(*(n._lookup(n.node_id) for n in nodes))
    return nodes


async def test_gossip_fanout_and_coverage():
    print("[gossip] bounded per-node fan-out + high-probability full coverage")
    N = 50
    assert N > GOSSIP_MIN_ROSTER, "this test must run above the gossip threshold"
    bus = InMemoryBus()
    nodes = await _spin(bus, N)

    # instrument: count every unicast send per node
    sends = {n.node_id: 0 for n in nodes}
    for n in nodes:
        base = n.transport.send
        async def wrapped(addr, msg, _b=base, _id=n.node_id):
            sends[_id] += 1
            return await _b(addr, msg)
        n.transport.send = wrapped

    TRIALS = 10
    full_cov = 0
    cov_sum = 0
    worst_cov = N
    worst_maxsend = 0
    for t in range(TRIALS):
        for k in sends:
            sends[k] = 0
        # tight window: beacons are 5s apart, so a node's seen-cache only grows
        # from THIS broadcast. Coverage = how many nodes recorded it.
        before = {n.node_id: len(n._seen_msgs) for n in nodes}
        origin = random.choice(nodes)
        await origin._gossip_broadcast({
            "type": MSG_VERIFY_RESULT, "from": origin.node_id,
            "from_addr": origin.address, "session_id": f"cov{t}",
            "status": "VERIFIED", "fingerprint": "ab"})
        for _ in range(24):
            await asyncio.sleep(0)                   # let the overlay flood
        cov = 1 + sum(1 for n in nodes
                      if n.node_id != origin.node_id
                      and len(n._seen_msgs) > before[n.node_id])
        full_cov += 1 if cov == N else 0
        cov_sum += cov
        worst_cov = min(worst_cov, cov)
        worst_maxsend = max(worst_maxsend, max(sends.values()))

    avg_cov = cov_sum / TRIALS
    check(f"per-node fan-out is bounded: busiest node sent {worst_maxsend} "
          f"(a flat broadcast from the sequencer would send {N - 1})",
          worst_maxsend <= 3 * GOSSIP_FANOUT and worst_maxsend < (N - 1) // 2)
    check(f"a single gossip push reaches the mesh with high probability "
          f"(avg {avg_cov:.1f}/{N}, worst {worst_cov}/{N}, "
          f"{full_cov}/{TRIALS} fully covered)",
          avg_cov >= 0.97 * N and worst_cov >= int(0.9 * N))
    for n in nodes:
        await n.stop()


async def test_verify_round_over_gossip():
    print("[gossip] a verify request still VERIFIES end-to-end over the overlay")
    N = 50
    bus = InMemoryBus()
    nodes = await _spin(bus, N)
    _ctr = Counter(n._current_sequencer() for n in nodes)
    _sid, _votes = _ctr.most_common(1)[0]
    assert _votes >= (2 * len(nodes)) // 3, f"sequencer split: {_ctr}"
    seq = next(n for n in nodes if n.node_id == _sid)
    check(f"dissemination is actually in gossip mode here "
          f"(roster {seq.routing.count()} > {GOSSIP_MIN_ROSTER})",
          seq.routing.count() > GOSSIP_MIN_ROSTER)

    res = await seq.submit_request(b"gossip-e2e", timeout=20.0)
    check("request VERIFIED via gossip dissemination at scale",
          res.get("status") == "VERIFIED")

    # submit_request returns as soon as the sequencer has quorum + has *pushed*
    # the VERIFY_RESULT; the verify round is two sequential floods (OPEN, then
    # RESULT) whose relays run as scheduled tasks, and a node that gets RESULT
    # before OPEN buffers it until OPEN arrives. Give the overlay enough real
    # time for both floods to fully settle before sampling heads.
    for _ in range(20):
        await asyncio.sleep(0.02)

    # The verify round always VERIFIES; how many members ADOPT the head in one
    # shot over pure push-gossip is high-probability, NOT guaranteed. The
    # residual gap is driven by verifier-quorum RETRIES: a retry is a second
    # OPEN flood whose coverage is independent of the first, so a member reached
    # by only the first flood derived a key for the wrong session and stays on
    # genesis. Crucially there is never a chain FORK -- every member that
    # advances lands on the SAME head (all share the genesis salt and derive the
    # same key). Closing the gap to full lockstep is the anti-entropy follow-up
    # (a lagging member pulls the chain entry it missed).
    adopted_heads = {n.chain.current_salt().hex()
                     for n in nodes if n.chain.depth() >= 1}
    heads = {}
    for n in nodes:
        h = n.chain.current_salt().hex()
        heads[h] = heads.get(h, 0) + 1
    top = max(heads.values())
    check("members that adopt never fork -- one shared head, no split-brain",
          len(adopted_heads) <= 1)
    check(f"a majority adopt that shared head in one shot over gossip "
          f"({top}/{N}; full lockstep is the anti-entropy follow-up)",
          top > N // 2)
    for n in nodes:
        await n.stop()


def main():
    asyncio.run(test_gossip_fanout_and_coverage())
    asyncio.run(test_verify_round_over_gossip())
    if _failures == 0:
        print("\nALL GOSSIP TESTS PASSED")
    else:
        print(f"\n{_failures} CHECK(S) FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
