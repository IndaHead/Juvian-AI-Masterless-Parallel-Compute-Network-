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
JUVIAN GRID :: LIVENESS / DEAD-SEQUENCER FAILOVER TESTS

Membership is now liveness-maintained: every node beacons periodically (the
beacon loop is also the maintenance tick) and any authenticated message
refreshes the sender's last_seen; a peer silent beyond PEER_EXPIRY_S is pruned
from the routing view. Because the rotating sequencer is the hash-argmin over
the LIVE membership, a dead sequencer's removal automatically recomputes the
role to the next-ranked survivor -- the deterministic fallback IS the selection
rule over the live set, with no extra election protocol.

What is proven here:
  * prune/touch semantics: silent peers expire, any authenticated traffic or
    beacon keeps a peer alive, idle-but-beaconing clusters never shrink;
  * THE HEADLINE: the current sequencer dies mid-operation; within roughly one
    expiry window every survivor drops it, all survivors agree on the same new
    sequencer (independently recomputable from public inputs over the live
    set), a request then VERIFIES end-to-end, and the live mesh stays in
    lockstep -- the previous permanent stall is now a bounded one;
  * expiry SKEW does not fork: nodes that prune the dead winner early and
    nodes that prune late converge on the same single chain;
  * the per-depth single-open guard: a second, different round for the same
    chain slot inside DEPTH_GUARD_S is refused (first-accepted-wins per
    member), narrowing the dual-candidate race a transient view split could
    otherwise open.

Honest limits (documented, not hidden): a request submitted INSIDE the expiry
window can still time out -- ordering resumes only once the ghost is pruned
(bounded by PEER_EXPIRY_S + one beacon tick). And a true network PARTITION in
which both sides retain a verifier quorum can still extend two divergent
chains; expiry-based failover narrows in-zone races, it is not partition
consensus.
==============================================================================
"""

import asyncio
import hashlib
import sys
import time
from collections import Counter

import juvian_node as jn
from juvian_transport import InMemoryBus, InMemoryTransport
from juvian_node import JuvianNode, SEQUENCER_TERM
from juvian_dht import KademliaRoutingTable

PASS = "  PASS"
FAIL = "  ** FAIL **"
_failures = 0


def check(label, cond):
    global _failures
    print(f"{PASS if cond else FAIL} {label}")
    if not cond:
        _failures += 1


async def _spin(bus, N, lookups=6):
    nodes = []
    for i in range(N):
        addr = f"10.40.{(i >> 8) & 255}.{i & 255}:9000"
        n = JuvianNode(f"fo{i}", addr, InMemoryTransport(addr, bus),
                       mandelbrot_iter=40, pow_difficulty=0,
                       history_path=f"/tmp/fo{i}.npy",
                       chain_path=f"/tmp/fo{i}.frac")
        await n.start()
        nodes.append(n)
    seed = nodes[0].address
    await asyncio.gather(*(n.bootstrap([seed], refresh=6) for n in nodes[1:]))
    for _ in range(lookups):
        await asyncio.gather(*(n._lookup(n.node_id) for n in nodes))
    return nodes


def _majority_sequencer(nodes):
    ctr = Counter(n._current_sequencer() for n in nodes)
    sid, votes = ctr.most_common(1)[0]
    assert votes >= (2 * len(nodes)) // 3, f"sequencer split: {ctr}"
    return next(n for n in nodes if n.node_id == sid)


def _expected_winner(ref, member_ids):
    """Recompute the argmin from PUBLIC inputs alone (seed, epoch, ids)."""
    depth = ref.chain.depth()
    seed = ref._epoch_seed(depth)
    epoch = (depth // SEQUENCER_TERM).to_bytes(8, "big")
    return min(member_ids, key=lambda pid: hashlib.sha256(
        seed + epoch + pid.encode()).digest())


def test_routing_liveness_unit():
    print("[liveness] routing prune/touch semantics")
    rt = KademliaRoutingTable("00" * 20)
    for i in range(6):
        rt.update(f"{i:02d}" + "ab" * 19, f"10.0.0.{i}:1")
    assert rt.count() == 6
    # age three of them past the cutoff
    aged = set()
    for b in rt.buckets:
        for p in b:
            if len(aged) < 3:
                p["last_seen"] -= 100.0
                aged.add(p["id"])
    # touch ONE aged peer back to life: any authenticated traffic counts
    revived = next(iter(aged))
    check("touch() refreshes a known peer", rt.touch(revived))
    check("touch() on an unknown peer is a no-op (False)",
          not rt.touch("ff" * 20))
    removed = set(rt.prune(50.0))
    check(f"prune drops exactly the silent peers ({len(removed)} of 2 aged, "
          f"revived one kept)",
          removed == aged - {revived} and rt.count() == 4)
    check("remove() drops a specific peer", rt.remove(revived)
          and rt.count() == 3)


async def test_dead_sequencer_failover():
    print("[failover] THE HEADLINE: sequencer dies; expiry prunes it "
          "everywhere; the next-ranked live member takes over; a request "
          "VERIFIES; no stall, no fork")
    old_exp, old_tick = jn.PEER_EXPIRY_S, jn.BEACON_INTERVAL_S
    jn.PEER_EXPIRY_S, jn.BEACON_INTERVAL_S = 4.0, 1.0
    try:
        N = 10
        bus = InMemoryBus()
        nodes = await _spin(bus, N)

        # idle-liveness pre-phase: sit beyond several expiry windows doing
        # nothing; beacons alone must keep every view complete.
        await asyncio.sleep(3 * jn.PEER_EXPIRY_S)
        complete = all(len(n.routing.peer_ids()) == N - 1 for n in nodes)
        check("idle-but-beaconing cluster never shrinks (3x expiry idle, all "
              "views still complete)", complete)

        w0 = _majority_sequencer(nodes)
        for r in range(2):
            res = await w0.submit_request(f"pre{r}".encode(), timeout=20.0)
            assert res.get("status") == "VERIFIED"
            await asyncio.sleep(0.3)

        live = [n for n in nodes if n is not w0]
        # SKEW: three survivors prune the dead winner immediately; the rest
        # discover its death only via expiry -- views must still converge on
        # one chain and one new sequencer.
        await w0.stop()
        for n in live[:3]:
            n.routing.remove(w0.node_id)

        t0 = time.time()
        while time.time() - t0 < 6 * jn.PEER_EXPIRY_S:
            await asyncio.sleep(0.4)
            if all(w0.node_id not in n.routing.peer_ids() for n in live):
                break
        gone_after = time.time() - t0
        check(f"dead sequencer expired from every survivor's view in "
              f"{gone_after:.1f}s (bound ~PEER_EXPIRY_S + tick = "
              f"{jn.PEER_EXPIRY_S + jn.BEACON_INTERVAL_S:.0f}s)",
              all(w0.node_id not in n.routing.peer_ids() for n in live)
              and gone_after <= 3 * jn.PEER_EXPIRY_S)

        expected = _expected_winner(live[0], sorted(n.node_id for n in live))
        got = {n._current_sequencer() for n in live}
        check("every survivor agrees on the SAME new sequencer", len(got) == 1)
        check("the new sequencer is the next-ranked live member, recomputable "
              "from public inputs over the survivors",
              got == {expected} and expected != w0.node_id)

        origin = next(n for n in live if n.node_id != expected)
        res = await origin.submit_request(b"post-failover", timeout=20.0)
        await asyncio.sleep(0.5)
        check(f"a request after failover is sequenced and VERIFIED "
              f"(status={res.get('status')})", res.get("status") == "VERIFIED")

        depths = {n.chain.depth() for n in live}
        digs = {n.chain.head_digest() for n in live}
        check(f"live mesh in lockstep after failover (depths={sorted(depths)}, "
              f"distinct digests={len(digs)}) -- skewed and non-skewed pruners "
              f"on ONE chain, no fork",
              len(depths) == 1 and len(digs) == 1 and depths == {3})
        check("the dead node's id never re-entered any survivor's view",
              all(w0.node_id not in n.routing.peer_ids() for n in live))
        for n in live:
            await n.stop()
    finally:
        jn.PEER_EXPIRY_S, jn.BEACON_INTERVAL_S = old_exp, old_tick


async def test_depth_guard_single_candidate():
    print("[failover] per-depth single-CANDIDATE guard: a different node's "
          "round for an already-held slot is refused inside DEPTH_GUARD_S; "
          "same-sequencer retries pass; the slot unblocks for a successor")
    old_exp, old_tick, old_guard = (jn.PEER_EXPIRY_S, jn.BEACON_INTERVAL_S,
                                    jn.DEPTH_GUARD_S)
    jn.PEER_EXPIRY_S, jn.BEACON_INTERVAL_S = 30.0, 5.0   # quiet defaults
    try:
        import base64 as _b64
        N = 6
        bus = InMemoryBus()
        nodes = await _spin(bus, N)
        seq = _majority_sequencer(nodes)
        victim = next(n for n in nodes if n is not seq)

        def open_msg(from_node, session, payload):
            return {"from": from_node.node_id, "from_addr": from_node.address,
                    "session_id": session,
                    "payload_b64": _b64.b64encode(payload).decode(),
                    "verifiers": [victim.node_id], "iter": 40,
                    "chain_index": victim.chain.depth()}

        # (1) the sanctioned sequencer opens the slot
        await victim._handle_verify_open(open_msg(seq, "slotA", b"cand-A"))
        check("first candidate's round is open at the member",
              "slotA" in victim.verifier.rounds)

        # (2) RETRY-compat: the SAME sequencer re-opens the same slot with a
        # fresh session (exactly what the quorum-retry path does) -- allowed.
        await victim._handle_verify_open(open_msg(seq, "slotA2", b"cand-A"))
        check("the SAME sequencer re-opening the slot (quorum retry) passes",
              "slotA2" in victim.verifier.rounds)

        # (3) the victim's view flips mid-slot (simulated expiry of the old
        # sequencer): a DIFFERENT candidate -- now the victim's computed
        # sequencer -- races the same slot within the guard window: REFUSED.
        victim.routing.remove(seq.node_id)
        successor = next(n for n in nodes
                         if n.node_id == victim._current_sequencer())
        assert successor.node_id != seq.node_id
        await victim._handle_verify_open(open_msg(successor, "slotB", b"cand-B"))
        check("a DIFFERENT candidate for the same held slot is refused "
              "(first-accepted-candidate-wins)",
              "slotB" not in victim.verifier.rounds)

        # (4) after DEPTH_GUARD_S the dead candidate's claim lapses and the
        # legitimate successor retakes the slot.
        jn.DEPTH_GUARD_S = 0.5
        await asyncio.sleep(0.6)
        await victim._handle_verify_open(open_msg(successor, "slotB", b"cand-B"))
        check("after DEPTH_GUARD_S the successor retakes the slot",
              "slotB" in victim.verifier.rounds)
        for n in nodes:
            await n.stop()
    finally:
        jn.PEER_EXPIRY_S, jn.BEACON_INTERVAL_S, jn.DEPTH_GUARD_S = (
            old_exp, old_tick, old_guard)


def main():
    test_routing_liveness_unit()
    asyncio.run(test_dead_sequencer_failover())
    asyncio.run(test_depth_guard_single_candidate())
    if _failures == 0:
        print("\nALL FAILOVER TESTS PASSED")
    else:
        print(f"\n{_failures} CHECK(S) FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
