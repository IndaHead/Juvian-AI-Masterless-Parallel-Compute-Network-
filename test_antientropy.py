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
JUVIAN GRID :: ANTI-ENTROPY REPAIR TESTS

Gossip dissemination (above GOSSIP_MIN_ROSTER) reaches the mesh with high
probability but not certainty, so a member can miss a round and fall behind.
Anti-entropy repair closes that gap: a lagging member pulls the chain entries it
missed from the SEQUENCER and re-derives + re-confirms each locally, so the mesh
converges to full lockstep instead of leaving a permanent tail.

Trust model: a pulled slot is RE-DERIVED from the member's own salt and adopted
only if the member's independently-computed fingerprint matches the one the round
committed with. On its own that is self-consistent but NOT canonical -- an insider
that knows the genesis salt could serve a fully forged-yet-self-consistent chain.
So adoption is additionally anchored to a verifiable HEAD DIGEST (a cumulative
hash chain over the ordered (index, fingerprint) pairs): a batch from ANY member
is adopted only up to the longest prefix whose re-derived head digest is vouched
for by a QUORUM of signed beacons (or the current sequencer's beacon). A batch
from the current sequencer is still trusted directly for backward compatibility.

Because EVERY member now archives the rounds it adopts and ANY member may serve,
repair survives a sequencer handover / failure -- it no longer depends on the one
original sequencer that happened to archive.

What is proven here:
  * a member deterministically driven behind catches up to the exact same head;
  * catch-up is digest-anchored: genuine rounds from a non-sequencer are refused
    with no anchor, a tampered payload is refused, a SELF-CONSISTENT forged
    sequence (right per-slot fingerprints, wrong head digest) is refused, and the
    genuine rounds ARE adopted once quorum-anchored;
  * FAILOVER: when the original (archiving) sequencer cannot serve, a laggard
    still converges to the canonical head from a non-sequencer, anchored by the
    quorum head digest;
  * across a large gossip mesh, repair drives the network to full lockstep.

Honest limit: a member whose partial routing table does not yet include enough of
the roster can neither identify the sequencer nor gather a quorum of attestations,
and stays behind until discovery fills that in -- a routing-completeness /
partial-view concern, separate from the repair mechanism.
==============================================================================
"""
import asyncio
from collections import Counter
import base64
import time

from juvian_transport import InMemoryBus, InMemoryTransport
from juvian_node import (JuvianNode, GOSSIP_MIN_ROSTER, REQUIRED_QUORUM,
                         PiMandelbrotKeyEngine)

PASS = "  PASS"
FAIL = "  ** FAIL **"
_failures = 0


def check(label, cond):
    global _failures
    print(f"{PASS if cond else FAIL} {label}")
    if not cond:
        _failures += 1


async def _spin(bus, N, lookups=5):
    nodes = []
    for i in range(N):
        addr = f"10.23.{(i >> 8) & 255}.{i & 255}:9000"
        n = JuvianNode(f"ae{i}", addr, InMemoryTransport(addr, bus),
                       mandelbrot_iter=40, pow_difficulty=0,
                       history_path=f"/tmp/ae{i}.npy",
                       chain_path=f"/tmp/ae{i}.frac")
        await n.start()
        nodes.append(n)
    seed = nodes[0].address
    await asyncio.gather(*(n.bootstrap([seed], refresh=6) for n in nodes[1:]))
    for _ in range(lookups):                       # converge routing tables
        await asyncio.gather(*(n._lookup(n.node_id) for n in nodes))
    return nodes


async def _settle(iters=25, dt=0.02):
    for _ in range(iters):
        await asyncio.sleep(dt)


async def test_catchup_heals_induced_lag():
    print("[anti-entropy] a member driven behind catches up to the exact head")
    N = 40
    assert N > GOSSIP_MIN_ROSTER
    bus = InMemoryBus()
    nodes = await _spin(bus, N)
    _ctr = Counter(n._current_sequencer() for n in nodes)
    _sid, _votes = _ctr.most_common(1)[0]
    assert _votes >= (2 * len(nodes)) // 3, f"sequencer split: {_ctr}"
    seq = next(n for n in nodes if n.node_id == _sid)
    # a laggard that is NOT the sequencer and that knows the sequencer (so it can
    # both recognize and address it -- isolating the mechanism from routing gaps)
    laggard = next(n for n in nodes
                   if n is not seq
                   and n._current_sequencer() == seq.node_id)

    # Deterministically hold it back: ignore verify gossip AND suppress its own
    # catch-up, so it cannot advance by any path during the lag phase.
    async def _noop(_msg):
        return
    real_open = laggard._handle_verify_open
    real_result = laggard._handle_verify_result
    real_catchup = laggard._maybe_request_catchup
    laggard._handle_verify_open = _noop
    laggard._handle_verify_result = _noop
    laggard._maybe_request_catchup = _noop

    ROUNDS = 3
    for r in range(ROUNDS):
        await seq.submit_request(f"r{r}".encode(), timeout=20.0)
        await _settle()

    check(f"lag induced: peers advanced to depth {seq.chain.depth()} while the "
          f"laggard is held at depth {laggard.chain.depth()}",
          seq.chain.depth() == ROUNDS and laggard.chain.depth() == 0)

    # Restore normal behaviour and let a discovery beacon reveal the gap.
    laggard._handle_verify_open = real_open
    laggard._handle_verify_result = real_result
    laggard._maybe_request_catchup = real_catchup
    for _ in range(3):
        await asyncio.gather(*(n.announce() for n in nodes))
        await _settle(30)

    check(f"laggard caught up to the head via repair "
          f"(depth {laggard.chain.depth()} == sequencer {seq.chain.depth()})",
          laggard.chain.depth() == seq.chain.depth())
    check(f"catch-up actually applied pulled rounds "
          f"(keys_caught_up={laggard.stats.get('keys_caught_up', 0)})",
          laggard.stats.get("keys_caught_up", 0) >= 1)
    # and it landed on the SAME head, not a fork
    check("caught-up head matches the sequencer's head (no fork)",
          laggard.chain.current_salt() == seq.chain.current_salt())
    for n in nodes:
        await n.stop()


async def test_catchup_anchoring_rejects_forgery():
    print("[anti-entropy] catch-up is anchored to a verifiable head digest: a "
          "forged sequence from any server is refused, a genuine one is adopted")
    N = 36
    bus = InMemoryBus()
    nodes = await _spin(bus, N)
    _ctr = Counter(n._current_sequencer() for n in nodes)
    _sid, _votes = _ctr.most_common(1)[0]
    assert _votes >= (2 * len(nodes)) // 3, f"sequencer split: {_ctr}"
    seq = next(n for n in nodes if n.node_id == _sid)
    impostor = max(nodes, key=lambda n: n.node_id)   # a NON-sequencer server

    laggard = next(n for n in nodes
                   if n is not seq and n is not impostor
                   and n._current_sequencer() == seq.node_id)

    async def _noop(_msg):
        return
    real_open = laggard._handle_verify_open
    real_result = laggard._handle_verify_result
    real_catchup = laggard._maybe_request_catchup
    laggard._handle_verify_open = _noop
    laggard._handle_verify_result = _noop
    laggard._maybe_request_catchup = _noop

    for r in range(3):
        await seq.submit_request(f"s{r}".encode(), timeout=20.0)
        await _settle()
    laggard._handle_verify_open = real_open
    laggard._handle_verify_result = real_result
    laggard._maybe_request_catchup = real_catchup

    assert laggard.chain.depth() == 0 and seq.chain.depth() == 3
    genuine = [seq._round_archive[ci] for ci in sorted(seq._round_archive)][:3]
    assert len(genuine) == 3
    D3 = seq.chain.head_digest()          # canonical head digest at depth 3

    # (1) NO anchor yet, non-sequencer server: even GENUINE rounds are refused,
    # because nothing trusted vouches for the head they lead to.
    await laggard._handle_chain_batch(
        {"from": impostor.node_id, "from_addr": impostor.address, "rounds": genuine})
    check("genuine rounds from a non-sequencer with NO anchor are refused",
          laggard.chain.depth() == 0)

    # Give the laggard a quorum of signed head attestations for (3, D3) -- exactly
    # what peers' beacons carry (every honest caught-up peer shares D3).
    attesters = [n for n in nodes if n is not laggard
                 and n.chain.head_digest() == D3][:REQUIRED_QUORUM]
    assert len(attesters) >= REQUIRED_QUORUM
    for a in attesters:
        laggard._head_attest[a.node_id] = {
            "depth": 3, "digest": a.chain.head_digest(), "ts": time.time()}

    # (2) TAMPERED slot-0 payload from a non-sequencer: the re-derived fingerprint
    # mismatches, so it is refused at the per-slot self-check (anchor or not).
    tampered = [dict(genuine[0])]
    tampered[0]["payload_b64"] = base64.b64encode(b"not-the-real-payload").decode()
    await laggard._handle_chain_batch(
        {"from": impostor.node_id, "from_addr": impostor.address, "rounds": tampered})
    check("tampered payload (bad re-derived fingerprint) is refused",
          laggard.chain.depth() == 0)

    # (3) SELF-CONSISTENT FORGERY: slots 0,1 genuine, slot 2 a DIFFERENT payload
    # whose fingerprint the forger re-derived so the per-slot check PASSES. It is
    # still refused, because the resulting head digest is not the one the quorum
    # attested -- the exact property the digest anchor adds over re-derivation.
    salt = laggard.chain.current_salt()             # depth 0 -> shared genesis salt
    salts = []
    for r in genuine:
        salts.append(salt)
        salt = PiMandelbrotKeyEngine.derive(
            base64.b64decode(r["payload_b64"]), salt, r["iter"]).fernet_key
    evil = b"evil-but-self-consistent"
    d2 = PiMandelbrotKeyEngine.derive(evil, salts[2], genuine[2]["iter"])
    forged = [genuine[0], genuine[1],
              {"chain_index": 2, "iter": genuine[2]["iter"],
               "payload_b64": base64.b64encode(evil).decode(),
               "verifiers": genuine[2]["verifiers"], "fingerprint": d2.fingerprint}]
    await laggard._handle_chain_batch(
        {"from": impostor.node_id, "from_addr": impostor.address, "rounds": forged})
    check("a self-consistent FORGED sequence (wrong head digest) is refused",
          laggard.chain.depth() == 0)

    # (4) POSITIVE CONTROL: the GENUINE rounds, now quorum-anchored, ARE adopted
    # from the non-sequencer server -- the new any-member-serves capability.
    await laggard._handle_chain_batch(
        {"from": impostor.node_id, "from_addr": impostor.address, "rounds": genuine})
    check(f"genuine rounds from a NON-sequencer, quorum-anchored, ARE adopted "
          f"(depth {laggard.chain.depth()} == {seq.chain.depth()}, no fork)",
          laggard.chain.depth() == seq.chain.depth()
          and laggard.chain.current_salt() == seq.chain.current_salt())
    for n in nodes:
        await n.stop()


async def test_failover_repair_without_original_sequencer():
    print("[anti-entropy] FAILOVER: repair completes from a NON-sequencer when "
          "the sequencer can't serve, anchored by the quorum head digest")
    import juvian_node as jn
    old_gate = jn.GOSSIP_MIN_ROSTER
    jn.GOSSIP_MIN_ROSTER = 10 ** 9       # reliable broadcast: every round verifies
    try:
        N = 12
        bus = InMemoryBus()
        nodes = await _spin(bus, N, lookups=6)
        _ctr = Counter(n._current_sequencer() for n in nodes)
        _sid, _votes = _ctr.most_common(1)[0]
        assert _votes >= (2 * len(nodes)) // 3, f"sequencer split: {_ctr}"
        seq = next(n for n in nodes if n.node_id == _sid)
        laggard = next(n for n in nodes
                       if n is not seq and n._current_sequencer() == seq.node_id)

        async def _noop(_msg):
            return
        saved = (laggard._handle_verify_result, laggard._maybe_request_catchup)
        laggard._handle_verify_result = _noop
        laggard._maybe_request_catchup = _noop

        ROUNDS = 3
        for r in range(ROUNDS):
            await seq.submit_request(f"f{r}".encode(), timeout=20.0)
            await _settle()
        assert laggard.chain.depth() == 0 and seq.chain.depth() == ROUNDS

        # The original (archiving) sequencer can no longer serve repair -- it has
        # departed / is unreachable for catch-up. It is STILL the current sequencer, so a
        # batch from it would be trusted directly; making it silent forces the
        # quorum-anchored, NON-sequencer path to carry the repair.
        seq_served = {"n": 0}
        async def seq_cannot_serve(_msg):
            seq_served["n"] += 1
            return
        seq._handle_chain_request = seq_cannot_serve

        laggard._handle_verify_result, laggard._maybe_request_catchup = saved
        for _ in range(5):
            await asyncio.gather(*(n.announce() for n in nodes))
            await _settle(30)

        check(f"laggard caught up despite the sequencer not serving "
              f"(depth {laggard.chain.depth()} == {seq.chain.depth()})",
              laggard.chain.depth() == seq.chain.depth())
        check("caught-up head matches the canonical head, by salt AND digest "
              "(no fork)",
              laggard.chain.current_salt() == seq.chain.current_salt()
              and laggard.chain.head_digest() == seq.chain.head_digest())
        print(f"     (sequencer serve attempts that returned nothing: "
              f"{seq_served['n']})")
        check(f"repair actually applied pulled rounds from a non-sequencer "
              f"(keys_caught_up={laggard.stats.get('keys_caught_up', 0)})",
              laggard.stats.get("keys_caught_up", 0) >= ROUNDS)
        for n in nodes:
            await n.stop()
    finally:
        jn.GOSSIP_MIN_ROSTER = old_gate


async def test_repair_converges_many_laggards():
    print("[anti-entropy] many simultaneous laggards all converge via repair")
    # Drive rounds via reliable broadcast (raise the gossip gate) so EVERY round
    # verifies -- this isolates the repair mechanism from gossip-round coverage
    # RNG. The lag here is INDUCED deterministically, not left to chance, and the
    # catch-up path (unicast request/batch) is unaffected by the gate.
    import juvian_node as jn
    old_gate = jn.GOSSIP_MIN_ROSTER
    jn.GOSSIP_MIN_ROSTER = 10 ** 9
    try:
        N = 40
        bus = InMemoryBus()
        nodes = await _spin(bus, N, lookups=6)
        _ctr = Counter(n._current_sequencer() for n in nodes)
        _sid, _votes = _ctr.most_common(1)[0]
        assert _votes >= (2 * len(nodes)) // 3, f"sequencer split: {_ctr}"
        seq = next(n for n in nodes if n.node_id == _sid)
        # a dozen laggards that all know the sequencer (so the only thing holding
        # them back is the induced suppression, not a routing gap)
        candidates = [n for n in nodes
                      if n is not seq and n._current_sequencer() == seq.node_id]
        laggards = candidates[:12]
        assert len(laggards) >= 8, "need a meaningful number of laggards"

        async def _noop(_msg):
            return
        saved = {}
        for L in laggards:
            saved[L.node_id] = (L._handle_verify_result, L._maybe_request_catchup)
            # suppress ONLY adoption + self-initiated catch-up: the laggard still
            # derives and COMMITS as a verifier, so rounds keep reaching quorum.
            L._handle_verify_result = _noop
            L._maybe_request_catchup = _noop

        ROUNDS = 3
        for r in range(ROUNDS):
            await seq.submit_request(f"m{r}".encode(), timeout=20.0)
            await _settle()

        held = sum(1 for L in laggards if L.chain.depth() == 0)
        check(f"{held}/{len(laggards)} laggards held at depth 0 while the "
              f"sequencer advanced to {seq.chain.depth()}",
              held == len(laggards) and seq.chain.depth() == ROUNDS)

        # restore and let discovery beacons reveal the gap to every laggard
        for L in laggards:
            L._handle_verify_result, L._maybe_request_catchup = saved[L.node_id]
        for _ in range(4):
            await asyncio.gather(*(n.announce() for n in nodes))
            await _settle(30)

        converged = sum(1 for L in laggards
                        if L.chain.depth() == seq.chain.depth())
        same_head = sum(1 for L in laggards
                        if L.chain.current_salt() == seq.chain.current_salt())
        total_caught = sum(L.stats.get("keys_caught_up", 0) for L in laggards)
        check(f"all {len(laggards)} laggards converged to the head via repair "
              f"({converged}/{len(laggards)} at depth {seq.chain.depth()})",
              converged == len(laggards))
        check("every converged laggard landed on the sequencer's head (no fork)",
              same_head == len(laggards))
        check(f"repair carried the missed slots "
              f"(catch-up adoptions across laggards={total_caught})",
              total_caught >= len(laggards))
        for n in nodes:
            await n.stop()
    finally:
        jn.GOSSIP_MIN_ROSTER = old_gate


def main():
    asyncio.run(test_catchup_heals_induced_lag())
    asyncio.run(test_catchup_anchoring_rejects_forgery())
    asyncio.run(test_failover_repair_without_original_sequencer())
    asyncio.run(test_repair_converges_many_laggards())
    if _failures == 0:
        print("\nALL ANTI-ENTROPY TESTS PASSED")
    else:
        print(f"\n{_failures} CHECK(S) FAILED")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
