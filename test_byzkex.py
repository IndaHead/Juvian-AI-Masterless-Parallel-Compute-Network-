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
JUVIAN GRID :: BYZANTINE-ROBUST BD TESTS (echo / eviction edge cases)
==============================================================================

test_authkex.py covers the main equivocation path (an insider is proven, evicted,
and the honest remainder re-keys). This file covers the parts that file does not:

  1. The extra echo round does not break a small honest run (N=4 regression).
  2. FRAMING IS IMPOSSIBLE. A forged eviction "proof" -- identical-z (a relay
     duplicate, not equivocation), a tampered second envelope (signature fails),
     or two envelopes from different owners -- must NOT exclude the accused, while
     a GENUINE proof (two validly-signed, conflicting z from one owner) MUST. The
     unforgeable second signature is the whole guarantee.
  3. The equivocator can be the INITIATOR (lowest id). It is still evicted and the
     honest remainder re-keys under the next-lowest survivor.
  4. ROUND-2 (X) equivocation closes the last safe-abort case with the same
     construction: a genuine round-2 proof (two conflicting signed X) evicts,
     while a CROSS-ROUND pair (a member's own legitimate round-1 z and round-2 X)
     must NOT -- the same-round guard, the one new subtlety vs round 1. And a
     non-initiator that equivocates only in round 2 (good X to all, a bad X to one
     victim) is proven, evicted, and the honest remainder re-keys to one salt.
"""
import asyncio

from juvian_transport import InMemoryBus, InMemoryTransport
from juvian_node import (JuvianNode, MSG_KEX_R1, MSG_KEX_R2, MSG_KEX_EVICT)
from juvian_ecdh import GroupKeyAgreement

PASS = "  PASS"
FAIL = "  ** FAIL **"
_fails = 0


def check(label, cond):
    global _fails
    print(f"{PASS if cond else FAIL} {label}")
    if not cond:
        _fails += 1


def _mk(bus, i):
    addr = f"10.9.0.{i}:9000"
    return JuvianNode(f"b{i}", addr, InMemoryTransport(addr, bus),
                      mandelbrot_iter=70, pow_difficulty=0,
                      history_path=f"/tmp/bzk{i}.npy",
                      chain_path=f"/tmp/bzk{i}.frac")


async def _spin(bus, n):
    nodes = [_mk(bus, i) for i in range(n)]
    for nd in nodes:
        await nd.start()
    await asyncio.sleep(0.1)
    for nd in nodes:
        await nd.transport.broadcast({
            "type": "BEACON", "from": nd.node_id,
            "from_addr": nd.address, "name": nd.name})
    await asyncio.sleep(0.2)
    return nodes


async def _settle_until(cond, timeout=3.0, step=0.05):
    waited = 0.0
    while waited < timeout:
        if cond():
            return
        await asyncio.sleep(step)
        waited += step


def _r1_env(node, kex_id, z_hex):
    """A round-1 envelope signed by `node` (the same shape _reliable_broadcast
    would produce), usable as equivocation evidence."""
    return node.identity.wrap({
        "type": MSG_KEX_R1, "from": node.node_id,
        "from_addr": node.address, "kex_id": kex_id, "z": z_hex})


def _r2_env(node, kex_id, x_hex):
    """A round-2 (X) envelope signed by `node`, usable as round-2 evidence."""
    return node.identity.wrap({
        "type": MSG_KEX_R2, "from": node.node_id,
        "from_addr": node.address, "kex_id": kex_id, "x": x_hex})


async def test_honest_echo_path_n4():
    print("[byzkex] honest N=4 run converges with the echo round")
    bus = InMemoryBus()
    nodes = await _spin(bus, 4)
    initiator = min(nodes, key=lambda n: n.node_id)
    res = await initiator.establish_session(timeout=8.0)
    await _settle_until(
        lambda: sum(n.genesis_source == "GROUP_DH" for n in nodes) == len(nodes))
    installed = [n for n in nodes if n.genesis_source == "GROUP_DH"]
    distinct = {n.genesis_salt for n in installed}
    print(f"     {len(installed)}/{len(nodes)} installed; "
          f"{len(distinct)} distinct salt(s); status={res.get('status')}")
    check("all 4 installed one shared salt",
          len(installed) == 4 and len(distinct) == 1)
    check("no spurious eviction in an honest run",
          all(not n._kex_excluded for n in nodes))
    for n in nodes:
        await n.stop()


async def test_framing_is_rejected():
    print("[byzkex] forged eviction proofs are rejected; a genuine one is acted on")
    bus = InMemoryBus()
    a, b, v = await _spin(bus, 3)
    kid = "frame-kex"
    accuser = b.node_id

    async def feed(culprit, proof):
        await v._handle_kex_evict({
            "type": MSG_KEX_EVICT, "from": accuser, "from_addr": b.address,
            "kex_id": kid, "culprit": culprit, "proof": proof})

    # (a) identical z from the same owner: a relay duplicate, not equivocation
    e1 = _r1_env(a, kid, "01")
    e1b = _r1_env(a, kid, "01")
    await feed(a.node_id, [e1, e1b])
    check("identical-z 'proof' does NOT exclude the accused",
          a.node_id not in v._kex_excluded)

    # (b) tampered second envelope: change z but keep the old signature
    e_real = _r1_env(a, kid, "01")
    e_tampered = dict(e_real)
    e_tampered["body"] = {**e_real["body"], "z": "02"}   # sig no longer matches
    await feed(a.node_id, [e_real, e_tampered])
    check("tampered (bad-signature) 'proof' does NOT exclude the accused",
          a.node_id not in v._kex_excluded)

    # (c) two envelopes from DIFFERENT owners attributed to one culprit
    ea = _r1_env(a, kid, "01")
    eb = _r1_env(b, kid, "02")
    await feed(a.node_id, [ea, eb])
    check("different-owner 'proof' does NOT exclude the accused",
          a.node_id not in v._kex_excluded)

    # positive control: a GENUINE proof -- two validly-signed, conflicting z
    g1 = _r1_env(a, kid, "01")
    g2 = _r1_env(a, kid, "02")
    await feed(a.node_id, [g1, g2])
    check("a GENUINE proof DOES exclude the culprit (verifier isn't rejecting all)",
          a.node_id in v._kex_excluded)
    for n in (a, b, v):
        await n.stop()


async def test_equivocator_is_initiator():
    print("[byzkex] the equivocator is the INITIATOR (lowest id): still evicted, "
          "honest remainder re-keys under the next-lowest survivor")
    bus = InMemoryBus()
    nodes = await _spin(bus, 5)
    roster = sorted(n.node_id for n in nodes)
    by_id = {n.node_id: n for n in nodes}
    eq = by_id[roster[0]]                 # equivocator IS the canonical initiator
    victim = by_id[roster[-1]]
    honest = [n for n in nodes if n is not eq]

    corrupt_z = GroupKeyAgreement(eq.node_id, roster).round1_public()
    orig_rb = eq._reliable_broadcast

    async def evil_rb(body):
        if body.get("type") == MSG_KEX_R1:
            good_env = eq.identity.wrap(dict(body))
            bad_env = eq.identity.wrap({**body, "z": corrupt_z})
            for p in list(eq.routing.all_peers().values()):
                env = bad_env if p["address"] == victim.address else good_env
                await eq.transport.send(p["address"], env)
            return
        return await orig_rb(body)

    eq._reliable_broadcast = evil_rb

    # the equivocator initiates (it is the lowest id); don't block on its result
    asyncio.ensure_future(eq.establish_session(timeout=4.0))
    await _settle_until(
        lambda: (all(eq.node_id in n._kex_excluded for n in honest)
                 and sum(n.genesis_source == "GROUP_DH" for n in honest)
                 == len(honest)),
        timeout=5.0)

    installed = [n for n in honest if n.genesis_source == "GROUP_DH"]
    distinct = {n.genesis_salt for n in installed}
    print(f"     {len(installed)}/{len(honest)} honest installed; "
          f"{len(distinct)} distinct salt(s); "
          f"eq excluded by honest = "
          f"{all(eq.node_id in n._kex_excluded for n in honest)}")

    check("the initiator-equivocator was evicted by every honest member",
          all(eq.node_id in n._kex_excluded for n in honest))
    check("all honest members re-keyed under the next-lowest survivor",
          len(installed) == len(honest))
    check("the honest members agree on ONE salt", len(distinct) == 1)
    check("the evicted initiator did not install the group salt",
          eq.genesis_source != "GROUP_DH")
    for n in nodes:
        await n.stop()


async def test_round2_proof_and_cross_round_framing():
    print("[byzkex] round-2: a genuine X-proof evicts; a CROSS-ROUND (z paired "
          "with X) 'proof' must NOT -- the same-round guard")
    bus = InMemoryBus()
    a, b, v = await _spin(bus, 3)
    kid = "r2-frame-kex"
    accuser = b.node_id

    async def feed(culprit, proof):
        await v._handle_kex_evict({
            "type": MSG_KEX_EVICT, "from": accuser, "from_addr": b.address,
            "kex_id": kid, "culprit": culprit, "proof": proof})

    # CROSS-ROUND: a's own (legitimately different) round-1 z and round-2 X are
    # both validly signed by a -- pairing them must NOT count as equivocation.
    z_env = _r1_env(a, kid, "01")
    x_env = _r2_env(a, kid, "02")
    await feed(a.node_id, [z_env, x_env])
    check("cross-round (z + X) 'proof' does NOT exclude the accused",
          a.node_id not in v._kex_excluded)

    # identical-X (a relay duplicate, not equivocation) must NOT exclude either
    await feed(a.node_id, [_r2_env(a, kid, "01"), _r2_env(a, kid, "01")])
    check("identical-X 'proof' does NOT exclude the accused",
          a.node_id not in v._kex_excluded)

    # GENUINE round-2 proof: two validly-signed, conflicting X for one owner
    await feed(a.node_id, [_r2_env(a, kid, "01"), _r2_env(a, kid, "02")])
    check("a GENUINE round-2 proof (two conflicting X) DOES exclude the culprit",
          a.node_id in v._kex_excluded)
    for n in (a, b, v):
        await n.stop()


async def test_round2_equivocator_evicted():
    print("[byzkex] a NON-initiator equivocates in ROUND 2 (good X to all, a bad "
          "X to one victim): proven, evicted, honest remainder re-keys")
    bus = InMemoryBus()
    nodes = await _spin(bus, 5)
    roster = sorted(n.node_id for n in nodes)
    by_id = {n.node_id: n for n in nodes}
    eq = by_id[roster[-2]]                 # equivocator (non-initiator)
    victim = by_id[roster[-1]]
    initiator = by_id[roster[0]]           # canonical initiator (lowest id)
    honest = [n for n in nodes if n is not eq]

    # a valid-but-different group element to feed the victim as the corrupt X
    corrupt_x = GroupKeyAgreement(eq.node_id, roster).round1_public()
    orig_rb = eq._reliable_broadcast

    async def evil_rb(body):
        # honest in round 1 (one z to everyone); equivocate ONLY on round-2 X
        if body.get("type") == MSG_KEX_R2:
            good_env = eq.identity.wrap(dict(body))
            bad_env = eq.identity.wrap({**body, "x": corrupt_x})
            for p in list(eq.routing.all_peers().values()):
                env = bad_env if p["address"] == victim.address else good_env
                await eq.transport.send(p["address"], env)
            return
        return await orig_rb(body)

    eq._reliable_broadcast = evil_rb

    res = await initiator.establish_session(timeout=8.0)
    await _settle_until(
        lambda: (all(eq.node_id in n._kex_excluded for n in honest)
                 and sum(n.genesis_source == "GROUP_DH" for n in honest)
                 == len(honest)),
        timeout=6.0)

    installed = [n for n in honest if n.genesis_source == "GROUP_DH"]
    distinct = {n.genesis_salt for n in installed}
    evictions = sum(n.stats.get("kex_evictions", 0) for n in nodes)
    print(f"     initiator -> {res.get('status')}; {len(installed)}/{len(honest)} "
          f"honest installed; {len(distinct)} distinct salt(s); "
          f"evictions logged = {evictions}")

    check("the round-2 equivocator was proven and EVICTED by every honest member",
          all(eq.node_id in n._kex_excluded for n in honest))
    check("an eviction was logged (round-2 proof acted on, not a silent abort)",
          evictions >= 1)
    check("the honest remainder re-keyed: all 4 honest members installed",
          len(installed) == len(honest))
    check("the steered victim RECOVERED via the re-key",
          victim.genesis_source == "GROUP_DH")
    check("the honest members agree on ONE salt", len(distinct) == 1)
    check("the initiator COMPLETED the session via the re-key (not an abort)",
          res.get("status") == "ESTABLISHED")
    check("the evicted culprit did NOT install the group salt",
          eq.genesis_source != "GROUP_DH")
    for n in nodes:
        await n.stop()


def main():
    print("=" * 64)
    print("JUVIAN GRID :: BYZANTINE-ROBUST BD TESTS")
    print("=" * 64)
    asyncio.run(test_honest_echo_path_n4())
    asyncio.run(test_framing_is_rejected())
    asyncio.run(test_equivocator_is_initiator())
    asyncio.run(test_round2_proof_and_cross_round_framing())
    asyncio.run(test_round2_equivocator_evicted())
    print("=" * 64)
    if _fails:
        print(f"{_fails} CHECK(S) FAILED")
        raise SystemExit(1)
    print("ALL BYZANTINE-ROBUST BD TESTS PASSED")
    print("=" * 64)


if __name__ == "__main__":
    main()
