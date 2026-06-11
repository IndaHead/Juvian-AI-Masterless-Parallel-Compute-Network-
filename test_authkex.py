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
JUVIAN GRID :: AUTHENTICATED-KEX TESTS (confirmation round + Byzantine-robust BD)
==============================================================================

A confidential genesis salt is only as good as the agreement that produced it.
Two distinct attacks matter for the Burmester-Desmedt (BD) key agreement, and
the node defends against both in layers:

1. KEY-CONFIRMATION ROUND (safety -- no split-brain). Before installing the
   derived salt, every member publishes an HMAC proving which key it derived and
   installs only once a MAJORITY of the roster has published the SAME proof. A
   member whose key diverges never reaches that majority, so it refuses to
   install rather than silently forking. Two disjoint majorities cannot exist,
   so at most one salt is ever installed network-wide. This is the backstop for
   an active MITM that perturbs values in transit and for a round-2 (X_i)
   equivocation: either way the disagreeing parties simply never reach quorum.

2. ROUND-1 ECHO -> EVICTION (Byzantine-robust BD -- liveness). The canonical BD
   attack is round-1 EQUIVOCATION: an insider signs a DIFFERENT z_i to a chosen
   victim (each validly signed by its own key). Classically the victim folds the
   corrupt value into its honest round-2 broadcast, perturbing EVERY member's
   key, so the whole KEX aborts -- and a naive retry re-includes the equivocator
   forever (a liveness DoS). The echo round closes this: once a member holds all
   round-1 z's it broadcasts the SIGNED envelopes it received, so every member
   can cross-check them. Two distinct, validly-signed z for one owner is a
   NON-REPUDIABLE proof of equivocation (a victim harvests the conflicting value
   from a peer's echo). The culprit is evicted and the honest remainder re-keys
   WITHOUT it -- so the group makes progress instead of looping aborts. Framing
   an honest member is impossible: a second signature under its key cannot be
   forged.

These tests prove: the honest path still converges on one salt with the extra
echo round, and an insider equivocation now ends in the culprit's eviction plus
a successful honest re-key (one salt, no fork, victim recovers) rather than a
full abort. Forged-proof / framing rejection is covered in test_byzkex.py.

Honest scope: a round-2 (X_i) equivocation is still handled as a safe no-install
abort (the same echo construction extends to it -- the natural next step); and a
member that stays SILENT (withholds its echo without equivocating) still stalls
the round to a timeout, exactly as withholding any round did before -- the new
property is specifically that a proven EQUIVOCATOR is evicted, not that every
liveness fault is repaired.
"""
import asyncio

from juvian_transport import InMemoryBus, InMemoryTransport
from juvian_node import JuvianNode, MSG_KEX_R1
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
    addr = f"10.7.0.{i}:9000"
    return JuvianNode(f"k{i}", addr, InMemoryTransport(addr, bus),
                      mandelbrot_iter=70, pow_difficulty=0,
                      history_path=f"/tmp/akx{i}.npy",
                      chain_path=f"/tmp/akx{i}.frac")


async def _spin_and_discover(bus, n):
    nodes = [_mk(bus, i) for i in range(n)]
    for nd in nodes:
        await nd.start()
    await asyncio.sleep(0.1)
    for nd in nodes:                       # one beacon round populates routing
        await nd.transport.broadcast({
            "type": "BEACON", "from": nd.node_id,
            "from_addr": nd.address, "name": nd.name})
    await asyncio.sleep(0.2)
    return nodes


async def _settle_until(cond, nodes, timeout=3.0, step=0.05):
    """Poll until cond() holds or timeout; the eviction task + re-key span
    several scheduled message rounds, so we wait on the outcome rather than a
    fixed sleep (keeps the test non-flaky)."""
    waited = 0.0
    while waited < timeout:
        if cond():
            return
        await asyncio.sleep(step)
        waited += step


async def test_honest_kex_confirms():
    print("[auth-kex] honest run: the confirmation round (with the echo round) "
          "still installs ONE shared salt")
    bus = InMemoryBus()
    nodes = await _spin_and_discover(bus, 5)
    initiator = min(nodes, key=lambda n: n.node_id)

    res = await initiator.establish_session(timeout=8.0)
    await _settle_until(
        lambda: sum(n.genesis_source == "GROUP_DH" for n in nodes) == len(nodes),
        nodes)

    installed = [n for n in nodes if n.genesis_source == "GROUP_DH"]
    distinct = {n.genesis_salt for n in installed}
    print(f"     initiator -> {res.get('status')}; "
          f"{len(installed)}/{len(nodes)} installed; "
          f"{len(distinct)} distinct salt(s)")

    check("initiator reports ESTABLISHED", res.get("status") == "ESTABLISHED")
    check("all 5 members installed a group-DH salt", len(installed) == len(nodes))
    check("every member installed the SAME salt (majority confirmed it)",
          len(distinct) == 1)
    for n in nodes:
        await n.stop()


async def test_equivocation_is_evicted():
    print("[auth-kex] insider equivocation: a member signs a DIFFERENT z to a "
          "victim -- the culprit must be EVICTED and the honest remainder re-key")
    bus = InMemoryBus()
    nodes = await _spin_and_discover(bus, 5)

    roster = sorted(n.node_id for n in nodes)
    by_id = {n.node_id: n for n in nodes}
    victim = by_id[roster[-1]]                       # equivocation target
    eq = by_id[roster[-2]]                            # the equivocator (non-initiator)
    initiator = by_id[roster[0]]                     # canonical initiator (lowest id)

    # A real group element g^r' that differs from eq's true z. eq hands this to
    # the victim and its real z to everyone else -- both validly eq-signed, so
    # the pair is a non-repudiable equivocation proof once cross-checked.
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

    res = await initiator.establish_session(timeout=8.0)
    # wait for the eviction + honest re-key to settle (culprit excluded by the
    # honest majority and the honest remainder installed)
    await _settle_until(
        lambda: (all(eq.node_id in n._kex_excluded for n in nodes if n is not eq)
                 and sum(n.genesis_source == "GROUP_DH" for n in nodes)
                 == len(nodes) - 1),
        nodes)

    honest = [n for n in nodes if n is not eq]
    installed = [n for n in nodes if n.genesis_source == "GROUP_DH"]
    distinct = {n.genesis_salt for n in installed}
    evictions = sum(n.stats.get("kex_evictions", 0) for n in nodes)
    print(f"     initiator -> {res.get('status')}; "
          f"{len(installed)}/{len(nodes)} installed; "
          f"{len(distinct)} distinct salt(s); "
          f"victim source = {victim.genesis_source}; "
          f"evictions logged = {evictions}")

    check("no split-brain: at most ONE genesis salt exists network-wide",
          len(distinct) <= 1)
    check("the equivocator was proven and EVICTED by every honest member",
          all(eq.node_id in n._kex_excluded for n in honest))
    check("an eviction was actually logged (proof acted on, not a silent skip)",
          evictions >= 1)
    check("the honest remainder re-keyed: all 4 honest members installed",
          len(installed) == len(nodes) - 1)
    check("the steered victim RECOVERED via the re-key (installed a salt)",
          victim.genesis_source == "GROUP_DH")
    check("the honest members agree on ONE salt (the re-key converged)",
          len(installed) >= 1 and len(distinct) == 1)
    check("the initiator COMPLETED the session via the re-key (not an abort)",
          res.get("status") == "ESTABLISHED")
    check("the evicted culprit did NOT install the group salt",
          eq.genesis_source != "GROUP_DH")
    for n in nodes:
        await n.stop()


def main():
    print("=" * 64)
    print("JUVIAN GRID :: AUTHENTICATED-KEX (CONFIRMATION + BYZANTINE-ROBUST BD)")
    print("=" * 64)
    asyncio.run(test_honest_kex_confirms())
    asyncio.run(test_equivocation_is_evicted())
    print("=" * 64)
    if _fails:
        print(f"{_fails} CHECK(S) FAILED")
        raise SystemExit(1)
    print("ALL AUTHENTICATED-KEX TESTS PASSED")
    print("=" * 64)


if __name__ == "__main__":
    main()
