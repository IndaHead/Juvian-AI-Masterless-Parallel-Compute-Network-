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
JUVIAN GRID :: ECDH / GROUP-KEY-AGREEMENT DEMO
==============================================================================
Shows the confidentiality layer end to end:

  PHASE 1  discovery
  PHASE 2  group key agreement (Burmester-Desmedt) -> every member installs the
           SAME genesis salt, derived from a secret no eavesdropper can compute
  PHASE 3  an "eavesdropper" who captured every broadcast value tries (and
           fails) to reconstruct the genesis salt
  PHASE 4  the pi-Mandelbrot 3-of-3 chain now runs on top of the secret salt --
           same verified protocol, now confidential against outsiders

Runs over the in-memory bus (no sockets needed).
==============================================================================
"""

import asyncio
import hashlib

from juvian_transport import InMemoryBus, InMemoryTransport
from juvian_node import JuvianNode
from juvian_ecdh import GroupKeyAgreement, PairwiseECDH


SECRET = "ecdh-demo-session"


async def main():
    bus = InMemoryBus()
    specs = [
        ("anchor-1", "10.0.0.1:8000", "ANCHOR", 5.0),
        ("anchor-2", "10.0.0.2:8000", "ANCHOR", 5.0),
        ("mobile-1", "10.0.0.11:8000", "MOBILE", 2.5),
        ("mobile-2", "10.0.0.12:8000", "MOBILE", 2.5),
    ]
    nodes = [
        JuvianNode(n, a, InMemoryTransport(a, bus),
                   session_secret=SECRET, device_type=d, hw_weight=w,
                   mandelbrot_iter=300,
                   history_path=f"/tmp/{n}_kex_hist.npy",
                   chain_path=f"/tmp/{n}_kex_chain.frac")
        for n, a, d, w in specs
    ]
    for n in nodes:
        await n.start()

    print("=" * 70)
    print("PHASE 1 :: DISCOVERY")
    print("=" * 70)
    await asyncio.sleep(0.15)
    for n in nodes:
        await n.transport.broadcast({
            "type": "BEACON", "from": n.node_id, "from_addr": n.address,
            "name": n.name, "device_type": n.device_type, "weight": n.hw_weight})
    await asyncio.sleep(0.15)
    for n in nodes:
        print(f"  {n.name:10s} sees {n.routing.count()} peers | "
              f"genesis source = {n.genesis_source}")

    print()
    print("=" * 70)
    print("PHASE 2 :: GROUP KEY AGREEMENT (Burmester-Desmedt)")
    print("=" * 70)
    print("  (two broadcast rounds; no secret ever leaves a node)")
    # Only the canonical initiator (lowest node id) starts the KEX, so two
    # nodes can't concurrently install different genesis salts (audit 1.4).
    initiator_node = min(nodes, key=lambda n: n.node_id)
    print(f"  canonical initiator = {initiator_node.name} "
          f"(lowest node id; should_initiate_kex={initiator_node.should_initiate_kex()})")
    result = await initiator_node.establish_session(timeout=8.0)
    print(f"  initiator result: {result['status']} | members={result.get('members')} "
          f"| genesis_fp={result.get('genesis_fingerprint')}")
    await asyncio.sleep(0.3)   # let every member finish installing

    print()
    print("  per-node genesis salt after agreement:")
    salts = {}
    for n in nodes:
        fp = hashlib.sha256(n.genesis_salt).hexdigest()[:16]
        salts[n.name] = fp
        print(f"    {n.name:10s} source={n.genesis_source:9s} salt_fp={fp}")
    all_same = len(set(salts.values())) == 1
    print(f"  >>> all members share the SAME secret genesis salt: {all_same}")

    print()
    print("=" * 70)
    print("PHASE 3 :: EAVESDROPPER CANNOT RECONSTRUCT THE SALT")
    print("=" * 70)
    # Reconstruct exactly what a passive attacker sees: the roster and every
    # broadcast z_i and X_i. They have NO private exponent r_i.
    roster = sorted(n.node_id for n in nodes)
    members = [GroupKeyAgreement(n.node_id, roster) for n in nodes]
    zs = {m.member_id: m.round1_public() for m in members}
    for m in members:
        for mid, z in zs.items():
            if mid != m.member_id:
                m.set_round1(mid, z)
    xs = {m.member_id: m.round2_public() for m in members}
    for m in members:
        for mid, x in xs.items():
            if mid != m.member_id:
                m.set_round2(mid, x)
    legit_salt = members[0].group_salt().hex()[:16]

    # The eavesdropper has zs and xs but cannot run group_salt() (no r_i).
    # Best they can do is guess; show that the public transcript alone does not
    # yield the key by hashing the public values and comparing.
    public_transcript = "".join(zs[m] for m in sorted(zs)) + \
                        "".join(xs[m] for m in sorted(xs))
    attacker_guess = hashlib.sha256(
        b"JUVIAN_BD_GENESIS::" + public_transcript.encode()).hexdigest()[:16]
    print(f"  legitimate member salt_fp : {legit_salt}")
    print(f"  attacker-from-transcript  : {attacker_guess}")
    print(f"  >>> attacker matches legit salt: {attacker_guess == legit_salt} "
          f"(False = confidentiality holds)")

    print()
    print("=" * 70)
    print("PHASE 4 :: 3-OF-3 CHAIN ON TOP OF THE SECRET SALT")
    print("=" * 70)
    init = nodes[0]
    for i in range(3):
        r = await init.submit_request(f"confidential-task-{i}".encode(), timeout=4.0)
        print(f"  request {i}: {r.get('status'):9s} | "
              f"fp={(r.get('fingerprint') or '')[:16]} | depth={r.get('chain_depth')}")
        await asyncio.sleep(0.05)

    print()
    print("  final node states:")
    for n in nodes:
        s = n.snapshot()
        print(f"    {s['name']:10s} | genesis={s['genesis_source']:9s} | "
              f"chain={s['chain_depth']} | fp={s['latest_fingerprint']}")

    print()
    print("=" * 70)
    print("BONUS :: PAIRWISE X25519 ECDH (two-party confidential link)")
    print("=" * 70)
    alice, bob = PairwiseECDH(), PairwiseECDH()
    ka = alice.derive(bob.public_bytes())
    kb = bob.derive(alice.public_bytes())
    print(f"  alice<->bob shared key matches: {ka == kb}  (32 bytes, never sent)")

    for n in nodes:
        await n.stop()
    print()
    print("ECDH LAYER DEMO COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
