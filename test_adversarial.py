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
JUVIAN GRID :: ADVERSARIAL TEST SUITE
==============================================================================
Verifies that the audit remediations actually hold under attack -- not just
that the happy path works. Each test states the finding it guards and FAILS
loudly (assertion) if the protection regresses.

Run: python3 test_adversarial.py
==============================================================================
"""

import asyncio
import hashlib
import time
from collections import Counter

import numpy as np

from juvian_identity import (
    Identity, verify_envelope, node_id_for, canonical, mint_pow, verify_pow,
)
from juvian_crypto import (
    ThreeWayVerification, SessionBootstrap, PiMandelbrotKeyEngine,
    derivation_proof, MAX_RETAINED_ROUNDS,
)
from juvian_ecdh import GroupKeyAgreement, P, Q
from juvian_transport import InMemoryBus, InMemoryTransport
from juvian_node import (JuvianNode, MAX_TENSOR_ELEMENTS, TENSOR_RATE_MAX,
                         SEQUENCER_TERM)


PASS, FAIL = "  PASS", "  ** FAIL **"


def check(label, condition):
    print((PASS if condition else FAIL), label)
    assert condition, label


# ---------------------------------------------------------------------------
# 2.1  message authentication: spoofing / tampering rejected
# ---------------------------------------------------------------------------
def test_message_auth():
    print("[2.1] message authentication")
    alice = Identity.generate()
    env = alice.wrap({"from": alice.node_id, "type": "BEACON", "v": 1})
    check("honest signed envelope verifies", verify_envelope(env) is not None)

    # tamper with the body after signing
    bad = {"pub": env["pub"], "sig": env["sig"],
           "body": {**env["body"], "v": 2}}
    check("tampered body rejected", verify_envelope(bad) is None)

    # spoof: claim a different `from` than the key hashes to
    spoof = alice.wrap({"from": "f" * 40, "type": "VERIFY_RESULT"})
    check("from-spoof rejected (id must match key)", verify_envelope(spoof) is None)

    # forge: attacker re-signs a body claiming to be alice
    mallory = Identity.generate()
    forged_body = {"from": alice.node_id, "type": "VERIFY_RESULT",
                   "status": "VERIFIED"}
    forged = {"pub": mallory.public_hex(),
              "sig": mallory.sign(canonical(forged_body)).hex(),
              "body": forged_body}
    check("forged result (mallory signs as alice) rejected",
          verify_envelope(forged) is None)


# ---------------------------------------------------------------------------
# 1.1  3-of-3 proof-of-derivation: free-riding rejected
# ---------------------------------------------------------------------------
def test_free_rider():
    print("[1.1] 3-of-3 proof-of-derivation (no free-riding)")
    salt = SessionBootstrap.pre_shared("s")
    payload = b"task"
    A = ThreeWayVerification("a" * 40)
    verifiers = ["a" * 40, "b" * 40, "c" * 40]
    rnd, my_proof = A.open_round("sid", payload, salt, verifiers, iterations=200)

    # honest B derives the key and produces its OWN identity-bound proof
    keyB = PiMandelbrotKeyEngine.derive(payload, salt, 200).fernet_key
    proofB = derivation_proof(keyB, "sid", "b" * 40)
    rB = A.submit_commitment("sid", "b" * 40, proofB, payload_hash=rnd.payload_hash)

    # free-rider C copies B's proof verbatim (best an eavesdropper can do)
    rC = A.submit_commitment("sid", "c" * 40, proofB, payload_hash=rnd.payload_hash)
    status = (rC or rB or {}).get("status")
    check("verifier copying another's proof does NOT verify", status != "VERIFIED")

    # and the honest path still works: C derives and proves for its own id
    A2 = ThreeWayVerification("a" * 40)
    rnd2, _ = A2.open_round("sid2", payload, salt, verifiers, iterations=200)
    A2.submit_commitment("sid2", "b" * 40,
                         derivation_proof(keyB, "sid2", "b" * 40),
                         payload_hash=rnd2.payload_hash)
    rC2 = A2.submit_commitment("sid2", "c" * 40,
                               derivation_proof(keyB, "sid2", "c" * 40),
                               payload_hash=rnd2.payload_hash)
    check("honest 3-of-3 (each proves for own id) verifies",
          (rC2 or {}).get("status") == "VERIFIED")


# ---------------------------------------------------------------------------
# 1.2  Burmester-Desmedt: malicious group elements rejected (no crash)
# ---------------------------------------------------------------------------
def test_bd_validation():
    print("[1.2] BD group-element validation")
    roster = ["a", "b", "c"]
    small_order = P - 1                      # order 2 element
    for bad_label, bad_val in [("z=0", 0), ("z=1", 1),
                               ("z=P-1 (order 2)", small_order)]:
        bd = GroupKeyAgreement("b", roster)
        accepted = bd.set_round1("a", format(bad_val, "x"))
        check(f"{bad_label} rejected by set_round1", accepted is False)

    # a valid subgroup element is accepted, and round2 does not crash
    bd = GroupKeyAgreement("b", roster)
    good = pow(2, 123456, P)                  # 2 generates the order-Q subgroup
    check("valid subgroup element accepted", bd.set_round1("a", format(good, "x")))
    bd.set_round1("c", format(pow(2, 654321, P), "x"))
    try:
        bd.round2_public()
        check("round2 completes on valid input (no crash)", True)
    except Exception as e:
        check(f"round2 unexpectedly raised: {e}", False)


# ---------------------------------------------------------------------------
# 3.1  malformed tensor input does not crash; sizes are bounded
# ---------------------------------------------------------------------------
def test_tensor_decode_safety():
    print("[3.1] tensor decode safety")
    bus = InMemoryBus()
    n = JuvianNode("n", "10.0.0.1:8000", InMemoryTransport("10.0.0.1:8000", bus))
    dec = n._safe_decode_array

    good = __import__("base64").b64encode(
        np.ones((2, 2), dtype=np.float64).tobytes()).decode()
    check("well-formed array decodes", dec(good, [2, 2]) is not None)
    check("shape/byte-count mismatch -> None (no crash)", dec(good, [9, 9]) is None)
    check("junk base64 -> None (no crash)", dec("!!!!notb64!!!!", [2, 2]) is None)
    check("oversized shape rejected by element cap",
          dec(good, [MAX_TENSOR_ELEMENTS, 2]) is None)
    check("negative dimension -> None", dec(good, [-1, 4]) is None)


# ---------------------------------------------------------------------------
# 2.2  unsolicited compute: non-members rejected, rate limited
# ---------------------------------------------------------------------------
async def test_tensor_membership_and_rate():
    print("[2.2] tensor membership + rate limit")
    bus = InMemoryBus()
    victim = JuvianNode("v", "10.0.0.1:8000",
                        InMemoryTransport("10.0.0.1:8000", bus))
    await victim.start()

    sent = {"n": 0}
    orig = victim.transport.send
    async def counting_send(addr, msg):
        sent["n"] += 1
        await orig(addr, msg)
    victim.transport.send = counting_send

    proj = np.ones((2, 4), dtype=np.float64)
    tensor = np.ones((4, 3), dtype=np.float64)
    import base64
    task = {
        "from": "deadbeef" * 5,        # a stranger, not in routing table
        "from_addr": "10.9.9.9:8000",
        "job_id": "j1",
        "tensor_b64": base64.b64encode(tensor.tobytes()).decode(),
        "tensor_shape": [4, 3],
        "proj_b64": base64.b64encode(proj.tobytes()).decode(),
        "proj_shape": [2, 4],
    }
    await victim._handle_tensor_task(task)
    check("stranger's tensor task ignored (no yield sent)", sent["n"] == 0)

    # now make the requester a known peer and flood past the rate cap
    victim.routing.update("deadbeef" * 5, "10.9.9.9:8000")
    for _ in range(TENSOR_RATE_MAX + 5):
        await victim._handle_tensor_task(task)
    check(f"rate limit caps accepted tasks at ~{TENSOR_RATE_MAX}",
          sent["n"] <= TENSOR_RATE_MAX)
    await victim.stop()


# ---------------------------------------------------------------------------
# 3.2  round table is bounded (no unbounded memory growth)
# ---------------------------------------------------------------------------
def test_round_eviction():
    print("[3.2] verification round eviction")
    salt = SessionBootstrap.pre_shared("s")
    V = ThreeWayVerification("a" * 40)
    for i in range(MAX_RETAINED_ROUNDS + 200):
        V.open_round(f"sid{i}", b"p", salt, ["a" * 40], iterations=60)
    check(f"rounds retained <= {MAX_RETAINED_ROUNDS}",
          len(V.rounds) <= MAX_RETAINED_ROUNDS)


# ---------------------------------------------------------------------------
# 1.4  canonical KEX initiator: only one node starts a key exchange
# ---------------------------------------------------------------------------
async def test_canonical_kex():
    print("[1.4] canonical KEX initiator")
    bus = InMemoryBus()
    nodes = [JuvianNode(f"n{i}", f"10.0.0.{i+1}:8000",
                        InMemoryTransport(f"10.0.0.{i+1}:8000", bus))
             for i in range(4)]
    for n in nodes:
        await n.start()
    await asyncio.sleep(0.1)
    for n in nodes:
        await n._broadcast({"type": "BEACON", "from": n.node_id,
                            "from_addr": n.address, "name": n.name,
                            "device_type": n.device_type, "weight": n.hw_weight})
    await asyncio.sleep(0.1)
    initiators = [n for n in nodes if n.should_initiate_kex()]
    check("exactly one node is the canonical KEX initiator", len(initiators) == 1)
    lowest = min(nodes, key=lambda n: n.node_id)
    check("the canonical initiator is the lowest node id",
          initiators[0].node_id == lowest.node_id)
    for n in nodes:
        await n.stop()


def test_replay():
    print("[replay] freshness + duplicate suppression")
    from juvian_identity import verify_envelope, REPLAY_MAX_AGE_S
    import time as _t
    alice = Identity.generate()

    env = alice.wrap({"from": alice.node_id, "type": "VERIFY_RESULT"})
    body = verify_envelope(env)
    check("fresh signed message verifies", body is not None)

    # a stale message (old timestamp, validly signed) is rejected on freshness
    stale_body = {"from": alice.node_id, "type": "VERIFY_RESULT",
                  "_n": "00" * 12, "_t": _t.time() - (REPLAY_MAX_AGE_S + 10)}
    stale = {"pub": alice.public_hex(),
             "sig": alice.sign(canonical(stale_body)).hex(),
             "body": stale_body}
    check("stale (old timestamp) message rejected", verify_envelope(stale) is None)

    # node-level duplicate suppression: same (from, nonce) seen twice -> replay
    bus = InMemoryBus()
    n = JuvianNode("n", "10.0.0.1:8000", InMemoryTransport("10.0.0.1:8000", bus))
    check("first sighting is not a replay", n._is_replay(body) is False)
    check("identical message second time IS a replay", n._is_replay(body) is True)
    other = verify_envelope(alice.wrap({"from": alice.node_id, "type": "BEACON"}))
    check("a different message (fresh nonce) is not a replay",
          n._is_replay(other) is False)


async def test_concurrent_no_fork():
    print("[4.1] concurrent initiators do not fork the chain")
    bus = InMemoryBus()
    nodes = [JuvianNode(f"n{i}", f"10.0.0.{i+1}:8000",
                        InMemoryTransport(f"10.0.0.{i+1}:8000", bus),
                        mandelbrot_iter=120,
                        history_path=f"/tmp/cf{i}.npy", chain_path=f"/tmp/cf{i}.frac")
             for i in range(4)]
    for n in nodes:
        await n.start()
    await asyncio.sleep(0.1)
    for n in nodes:
        await n._broadcast({"type": "BEACON", "from": n.node_id,
                            "from_addr": n.address, "name": n.name,
                            "device_type": n.device_type, "weight": n.hw_weight})
    await asyncio.sleep(0.15)

    # two DIFFERENT nodes fire requests at the same instant
    a, b = nodes[1], nodes[2]
    r1, r2 = await asyncio.gather(
        a.submit_request(b"concurrent-A", timeout=4.0),
        b.submit_request(b"concurrent-B", timeout=4.0),
    )
    await asyncio.sleep(0.2)
    statuses = sorted([r1.get("status"), r2.get("status")])
    check("both concurrent requests resolved VERIFIED",
          statuses == ["VERIFIED", "VERIFIED"])

    depths = {n.name: n.chain.depth() for n in nodes}
    fps = {n.name: (n.chain.entries[-1].fingerprint if n.chain.entries else None)
           for n in nodes}
    check("all nodes reached the SAME chain depth (no fork)",
          len(set(depths.values())) == 1)
    check("all nodes share the SAME head fingerprint (no fork)",
          len(set(fps.values())) == 1)
    check("exactly two entries appended (one per request, totally ordered)",
          list(depths.values())[0] == 2)
    for n in nodes:
        await n.stop()


async def test_sequencer_rotation_and_agreement():
    print("[sequencer] verifiable rotating selection: nodes agree, leadership "
          "rotates across members and epochs, a ground low id buys nothing")
    bus = InMemoryBus()
    nodes = [JuvianNode(f"sq{i}", f"10.0.9.{i+1}:8000",
                        InMemoryTransport(f"10.0.9.{i+1}:8000", bus),
                        mandelbrot_iter=40, pow_difficulty=0,
                        history_path=f"/tmp/sqr{i}.npy", chain_path=f"/tmp/sqr{i}.frac")
             for i in range(16)]
    for n in nodes:
        await n.start()
    seed = nodes[0].address
    await asyncio.gather(*(n.bootstrap([seed], refresh=6) for n in nodes[1:]))
    for _ in range(8):                              # converge routing tables
        await asyncio.gather(*(n._lookup(n.node_id) for n in nodes))

    # (1) AGREEMENT: at this size every routing view is complete, so every node
    # independently computes the SAME sequencer with no election traffic.
    sids = {n._current_sequencer() for n in nodes}
    check(f"all {len(nodes)} nodes agree on one sequencer (distinct={len(sids)})",
          len(sids) == 1)
    seq_id = next(iter(sids))

    # (2) VERIFIABILITY: a third party recomputes the winner from PUBLIC inputs
    # alone (the member ids, the epoch seed, the epoch index) and gets the same id.
    ref = nodes[0]
    members = sorted({n.node_id for n in nodes})
    seed_b = ref._epoch_seed(ref.chain.depth())
    epoch = ref.chain.depth() // SEQUENCER_TERM
    recomputed = min(members, key=lambda pid: hashlib.sha256(
        seed_b + epoch.to_bytes(8, "big") + pid.encode()).digest())
    check("the chosen sequencer is independently verifiable from public inputs",
          recomputed == seq_id and seq_id in members)

    # (3) ROTATION + DISTRIBUTION (degrinding): sweep the epoch index and see who
    # leads each epoch. The hash makes the id's VALUE irrelevant -- leadership
    # must spread across essentially all members, the lowest id (the OLD
    # permanent sequencer) must hold no special share, and every node computes
    # the same winner per epoch.
    EPOCHS = 400
    winners = []
    for k in range(EPOCHS):
        depth_k = k * SEQUENCER_TERM
        w = ref._sequencer_for(depth_k)
        if k < 25:                                  # per-epoch agreement spot-check
            assert all(nd._sequencer_for(depth_k) == w for nd in nodes), \
                "per-epoch sequencer disagreement"
        winners.append(w)
    dist = Counter(winners)
    lowest_id = min(n.node_id for n in nodes)
    top_share = max(dist.values()) / EPOCHS
    # Uniform over 16 members -> expected share 1/16 = 0.0625, std ~0.012; the
    # 0.25 ceiling is >10 sigma, and 14/16 distinct leaders over 400 epochs is
    # essentially certain (P[a given member never wins] ~ e^-25). Robust bounds.
    check(f"leadership rotates across the membership over {EPOCHS} epochs "
          f"(distinct leaders={len(dist)}/{len(members)})",
          len(dist) >= 14)
    check(f"no single id monopolises leadership (top share={top_share:.2f} <= 0.25)",
          top_share <= 0.25)
    check(f"the lowest node id is just another member, not a permanent sequencer "
          f"(it led {dist.get(lowest_id, 0)}/{EPOCHS})",
          dist.get(lowest_id, 0) <= EPOCHS // 4)
    for n in nodes:
        await n.stop()


async def test_rogue_round_rejected():
    print("[4.1] a non-sequencer cannot drive a round")
    bus = InMemoryBus()
    nodes = [JuvianNode(f"m{i}", f"10.0.1.{i+1}:8000",
                        InMemoryTransport(f"10.0.1.{i+1}:8000", bus))
             for i in range(3)]
    for n in nodes:
        await n.start()
    await asyncio.sleep(0.1)
    for n in nodes:
        await n._broadcast({"type": "BEACON", "from": n.node_id,
                            "from_addr": n.address, "name": n.name,
                            "device_type": n.device_type, "weight": n.hw_weight})
    await asyncio.sleep(0.15)

    _ctr = Counter(n._current_sequencer() for n in nodes)
    _sid, _votes = _ctr.most_common(1)[0]
    assert _votes >= (2 * len(nodes)) // 3, f"sequencer split: {_ctr}"
    seq = next(n for n in nodes if n.node_id == _sid)
    victim = max(nodes, key=lambda n: n.node_id)
    rogue = next(n for n in nodes if n is not seq)   # any non-sequencer id

    sent = {"n": 0}
    orig = victim.transport.send
    async def counting(addr, msg):
        sent["n"] += 1
        await orig(addr, msg)
    victim.transport.send = counting

    # a non-sequencer tries to open a round directly at the victim
    import base64 as _b64
    await victim._handle_verify_open({
        "from": rogue.node_id, "from_addr": rogue.address,
        "session_id": "rogue1",
        "payload_b64": _b64.b64encode(b"hijack").decode(),
        "verifiers": [victim.node_id], "iter": 80, "chain_index": 0,
    })
    check("victim ignores a round opened by a non-sequencer (no commit sent)",
          sent["n"] == 0 and "rogue1" not in victim.verifier.rounds)
    for n in nodes:
        await n.stop()


async def test_quorum_liveness():
    print("[4.3] a silent verifier no longer stalls a request (quorum)")
    bus = InMemoryBus()
    nodes = [JuvianNode(f"q{i}", f"10.0.2.{i+1}:8000",
                        InMemoryTransport(f"10.0.2.{i+1}:8000", bus),
                        mandelbrot_iter=100,
                        history_path=f"/tmp/ql{i}.npy", chain_path=f"/tmp/ql{i}.frac")
             for i in range(5)]
    for n in nodes:
        await n.start()
    await asyncio.sleep(0.1)
    for n in nodes:
        await n._broadcast({"type": "BEACON", "from": n.node_id,
                            "from_addr": n.address, "name": n.name,
                            "device_type": n.device_type, "weight": n.hw_weight})
    await asyncio.sleep(0.15)

    _ctr = Counter(n._current_sequencer() for n in nodes)
    _sid, _votes = _ctr.most_common(1)[0]
    assert _votes >= (2 * len(nodes)) // 3, f"sequencer split: {_ctr}"
    seq = next(n for n in nodes if n.node_id == _sid)
    silent = [n for n in nodes if n is not seq][0]
    async def _noop(msg):           # this verifier never answers
        return
    silent._handle_verify_open = _noop

    res = await seq.submit_request(b"liveness", timeout=4.0)
    check("request VERIFIED despite one unresponsive verifier",
          res.get("status") == "VERIFIED")
    check("quorum of independent confirmations still met",
          len(res.get("verifiers", [])) >= seq.verifier_quorum)
    for n in nodes:
        await n.stop()


async def test_quorum_retry():
    print("[4.3] retry with a disjoint verifier set when a quorum can't be met")
    bus = InMemoryBus()
    nodes = [JuvianNode(f"r{i}", f"10.0.3.{i+1}:8000",
                        InMemoryTransport(f"10.0.3.{i+1}:8000", bus),
                        mandelbrot_iter=100,
                        history_path=f"/tmp/rt{i}.npy", chain_path=f"/tmp/rt{i}.frac")
             for i in range(6)]
    for n in nodes:
        await n.start()
    await asyncio.sleep(0.1)
    for n in nodes:
        await n._broadcast({"type": "BEACON", "from": n.node_id,
                            "from_addr": n.address, "name": n.name,
                            "device_type": n.device_type, "weight": n.hw_weight})
    await asyncio.sleep(0.15)

    _ctr = Counter(n._current_sequencer() for n in nodes)
    _sid, _votes = _ctr.most_common(1)[0]
    assert _votes >= (2 * len(nodes)) // 3, f"sequencer split: {_ctr}"
    seq = next(n for n in nodes if n.node_id == _sid)
    seq.verifier_fanout = 3            # small fanout so the first set can fail
    seq.verifier_quorum = 3
    payload = b"retry-me"

    # compute the first verifier set the sequencer will pick, and silence the
    # non-sequencer members of it, forcing a retry onto a disjoint set
    candidates = seq.routing.peer_ids() + [seq.node_id]
    first = set(ThreeWayVerification.select_verifiers(
        hashlib.sha256(payload).hexdigest(), candidates, 3))
    by_id = {n.node_id: n for n in nodes}
    async def _noop(msg):
        return
    for nid in first:
        if nid != seq.node_id:
            by_id[nid]._handle_verify_open = _noop

    opens = []
    orig = seq._disseminate
    async def _spy(body):
        if body.get("type") == "VERIFY_OPEN":
            opens.append(body["session_id"])
        return await orig(body)
    seq._disseminate = _spy

    res = await seq.submit_request(payload, timeout=1.5)
    check("request VERIFIED after retrying a fresh verifier set",
          res.get("status") == "VERIFIED")
    check("a retry actually occurred (more than one verify-open round)",
          len(opens) >= 2)
    for n in nodes:
        await n.stop()


def test_pow_cost_scales():
    print("[2.3/4.4] minting cost scales with difficulty (Sybil pricing)")
    import os
    import statistics
    pub = os.urandom(32)
    nonce8, _ = mint_pow(pub, 8)
    check("a minted cert verifies at its own difficulty", verify_pow(pub, nonce8, 8))
    check("an empty cert is rejected when difficulty is required",
          not verify_pow(pub, "", 8))

    def avg_attempts(diff, samples):
        return statistics.mean(mint_pow(os.urandom(32), diff)[1]
                               for _ in range(samples))
    a_lo = avg_attempts(8, 24)      # ~2**8  = 256 expected
    a_hi = avg_attempts(12, 24)     # ~2**12 = 4096 expected (~16x)
    check(f"+4 difficulty bits costs materially more work "
          f"(d12 avg {a_hi:.0f} >> d8 avg {a_lo:.0f})", a_hi > a_lo * 3)


async def test_sybil_pow_admission():
    print("[2.3/4.4] a no-PoW Sybil is denied admission to the mesh")
    DIFF = 12
    bus = InMemoryBus()

    def mk(name, ip, diff):
        return JuvianNode(name, f"10.0.4.{ip}:8000",
                          InMemoryTransport(f"10.0.4.{ip}:8000", bus),
                          mandelbrot_iter=100, pow_difficulty=diff,
                          history_path=f"/tmp/sy_{name}.npy",
                          chain_path=f"/tmp/sy_{name}.frac")

    async def beacon(n):
        await n._broadcast({"type": "BEACON", "from": n.node_id,
                            "from_addr": n.address, "name": n.name,
                            "device_type": n.device_type, "weight": n.hw_weight})

    honest = [mk(f"h{i}", i + 1, DIFF) for i in range(3)]
    for n in honest:
        await n.start()
    await asyncio.sleep(0.1)
    for n in honest:
        await beacon(n)
    await asyncio.sleep(0.15)

    sybil = mk("sybil", 99, 0)      # difficulty 0 -> empty birth certificate
    await sybil.start()
    await beacon(sybil)             # tries to join the honest mesh
    await asyncio.sleep(0.15)

    legit = mk("legit", 50, DIFF)   # control: a real late joiner WITH a cert
    await legit.start()
    await beacon(legit)
    await asyncio.sleep(0.15)

    sybil_seen = any(sybil.node_id in n.routing.peer_ids() for n in honest)
    legit_seen = all(legit.node_id in n.routing.peer_ids() for n in honest)
    check("Sybil with no PoW is in NO honest node's routing table", not sybil_seen)
    check("a legitimate peer WITH a valid PoW cert is admitted (control)",
          legit_seen)

    # the Sybil also cannot get a request verified -- it is not a member, so its
    # forwarded request never reaches a sequencer that will act for it
    res = await sybil.submit_request(b"let me in", timeout=1.0)
    check("Sybil cannot drive a verified request through the honest mesh",
          res.get("status") != "VERIFIED")

    for n in honest + [sybil, legit]:
        await n.stop()


async def test_zone_tensor_locality():
    print("[edge] tensor compute prefers same-zone peers, with a Byzantine floor")
    bus = InMemoryBus()

    def mk(name, ip, zone):
        return JuvianNode(name, f"10.0.5.{ip}:8000",
                          InMemoryTransport(f"10.0.5.{ip}:8000", bus),
                          mandelbrot_iter=80, pow_difficulty=0, geo_zone=zone,
                          history_path=f"/tmp/zl_{name}.npy",
                          chain_path=f"/tmp/zl_{name}.frac")

    zoneA = [mk(f"a{i}", 10 + i, "ZA") for i in range(4)]   # big zone
    zoneB = [mk(f"b{i}", 20 + i, "ZB") for i in range(2)]   # small zone
    nodes = zoneA + zoneB
    for n in nodes:
        await n.start()
    await asyncio.sleep(0.1)
    for n in nodes:
        await n.announce()
    await asyncio.sleep(0.2)

    A_addr = {n.address for n in zoneA}
    B_addr = {n.address for n in zoneB}

    def spy_on(node):
        sent = []
        orig = node._send
        async def spy(addr, body):
            if body.get("type") == "TENSOR_TASK":
                sent.append(addr)
            return await orig(addr, body)
        node._send = spy
        return sent

    # big zone: initiator has 3 same-zone peers (+itself = 4) -> stays local
    init_a = zoneA[0]
    sent_a = spy_on(init_a)
    await init_a.run_tensor_job(np.random.randn(6, 4, 4), timeout=1.5)
    check("big-zone job dispatched only within the initiator's zone",
          set(sent_a) and set(sent_a).issubset(A_addr) and not (set(sent_a) & B_addr))

    # small zone: initiator has 1 same-zone peer -> borrows nearest out-of-zone
    init_b = zoneB[0]
    sent_b = spy_on(init_b)
    await init_b.run_tensor_job(np.random.randn(6, 4, 4), timeout=1.5)
    check("small-zone job kept the same-zone peer", zoneB[1].address in set(sent_b))
    check("small-zone job borrowed out-of-zone peers to hold the MAD floor",
          len(set(sent_b) & A_addr) >= 1)

    for n in nodes:
        await n.stop()


async def test_telemetry_load_shedding():
    print("[edge] cooperative thermal load-shedding (not a security control)")
    bus = InMemoryBus()

    def mk(name, ip):
        return JuvianNode(name, f"10.0.6.{ip}:8000",
                          InMemoryTransport(f"10.0.6.{ip}:8000", bus),
                          mandelbrot_iter=80, pow_difficulty=0,
                          history_path=f"/tmp/tl_{name}.npy",
                          chain_path=f"/tmp/tl_{name}.frac")

    nodes = [mk(f"t{i}", 10 + i) for i in range(3)]
    for n in nodes:
        await n.start()
    await asyncio.sleep(0.1)
    for n in nodes:
        await n.announce()
    await asyncio.sleep(0.2)

    hot = nodes[0]
    others = nodes[1:]
    hot.simulated_temp = 95.0          # simulate overheating
    await hot.telemetry.tick()
    check("overheating node zeroed its own compute weight", hot.hw_weight == 0.0)
    check("overheating node entered the bouncing state", hot.telemetry.is_bouncing)

    # the governor: while bounced, the node drops incoming tensor tasks
    yielded = []
    orig_send = hot._send
    async def spy(addr, body):
        if body.get("type") == "TENSOR_YIELD":
            yielded.append(addr)
        return await orig_send(addr, body)
    hot._send = spy
    await hot._handle_tensor_task({"from": others[0].node_id,
                                   "from_addr": others[0].address, "job_id": "j"})
    check("bounced node's governor drops incoming tensor tasks", not yielded)
    hot._send = orig_send

    await asyncio.sleep(0.2)            # let the handover signal propagate
    peer_weights = [n.routing.all_peers().get(hot.node_id, {}).get("weight")
                    for n in others]
    check("peers down-weighted the bounced node via the handover signal",
          all(w == 0.0 for w in peer_weights))

    hot.simulated_temp = 40.0           # cools back down
    await hot.telemetry.tick()
    check("recovered node restored its nominal compute weight",
          hot.hw_weight == hot.nominal_hw_weight and not hot.telemetry.is_bouncing)

    for n in nodes:
        await n.stop()


def main():
    print("=" * 60)
    print("JUVIAN GRID :: ADVERSARIAL TEST SUITE")
    print("=" * 60)
    test_message_auth()
    test_replay()
    test_free_rider()
    test_bd_validation()
    test_tensor_decode_safety()
    asyncio.run(test_tensor_membership_and_rate())
    test_round_eviction()
    asyncio.run(test_canonical_kex())
    asyncio.run(test_concurrent_no_fork())
    asyncio.run(test_sequencer_rotation_and_agreement())
    asyncio.run(test_rogue_round_rejected())
    asyncio.run(test_quorum_liveness())
    asyncio.run(test_quorum_retry())
    test_pow_cost_scales()
    asyncio.run(test_sybil_pow_admission())
    asyncio.run(test_zone_tensor_locality())
    asyncio.run(test_telemetry_load_shedding())
    print("=" * 60)
    print("ALL ADVERSARIAL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
