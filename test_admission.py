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
JUVIAN GRID :: ADMISSION POLICY TESTS (Sybil BOUNDING on top of PoW)

Proof-of-work *prices* identities but does not *bound* them -- a well-resourced
attacker pays the work and a Sybil majority still breaks every honest-majority
quorum (verifier quorum, attestation quorum, the tensor outlier floor). The
admission policy bounds the population. It is layered ON TOP of PoW (never
replacing it) and is OFF by default, so a node with no policy behaves exactly as
before. Two independent, combinable controls:

  * allowlist  -- only ids on a configured list are ever admitted (closed fleet;
                  airtight, but needs out-of-band provisioning);
  * vouching   -- founders seed the root of trust; any ALREADY-ADMITTED member
                  signs a voucher for a newcomer, and a newcomer is admitted once
                  it presents >= threshold vouchers from DISTINCT admitted members.

A voucher is a self-certifying object (issuer signs over the subject's id, carries
the issuer's pubkey); the subject's own signed envelope proves it holds the key
for the vouched id, so a voucher cannot be replayed onto another identity.

What is proven here:
  * voucher primitives: a valid voucher verifies; a voucher for a different
    subject, a forged issuer id, a tampered field, and a wrong-key signature are
    all rejected;
  * OFF BY DEFAULT: with no policy, admission is inactive and an ordinary peer is
    admitted (the no-regression guarantee);
  * ALLOWLIST: a listed peer is admitted; an unlisted one is refused and never
    enters the routing view, even though it cleared PoW; allowlist is
    authoritative (vouchers cannot widen it);
  * VOUCHING: a newcomer with >= threshold admitted-issuer vouchers is admitted;
    an under-vouched newcomer is refused; a newcomer "vouched" only by
    NON-admitted issuers (other Sybils) is refused -- the bound that makes the
    policy meaningful; transitive admission promotes a chain in voucher order;
  * END-TO-END: in a vouching mesh a properly vouched node joins, is sequenced,
    and gets a request VERIFIED, while a Sybil is kept out of every honest view
    and cannot drive a verified request.

Honest limit (documented, not hidden): vouching is TRANSITIVE -- a vouched member
can itself vouch -- so a coalition of `threshold` careless-or-malicious admitted
members can still admit arbitrary Sybils. The allowlist is the airtight control;
vouching raises the bar to a real/social cost but is bounded-not-airtight.
==============================================================================
"""

import asyncio
import sys

from juvian_transport import InMemoryBus, InMemoryTransport
from juvian_node import JuvianNode, VOUCH_THRESHOLD
from juvian_identity import Identity, make_voucher, verify_voucher

PASS = "  PASS"
FAIL = "  ** FAIL **"
_failures = 0


def check(label, cond):
    global _failures
    print(f"{PASS if cond else FAIL} {label}")
    if not cond:
        _failures += 1


def _mk(bus, name, ip, **kw):
    addr = f"10.55.{(ip >> 8) & 255}.{ip & 255}:8000"
    return JuvianNode(name, addr, InMemoryTransport(addr, bus),
                      mandelbrot_iter=40, pow_difficulty=0,
                      history_path=f"/tmp/adm_{name}.npy",
                      chain_path=f"/tmp/adm_{name}.frac", **kw)


async def _settle(loops=8):
    for _ in range(loops):
        await asyncio.sleep(0.05)


def test_voucher_primitives():
    print("[admission] voucher object: valid verifies; subject-swap, forged "
          "issuer, tamper, and wrong-key signatures are rejected")
    issuer = Identity.generate(0)
    subject = "abcdef0123" * 4                  # a 40-hex subject id
    v = make_voucher(issuer, subject)
    check("a valid voucher returns the issuer id",
          verify_voucher(v, subject) == issuer.node_id)
    check("a voucher presented for a DIFFERENT subject is rejected",
          verify_voucher(v, "ff" * 20) is None)

    forged = dict(v); forged["iss"] = "00" * 20
    check("a voucher with a forged issuer id (key mismatch) is rejected",
          verify_voucher(forged, subject) is None)

    tampered = dict(v); tampered["iat"] = (v["iat"] or 0) + 5.0
    check("a voucher with a tampered timestamp (signature breaks) is rejected",
          verify_voucher(tampered, subject) is None)

    # a voucher whose signature was made by a different key than `pub`
    other = Identity.generate(0)
    swapped = dict(v); swapped["pub"] = other.public_hex()
    check("a voucher whose pubkey does not match the signature is rejected",
          verify_voucher(swapped, subject) is None)
    check("a non-dict voucher is rejected", verify_voucher(None, subject) is None)


async def test_admission_off_by_default():
    print("[admission] OFF by default: no policy -> admission inactive, an "
          "ordinary peer is admitted (no regression)")
    bus = InMemoryBus()
    p, q = _mk(bus, "p", 1), _mk(bus, "q", 2)
    await p.start(); await q.start()
    check("admission is inactive with no allowlist/founders",
          not p._admission_active())
    for n in (p, q):
        await n.announce()
    await _settle()
    check("an ordinary peer is admitted when no policy is configured",
          q.node_id in p.routing.peer_ids())
    for n in (p, q):
        await n.stop()


async def test_allowlist_admission():
    print("[admission] ALLOWLIST: listed peer admitted; unlisted refused (even "
          "though it cleared PoW) and never enters the view")
    bus = InMemoryBus()
    a, b = _mk(bus, "a", 1), _mk(bus, "b", 2)
    await a.start(); await b.start()
    allow = {a.node_id, b.node_id}
    for n in (a, b):
        n.allowlist = set(allow)
        n._admitted = {n.node_id}
    intruder = _mk(bus, "x", 9)                  # valid identity, NOT listed
    await intruder.start()
    for n in (a, b, intruder):
        await n.announce()
    await _settle()
    check("a listed peer is admitted", b.node_id in a.routing.peer_ids())
    check("an UNLISTED peer is refused and absent from the view",
          intruder.node_id not in a.routing.peer_ids())
    check("the refusal is counted", a.stats.get("admission_denied", 0) >= 1)

    # allowlist is authoritative: even a flood of (valid) vouchers can't widen it
    voucher_a = a.my_voucher_for(intruder.node_id)
    a._fold_vouchers([voucher_a], intruder.node_id)
    check("with an allowlist set, vouchers cannot widen admission",
          not a._is_admitted(intruder.node_id))
    # ... and neither can founder status: the allowlist overrides it (airtight
    # means airtight -- only self is exempt, so misconfiguration is loud).
    a.founders = {intruder.node_id}
    check("with an allowlist set, FOUNDER status cannot widen admission either",
          not a._is_admitted(intruder.node_id))
    a.founders = set()
    check("self stays admitted even if absent from its own allowlist",
          a._is_admitted(a.node_id))
    for n in (a, b, intruder):
        await n.stop()


async def test_vouching_threshold_and_bound():
    print("[admission] VOUCHING: >=threshold admitted-issuer vouchers admit a "
          "newcomer; under-vouched and Sybil-only-vouched newcomers refused")
    bus = InMemoryBus()
    f1, f2 = _mk(bus, "f1", 11), _mk(bus, "f2", 12)
    await f1.start(); await f2.start()
    founders = {f1.node_id, f2.node_id}
    for n in (f1, f2):
        n.founders = set(founders)
        n.vouch_threshold = 2
        n._admitted = {n.node_id} | founders

    # genuine newcomer: both founders vouch, it presents both
    nc = _mk(bus, "nc", 13, founders=founders, vouch_threshold=2)
    await nc.start()
    nc.present_voucher(f1.my_voucher_for(nc.node_id))
    nc.present_voucher(f2.my_voucher_for(nc.node_id))
    for n in (f1, f2, nc):
        await n.announce()
    await _settle()
    check("a newcomer with 2 distinct founder vouchers is admitted",
          nc.node_id in f1.routing.peer_ids()
          and nc.node_id in f2.routing.peer_ids())

    # under-vouched: only one founder voucher, threshold is 2
    under = _mk(bus, "under", 14, founders=founders, vouch_threshold=2)
    await under.start()
    under.present_voucher(f1.my_voucher_for(under.node_id))
    await under.announce()
    await _settle()
    check("an under-vouched newcomer (1 of 2) is refused",
          under.node_id not in f1.routing.peer_ids())

    # Sybil vouched only by NON-admitted issuers (itself + the under node, which
    # is not admitted) -- the bound that matters: vouchers from outside the
    # admitted set carry no weight.
    sybil = _mk(bus, "sybil", 15, founders=founders, vouch_threshold=2)
    await sybil.start()
    sybil.present_voucher(under.my_voucher_for(sybil.node_id))
    sybil.present_voucher(sybil.my_voucher_for(sybil.node_id))  # self-vouch (ignored)
    await sybil.announce()
    await _settle()
    check("a Sybil vouched only by NON-admitted issuers is refused",
          sybil.node_id not in f1.routing.peer_ids())
    check("self-vouching does not count toward the threshold",
          sybil.node_id not in f1._admitted)
    for n in (f1, f2, nc, under, sybil):
        await n.stop()


async def test_transitive_admission_order():
    print("[admission] transitive: a newcomer vouched by founder+newly-admitted "
          "member is promoted once its issuers are admitted (order-independent)")
    bus = InMemoryBus()
    f1, f2 = _mk(bus, "f1", 21), _mk(bus, "f2", 22)
    await f1.start(); await f2.start()
    founders = {f1.node_id, f2.node_id}
    # verifier node we will probe; threshold 2
    v = _mk(bus, "v", 20, founders=founders, vouch_threshold=2)
    await v.start()
    v._admitted = {v.node_id} | founders

    # m1 is admitted via two founder vouchers
    m1 = _mk(bus, "m1", 23)
    await m1.start()
    # m2 is vouched by f1 and m1 -- m1 is NOT a founder, so m2 can only be
    # admitted AFTER m1 is. Deliver m2's vouchers FIRST (m1 not yet admitted).
    m2 = _mk(bus, "m2", 24)
    await m2.start()

    v._fold_vouchers([f1.my_voucher_for(m2.node_id),
                      m1.my_voucher_for(m2.node_id)], m2.node_id)
    check("m2 is NOT admitted yet (its issuer m1 is not admitted)",
          not v._is_admitted(m2.node_id))

    # now admit m1 via two founder vouchers; this must promote m2 transitively
    v._fold_vouchers([f1.my_voucher_for(m1.node_id),
                      f2.my_voucher_for(m1.node_id)], m1.node_id)
    check("m1 becomes admitted on two founder vouchers", v._is_admitted(m1.node_id))
    check("m2 is promoted transitively once m1 is admitted",
          v._is_admitted(m2.node_id))
    for n in (f1, f2, v, m1, m2):
        await n.stop()


async def test_vouching_end_to_end():
    print("[admission] END-TO-END: vouched node joins a mesh, is sequenced, gets "
          "a request VERIFIED; a Sybil is kept out and cannot")
    import juvian_node as jn
    old_gate = jn.GOSSIP_MIN_ROSTER
    jn.GOSSIP_MIN_ROSTER = 10 ** 9            # reliable broadcast for hard lockstep
    try:
        bus = InMemoryBus()
        # three founders form the initial admitted mesh
        founders_nodes = [_mk(bus, f"f{i}", 30 + i) for i in range(3)]
        for n in founders_nodes:
            await n.start()
        founders = {n.node_id for n in founders_nodes}
        for n in founders_nodes:
            n.founders = set(founders)
            n.vouch_threshold = 2
            n._admitted = {n.node_id} | founders
        seed = founders_nodes[0].address
        await asyncio.gather(*(n.bootstrap([seed], refresh=4)
                              for n in founders_nodes[1:]))
        for _ in range(5):
            await asyncio.gather(*(n._lookup(n.node_id) for n in founders_nodes))

        # a properly vouched newcomer (2 founder vouchers)
        nc = _mk(bus, "nc", 40, founders=founders, vouch_threshold=2)
        await nc.start()
        nc.present_voucher(founders_nodes[0].my_voucher_for(nc.node_id))
        nc.present_voucher(founders_nodes[1].my_voucher_for(nc.node_id))
        await nc.bootstrap([seed], refresh=4)
        for _ in range(5):
            await asyncio.gather(*(n._lookup(n.node_id)
                                   for n in founders_nodes + [nc]))
        await _settle()
        admitted_everywhere = all(nc.node_id in n.routing.peer_ids()
                                  for n in founders_nodes)
        check("a properly vouched newcomer is admitted across the mesh",
              admitted_everywhere)

        # a Sybil with no admitted-issuer vouchers
        sybil = _mk(bus, "sybil", 41, founders=founders, vouch_threshold=2)
        await sybil.start()
        await sybil.bootstrap([seed], refresh=4)
        await _settle()
        check("a Sybil with no valid vouchers is in NO founder's view",
              all(sybil.node_id not in n.routing.peer_ids()
                  for n in founders_nodes))

        # the vouched newcomer can drive a verified request end to end
        res = await nc.submit_request(b"vouched-request", timeout=20.0)
        await _settle()
        check(f"the vouched newcomer gets a request VERIFIED "
              f"(status={res.get('status')})", res.get("status") == "VERIFIED")

        # the Sybil cannot
        res2 = await sybil.submit_request(b"let-me-in", timeout=3.0)
        check("the Sybil cannot drive a verified request",
              res2.get("status") != "VERIFIED")
        for n in founders_nodes + [nc, sybil]:
            await n.stop()
    finally:
        jn.GOSSIP_MIN_ROSTER = old_gate


async def test_nonadmitted_participation_surfaces():
    print("[admission] participation surfaces: a non-admitted sender's gossip "
          "is NOT relayed, its chain requests are NOT served, and a Sybil "
          "voucher flood cannot grow the pending cache without bound")
    import juvian_node as jn
    bus = InMemoryBus()
    f1, f2, f3 = _mk(bus, "g1", 51), _mk(bus, "g2", 52), _mk(bus, "g3", 54)
    for n in (f1, f2, f3):
        await n.start()
    founders = {f1.node_id, f2.node_id, f3.node_id}
    for n in (f1, f2, f3):
        n.founders = set(founders)
        n.vouch_threshold = 2
        n._admitted = {n.node_id} | founders
        n.verifier_fanout = 2      # 3-node mini-mesh: keep the round about
        n.verifier_quorum = 2      # admission, not quorum headroom
    outsider = _mk(bus, "out", 53, founders=founders, vouch_threshold=2)
    await outsider.start()
    for n in (f1, f2, f3, outsider):
        await n.announce()
    await _settle()
    assert outsider.node_id not in f1.routing.peer_ids()

    # (1) RELAY: a gossip-marked envelope from the non-admitted outsider must
    # not be re-forwarded by an honest node (no free N x fanout amplifier).
    relayed = {"n": 0}
    async def counting_gossip(envelope, exclude_addrs=None):
        relayed["n"] += 1
    f1._gossip = counting_gossip
    env = outsider.identity.wrap({"type": "BEACON", "from": outsider.node_id,
                                  "from_addr": outsider.address, "_g": True,
                                  "name": "out", "device_type": "X",
                                  "weight": 1.0})
    await f1._on_message(env, outsider.address)
    check("a non-admitted origin's _g message is NOT relayed",
          relayed["n"] == 0)
    # control: the SAME shape from an admitted origin IS relayed
    env2 = f2.identity.wrap({"type": "BEACON", "from": f2.node_id,
                             "from_addr": f2.address, "_g": True,
                             "name": "g2", "device_type": "X", "weight": 1.0})
    await f1._on_message(env2, f2.address)
    check("an admitted origin's _g message IS relayed (control)",
          relayed["n"] == 1)

    # (2) SERVE: drive one round between founders, then have the outsider ask
    # for the archive -- it must get nothing; an admitted peer gets the rounds.
    await f2.bootstrap([f1.address], refresh=3)
    await f3.bootstrap([f1.address], refresh=3)
    for _ in range(4):
        import asyncio as _a
        await _a.gather(*(n._lookup(n.node_id) for n in (f1, f2, f3)))
    res = await f1.submit_request(b"adm-round", timeout=20.0)
    assert res.get("status") == "VERIFIED"
    await _settle()
    served = {"n": 0}
    orig_send = f1._send
    async def counting_send(addr, body):
        if body.get("type") == "CHAIN_BATCH":
            served["n"] += 1
        await orig_send(addr, body)
    f1._send = counting_send
    await f1._handle_chain_request({"from": outsider.node_id,
                                    "from_addr": outsider.address,
                                    "have_depth": 0})
    check("a non-admitted requester is NOT served the archive", served["n"] == 0)
    await f1._handle_chain_request({"from": f2.node_id,
                                    "from_addr": f2.address, "have_depth": 0})
    check("an admitted requester IS served (control)", served["n"] == 1)
    f1._send = orig_send

    # (3) BOUND: a swarm of distinct never-admitted subjects cannot grow the
    # pending voucher cache past VOUCHER_CACHE_MAX.
    issuer = Identity.generate(0)
    for i in range(jn.VOUCHER_CACHE_MAX + 64):
        sub = f"{i:040x}"[:40]
        f1._fold_vouchers([make_voucher(issuer, sub)], sub)
    check(f"pending voucher cache is bounded "
          f"({len(f1._vouchers)} <= {jn.VOUCHER_CACHE_MAX})",
          len(f1._vouchers) <= jn.VOUCHER_CACHE_MAX)
    for n in (f1, f2, f3, outsider):
        await n.stop()


def main():
    test_voucher_primitives()
    asyncio.run(test_admission_off_by_default())
    asyncio.run(test_allowlist_admission())
    asyncio.run(test_vouching_threshold_and_bound())
    asyncio.run(test_transitive_admission_order())
    asyncio.run(test_nonadmitted_participation_surfaces())
    asyncio.run(test_vouching_end_to_end())
    if _failures == 0:
        print("\nALL ADMISSION TESTS PASSED")
    else:
        print(f"\n{_failures} CHECK(S) FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
