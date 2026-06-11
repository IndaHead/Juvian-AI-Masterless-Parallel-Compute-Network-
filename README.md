# Juvian Grid — End-to-End Node

A decentralized, masterless peer-to-peer mesh where nodes establish a
**confidential shared secret** by group Diffie–Hellman, derive **per-request
encryption keys** from a π-scaled Mandelbrot iteration seeded by that secret,
agree on each key by **quorum consensus** (proof-of-derivation), chain those keys into a
forward-secret session history, and run **distributed tensor reductions** with
Byzantine-fault filtering — all over plain UDP on a LAN.

This README is deliberately precise about what is real and what is not, because
the project's own spec documents oversell it. See **[Honest scope](#honest-scope)**.

---

## What actually works (tested end-to-end)

Running `python3 demo_local.py` and `python3 demo_ecdh.py` exercises the whole
stack over an in-memory transport and prints real results:

- **Discovery** — 5 nodes broadcast beacons and each builds a peer table.
- **Iterative Kademlia discovery** — a node can instead JOIN from a single seed
  address (no O(N²) flood) by looking up its own id through the DHT; the lookup
  lands on the true XOR-closest node (exactly so for the large majority of
  targets, never worse than the second-closest — exact-closest isn't guaranteed
  once buckets can't hold the whole roster) and its cost grows ~log N (a handful
  of queries at N=20–80, not N). At zone scale (~≤40) the roster converges to
  near-complete coverage; beyond that a Kademlia table holds O(k·log N) contacts,
  not everyone — correct DHT behaviour, and why large scale wants lookup-based
  verifier selection rather than a complete roster.
- **Group key agreement** — members run Burmester–Desmedt over a 2048-bit MODP
  group; every member computes the identical group key with no secret on the
  wire, and it becomes the chain's genesis salt. `demo_ecdh.py` also shows an
  eavesdropper with the full transcript failing to reconstruct it.
- **Byzantine-robust BD (round-1 + round-2 echo → eviction → re-key)** — the BD
  equivocation attack has two forms: round-1 (an insider signs a different `z` to
  a victim) and round-2 (it signs a different `X`). Before each round advances,
  every member echoes the *signed* envelopes it received and withholds the next
  step until that view is consistent. Because an honest member broadcasts one
  value to all, an equivocator must emit two separately-signed envelopes — and two
  validly-signed, conflicting values for one owner is a **non-repudiable proof**,
  which a victim can harvest from a peer's echo. The culprit is **evicted** (every
  receiver re-verifies the proof itself; framing is impossible without forging a
  second signature) and the honest remainder re-keys under the lowest surviving
  id, instead of the whole group aborting. The round-2 proof additionally requires
  both halves to be the *same round*, so a member's own legitimate `z` and `X`
  cannot be paired into a bogus proof. Honest scope: a single *wrong-but-consistent*
  contribution (the same bad value to everyone) is not equivocation and remains a
  safe no-install abort — proving it would need a zero-knowledge proof of correct
  computation BD does not carry — and a member that stays silent still stalls to a
  timeout (the baseline liveness class). Only a *proven equivocator* is evicted.
- **π-Mandelbrot key derivation** — each request's key is derived fresh from
  `SHA-256(payload)` → π-scaled Mandelbrot coordinates → 1000-iteration escape
  path → `SHA-256(escape_state ‖ previous_key)`. The previous verified key is
  chained in as salt, so the chain is forward-secret.
- **Quorum verification (proof-of-derivation)** — verifiers are selected by XOR
  proximity to the payload hash, found via an iterative Kademlia lookup
  (`_verifier_pool`) rather than a full-roster scan, so selection works from a
  partial routing table at scale (identical to a roster scan at zone scale). Each
  derives the key independently and returns an **identity-bound proof** (HMAC of
  the key keyed to its own node id), not a bare fingerprint a free-rider could
  echo. The round confirms as soon as a **quorum** of valid proofs arrives
  (default 3 of up to 5 invited; both configurable per node) and retries on a
  disjoint verifier set if a quorum can't be met, so one offline verifier no
  longer stalls it. Each proof is checked against the confirming node's own key,
  so an absent or wrong proof can never push a wrong key through. The key itself
  is never transmitted. (The round is *announced* to all members for lockstep
  chain adoption; at zone scale that is a guaranteed reliable broadcast, and
  above a roster threshold it switches to bounded-fan-out **gossip** — see
  "Gossip dissemination" below — so the sequencer's fan-out stays O(fanout)
  instead of O(N).)
- **Verifiable rotating sequencer (degrindable ordering)** — all requests are
  totally ordered through one sequencer per epoch (32 chain slots), chosen as the
  member minimising `H(epoch_seed ‖ epoch ‖ id)` where the seed is the verifiable
  chain **head digest at the epoch boundary**. Any third party recomputes the
  winner from public inputs; no election traffic. Because the hash makes the id's
  *value* irrelevant, grinding a low node id no longer buys leadership — over a
  400-epoch sweep all 16 test members lead and the lowest id holds only its fair
  share. The sequencer has **ordering power only** (members adopt only their own
  re-derived, fingerprint-checked keys), and that power now expires every term.
  Honest residuals: the genesis epoch is still id-grindable (one term); an
  *incumbent* leader can bias the next epoch's seed via boundary-slot payload
  placement (active, on-chain-visible — full unbiasability needs commit-reveal /
  threshold randomness). Routing buckets are sized (k=48) so a zone's membership
  fits every node's view completely — beyond ~2×k members per zone, membership
  agreement would need to move on-chain (the documented federation boundary).
- **Liveness-maintained membership + dead-sequencer failover** — the beacon loop
  doubles as a maintenance tick: any peer silent beyond `PEER_EXPIRY_S` (no
  beacon, no authenticated traffic — every verified message refreshes liveness)
  is pruned from the view, and a stopped node goes fully dark instead of
  lingering as a half-alive ghost. Because the rotating sequencer is the
  hash-argmin over the *live* membership, a dead sequencer's removal
  automatically recomputes the role to the next-ranked survivor — the
  deterministic fallback IS the selection rule; no election protocol. Forwarded
  requests retry with a recomputed target each attempt (sequencing directly if
  expiry promotes the origin), and results are accepted only from a node the
  request was actually sent to. What was a *permanent* stall on sequencer death
  is now bounded by ~`PEER_EXPIRY_S` + one beacon tick; a request submitted
  inside that window can still time out — a bounded outage, not zero outage. A
  per-depth single-candidate guard (`DEPTH_GUARD_S`) makes each member honor at
  most one opener per chain slot at a time, narrowing the dual-candidate race a
  transient expiry-skew view split could open; honest limit: a true network
  partition where both sides retain a verifier quorum is not solved by expiry —
  that is partition consensus, out of scope.
- **Chain lockstep** — every session member adopts each verified key, so all
  nodes stay at identical chain depth and fingerprint. In the demo all 5 nodes
  converge on the same fingerprint after 4 chained requests, 0 rejections.
- **Gossip dissemination (above zone scale)** — once the roster passes a
  threshold, the per-request verify broadcasts flood the mesh by gossip (push to
  a few peers; each relays on first receipt; dedup on the origin's signed nonce)
  with O(fanout) per-node cost rather than an O(N) fan-out at the sequencer.
  Below the threshold the guaranteed reliable broadcast is kept (hard lockstep,
  no change). Honest limit: gossip coverage is high-probability, not guaranteed —
  every adopter still lands on the same head (no fork), but a node missed by a
  flood lags until **anti-entropy repair** (see next bullet) pulls the slots it
  missed. KEX is *not* gossiped (one-time path; preserves the
  authenticated-KEX equivocation analysis).
- **Anti-entropy repair (closes the gossip tail; survives a sequencer handover)**
  — a node that sees a round `chain_index` ahead of its own depth (from a verify
  message, or proactively from a peer's `chain_depth`/`chain_digest` in the beacon)
  pulls the missed slots from several archive-holders and **re-derives each key
  from its own salt**, refusing any slot whose fingerprint disagrees. Every member
  now archives and serves (not just the sequencer), and the chain carries a
  cumulative, order-binding **head digest**: a batch from *any* server is adopted
  only up to the longest prefix whose re-derived head digest is anchored by a
  **quorum** of signed beacons (or the current sequencer's). So a tampered payload,
  an unanchored batch, and even a *self-consistent forged* sequence are all
  refused, and repair converges to the canonical head **even after the original
  sequencer leaves** — carried by a non-sequencer, quorum-anchored, with no fork.
  Residual (liveness, never safety): a node whose routing view is too sparse to
  gather a quorum of attestations (and that cannot see the sequencer) waits until
  discovery fills its view in.
- **Reliable broadcast** — the critical broadcasts (KEX rounds, verify
  open/result) are delivered to each peer over the reliable unicast path (ARQ),
  not best-effort, so a dropped packet can't split the group key or desync a
  chain. Tested: a KEX and three pipelined verify rounds both hold every member
  in lockstep over a 20–30%-loss network. (Beacons stay best-effort — they
  repeat.)
- **Distributed tensor reduce** — an anchor shards a matrix across workers,
  collects yields, runs a coordinate-wise **MAD filter** (median-aggregated
  Z-scores, threshold 4.5) to purge poisoned contributions, then a weighted
  SVD energy reduction. The demo injects one Byzantine worker and confirms it is
  purged while the four honest workers are kept. Workers are chosen **zone-local**
  where possible (with a contributor floor so small zones don't starve the MAD
  filter); zones are a locality hint only and never shard consensus.
- **Sybil-priced admission** — every node carries a **proof-of-work birth
  certificate**; a peer with no valid cert (or below the required difficulty) is
  refused admission to the mesh. Tested: the Sybil is in no honest routing table
  and cannot drive a verified request.
- **Admission policy (Sybil BOUNDING, on top of PoW)** — PoW prices identities
  but does not bound them, and every honest-majority quorum (verifiers, the
  catch-up attestation anchor, the tensor outlier floor) ultimately assumes a
  bounded adversary. Two combinable, **off-by-default** controls close that: an
  **allowlist** (only listed ids are ever admitted — airtight for a closed
  fleet, and authoritative: neither vouchers nor founder status can widen it),
  and **vouching** (founders seed the root of trust; a newcomer is admitted
  once it presents ≥ threshold signed vouchers from *distinct already-admitted*
  members — so minting N Sybils costs N×threshold genuine endorsements, a real
  social cost, not compute). Vouchers are self-certifying signed objects bound
  to the subject's id, carried on the newcomer's own beacons; admission gates
  **participation**, not just visibility — a non-admitted node never enters a
  routing view, is never sequenced, is not served chain repair, and its gossip
  is not relayed (no free amplification). Honest limits: vouching is transitive,
  so a coalition of `threshold` careless or malicious admitted members can still
  admit Sybils (the allowlist is the airtight control); and policy consistency
  across a zone is the operator's job — nodes configured with different
  policies see different memberships, which degrades liveness, never key
  integrity.
- **Cooperative load-shedding** — an optional, off-by-default thermal/battery
  monitor lets a node shed compute and signal peers to down-weight it (a
  cooperative aid for honest nodes, not a security control).
- **Persistence** — the session key chain is written to a `.frac` file and
  reloaded losslessly.

Example demo output (abridged):

```
request 0: VERIFIED | fp=69a37732d6b4094f | chain_depth=1
request 1: VERIFIED | fp=5f3832491e44dbaf | chain_depth=2
request 2: VERIFIED | fp=63bea3e5068bc668 | chain_depth=3
request 3: VERIFIED | fp=80fe025fe81c63af | chain_depth=4
  keys verified: 4 | rejected: 0

Byzantine attacker purged: True

anchor-1 | chain=4 | fp=80fe025fe81c63af
anchor-2 | chain=4 | fp=80fe025fe81c63af
mobile-1 | chain=4 | fp=80fe025fe81c63af
mobile-2 | chain=4 | fp=80fe025fe81c63af
mobile-3 | chain=4 | fp=80fe025fe81c63af
```

---

## Files

| File | Lines | Role |
|------|------:|------|
| `juvian_node.py` | 2486 | The node daemon — discovery, group key agreement, Byzantine-robust BD KEX, verifiable rotating sequencer, verification, gossip dissemination, failover-capable anti-entropy repair, tensor jobs, persistence |
| `juvian_crypto.py` | 529 | π-Mandelbrot key engine, session bootstrap, key chain (with verifiable head digest), quorum verification, cipher |
| `juvian_ecdh.py` | 234 | Key-agreement layer — X25519 pairwise ECDH + Burmester–Desmedt group key agreement |
| `juvian_reliable.py` | 250 | Reliable chunked datagram layer (ARQ) over UDP + lossy-network test harness |
| `juvian_web.py` | 215 | Optional per-node live web dashboard (aiohttp) |
| `run_node.py` | 213 | Real-UDP entry point + interactive command prompt |
| `juvian_fractal_storage.py` | 189 | 8-way isometric IFS storage for the chain `.frac` file |
| `test_adversarial.py` | 699 | Adversarial regression suite (auth, replay, free-riding, BD, tensor, ordering, sequencer rotation/degrinding, quorum + verifiers-only commit) |
| `test_reliable.py` | 150 | Reliable-transport tests (chunking, loss, dedup, end-to-end tensor + KEX + verify-lockstep over loss) |
| `test_authkex.py` | 206 | Authenticated-KEX tests (key-confirmation round + Byzantine-robust BD: honest convergence with the echo round; a non-initiator insider equivocation is proven, EVICTED, and the honest remainder re-keys to one salt with the victim recovering and the initiator completing) |
| `test_byzkex.py` | 315 | Byzantine-robust BD edge cases (honest N=4 regression with both echo rounds; three classes of forged round-1 proof rejected while a genuine one is acted on; an initiator-equivocator still evicted with the remainder re-keying under the next survivor; and round-2: a cross-round `z`+`X` proof rejected by the same-round guard, a genuine two-conflicting-`X` proof evicts, and a non-initiator that equivocates only in round 2 is evicted with the honest remainder re-keying) |
| `test_discovery.py` | 236 | Iterative Kademlia discovery tests (single-seed join, O(log N) lookup, near-closest lookup, consensus on discovered roster, lookup-based verifier selection) |
| `test_gossip.py` | 175 | Gossip dissemination tests above the size gate (N=50): bounded per-node fan-out, high-probability mesh coverage, verify round VERIFIES end-to-end over the overlay with no fork |
| `test_admission.py` | 406 | Admission policy tests: voucher primitives (forge/tamper/subject-swap rejected), off-by-default no-regression, allowlist authority (vouchers AND founder status cannot widen it), vouching threshold + Sybil-only-vouched refusal, transitive promotion, participation gating (no relay / no serving for non-admitted), end-to-end vouched join vs Sybil exclusion |
| `test_failover.py` | 261 | Liveness / dead-sequencer failover tests: prune/touch semantics, idle-but-beaconing clusters never shrink, the headline sequencer-death failover (expiry -> agreed next-ranked survivor -> request VERIFIES, no fork even with skewed pruning), per-depth single-candidate guard |
| `test_antientropy.py` | 383 | Anti-entropy repair tests (N=12/36/40): held-back and a dozen simultaneous laggards catch up to the exact head with no fork; digest-anchored adoption refuses no-anchor, tampered, and self-consistent forged batches; failover repair from a non-sequencer when the sequencer can't serve |
| `demo_local.py` | 163 | Full end-to-end demo over the in-memory bus |
| `demo_ecdh.py` | 148 | Group-key-agreement demo: confidential genesis salt + eavesdropper check |
| `juvian_transport.py` | 200 | `UDPTransport` (reliable+chunked) and `InMemoryTransport` (tests) behind one interface |
| `juvian_tensor.py` | 110 | Tensor map worker + MAD-filtering reducer |
| `juvian_dht.py` | 148 | Kademlia 160-bit XOR routing table (k=48: zone-complete views; liveness touch/prune) |
| `juvian_history.py` | 79 | Memory-mapped audit ledger |

---

## Run it

### Option A — one-command demo (no network needed)

```bash
pip install numpy cryptography
python3 demo_local.py
```

This is the fastest way to see the whole protocol work.

To see the **confidentiality layer** (group key agreement + an eavesdropper
failing to reconstruct the secret salt, then the chain running on top of it):

```bash
python3 demo_ecdh.py
```

### Option B — a real LAN mesh

Open three terminals on the same machine (or three machines on the same LAN).
All nodes **must share the same `--session-secret`**, or their genesis salts
differ and every verification rejects.

```bash
# terminal 1
python3 run_node.py --name anchor-1 --port 9101 --type ANCHOR

# terminal 2
python3 run_node.py --name mobile-1 --port 9102 --type MOBILE

# terminal 3
python3 run_node.py --name mobile-2 --port 9103 --type MOBILE
```

For multiple machines, also pass `--advertise-host <this-machine-LAN-IP>` so
peers reply to the right address. Discovery uses UDP broadcast on port 8001.

Then, at any node's prompt:

```
req hello world      # submit a request -> runs real quorum verification
tensor 8             # distributed tensor reduce over an 8x8 matrix
peers                # list discovered peers
chain                # chain depth + latest fingerprint
snap                 # full JSON snapshot
quit                 # persist chain and exit
```

### Optional — live web dashboard per node

```bash
pip install aiohttp
python3 run_node.py --name anchor-1 --port 9101 --web 8080
# open http://127.0.0.1:8080
```

The dashboard shows this node's real peers, chain depth, fingerprints, and lets
you submit requests / tensor jobs from the browser.

---

## Security model — what the system gives you

The **quorum π-Mandelbrot chain** provides:
- **Key-agreement integrity** — three independent nodes must derive the
  identical key from the same payload, or the round fails. A node whose chain
  has drifted refuses to adopt rather than corrupt its chain.
- **Tamper-evidence** — any change to the payload changes the coordinates and
  therefore every fingerprint; mismatches are rejected.
- **No key transmission** — only SHA-256 fingerprints cross the wire; the key
  never does.
- **Forward secrecy across the chain** — each key salts the next, so a single
  leaked key doesn't reconstruct the future chain without the payloads.

The **key-agreement layer** (`juvian_ecdh.py`, run via `establish_session()` /
the `kex` command) adds confidentiality on top:
- **Confidential genesis salt via group Diffie–Hellman.** Before any requests,
  members run **Burmester–Desmedt** contributory group key agreement: two
  broadcast rounds in which only public values (`zᵢ = g^rᵢ`, `Xᵢ`) travel the
  wire, after which every member independently computes the identical group key
  `K = g^(r₀r₁ + r₁r₂ + … + r_{N-1}r₀)`. That key becomes the genesis salt. An
  eavesdropper who captures the entire transcript still cannot compute `K`
  without a secret exponent `rᵢ` (the discrete-log / DH assumption) — so every
  chained π-Mandelbrot key now depends on a secret outsiders lack. `demo_ecdh.py`
  demonstrates the eavesdropper failing to reconstruct the salt.
- **Pairwise X25519 ECDH** is also provided (`PairwiseECDH`) for a direct
  two-party confidential link — genuine elliptic-curve Diffie–Hellman over
  Curve25519 using the vetted `cryptography` library.

What it still does **not** give you, by design:
- **Confidentiality against the verifying nodes themselves.** The three
  verifiers all derive the *same* key, so they can all decrypt. The scheme is
  consensus + integrity among participants, plus confidentiality against
  *outsiders* once the group salt is established — not secrecy from the
  participants who must verify.
- **Absolute Sybil resistance.** Identity creation is now *priced* by a
  proof-of-work birth certificate — a peer with no valid cert is refused
  admission, and grinding ids toward an XOR target costs work per id — but PoW
  does not *prevent* Sybils: a well-resourced attacker can pay the work, and the
  difficulty must stay low enough for weak devices. The **admission policy**
  (allowlist / vouching, off by default — see the architecture bullet) closes
  this for deployments that enable it by *bounding* the population instead of
  pricing it; with the policy off, the MAD filter remains the compute-side
  mitigation and the bound does not exist.
  On the KEX: every Burmester–Desmedt message is a signed, PoW-gated envelope,
  so an *outsider* man-in-the-middle cannot forge or alter a member's element,
  and a key-confirmation round — a majority must publish a matching derivation
  proof before anyone installs the genesis salt — makes a silent key-split
  impossible. An *insider* equivocation in **either** round (signing different
  elements to different members) cannot fork the group, and is now **proven and
  evicted**: an echo of the signed envelopes turns two conflicting signatures from
  one owner into a non-repudiable proof, the culprit is removed, and the honest
  remainder re-keys instead of the whole group aborting (see "Byzantine-robust
  BD" in `REMEDIATION_STATUS.md`). Residual liveness, not safety: a single
  *wrong-but-consistent* contribution (the same bad value to everyone) is not
  equivocation and remains a safe no-install abort — proving it would need a
  zero-knowledge proof of correct computation — and a member that simply stays
  silent stalls the round to a timeout. Only a *proven equivocator* is evicted.
- **Late-joiner sync for free.** A node that joins mid-session starts at the
  genesis salt and needs a re-key (a fresh `establish_session`) to rejoin the
  live chain.

On the Mandelbrot step specifically: most payload-derived coordinates escape
within the first few iterations (you'll often see `escape_iter = 1` in the
logs), so the entropy lives in the early iterations plus the final state, not in
1000 iterations of rich orbital dynamics. The final `SHA-256` makes the whole
thing a sound KDF regardless of how quickly the orbit escapes — the Mandelbrot
stage is a deterministic, payload-bound mixing function, not the security
guarantee on its own.

---

## Honest scope

- **Scale.** This is a real mesh of real OS processes exchanging real UDP
  datagrams — on the order of a handful to a few dozen nodes on a LAN. The
  **1,000,000-device "planetary substrate"** is the in-browser *simulation*
  (`juvian_1m_simulation.html`); those counters are mathematical state, not a
  million live processes. Don't cite the simulation's numbers as benchmarks.
- **Fractal compression.** The `.frac` writer stores the full reference data
  alongside the IFS transforms and recovers the chain from that reference. The
  IFS grid reconstruction is **lossy**, so this is a structured persistence
  format, not a net compression win — the "reassembles out of thin air" framing
  in the spec docs does not hold.
- **Telemetry logs / generated videos** included with the spec are illustrative
  artifacts, not captured output from a running system.

What's genuinely solid and worth keeping: the EWMA adaptive timeouts, the
Kademlia XOR routing, the MAD Byzantine filter, the hardware-weight governance,
the no-key-transmission verification handshake, and the memory-mapped ledger.

---

## License

Co-authored sovereign substrate, released as open source.

**Lead Author:** Jason M. Vajler
**Co-Author:** Dwayne Aubery

Copyright 2026 Jason M. Vajler & Dwayne Aubery.

Licensed under the **Apache License, Version 2.0** — see the [`LICENSE`](LICENSE)
file for the full terms and [`NOTICE`](NOTICE) for the attribution that
redistributions must preserve. Every source module carries the standard Apache
license header. Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
