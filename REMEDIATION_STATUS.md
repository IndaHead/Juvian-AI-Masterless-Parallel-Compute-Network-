# Juvian Grid — Remediation Status (verified)

*Juvian Grid — Lead Author: Jason M. Vajler | Co-Author: Dwayne Aubery. Copyright 2026. Licensed under the Apache License, Version 2.0 (see LICENSE).*

This supersedes the status implied by `AUDIT.md`. Important correction: that
audit was written from scratch without checking for remediation that earlier
work had already landed, so it materially over-stated what was still broken. The
table below reflects the **actual current code**, and every "FIXED" row is backed
by an assertion in `test_adversarial.py` (run it: `python3 test_adversarial.py`).

## Critical bug found and fixed this session

While building the verification suite I found the system was **completely
non-functional**: `ThreeWayVerification.submit_commitment` accepted a verifier's
proof but never stored it (`rnd.proofs[node_id] = proof` was missing), so no
round could ever reach quorum — every request timed out, 0 keys verified. This
was the half-finished state of an in-progress refactor. Fixed by storing the
proof before the quorum check. Both demos now pass again (4/4 and 3/3 verified),
and the earlier suspicion that a premature `drop_key` was nulling the key was a
red herring — `drop_key` is defined but never called.

## Status of each audit finding

| # | Finding | Status | Evidence |
|---|---------|--------|----------|
| 1.1 | 3-of-3 free-riding | **FIXED** | proof-of-derivation (HMAC bound to verifier id); copied proof rejected |
| 1.2 | BD unvalidated group elements | **FIXED** | `_valid_element` rejects 0/1/`P-1`/out-of-subgroup; no crash |
| 1.3 | KEX unauthenticated (MITM) | **FIXED** | KEX messages are signed envelopes (outsider MITM already blocked) + a key-confirmation round: a member installs the genesis salt only once a *majority* publishes a derivation proof matching the key it derived. Silent key-split is impossible. A round-1 **or** round-2 insider equivocation is now proven non-repudiably and the culprit **evicted**, with the honest remainder re-keying (Byzantine-robust BD); only a *wrong-but-consistent* contribution or an in-transit MITM remains a safe no-install abort — see "Authenticated KEX" and "Byzantine-robust BD" |
| 1.4 | Concurrent KEX splits genesis | **FIXED** | canonical initiator (lowest node id); only one node starts a KEX |
| 1.6 | π/sin entropy fold | **ACCEPTED** | cosmetic; SHA-256 keeps the KDF sound; documented |
| 1.7 | Key depends on ~128 bits | **ACCEPTED** | sufficient for a KDF; documented |
| 2.1 | No message authentication | **FIXED** | Ed25519 signed envelopes, self-certifying ids; spoof/tamper/forge all rejected |
| — | Replay of signed messages | **FIXED (new)** | nonce + timestamp in signed body; freshness window + bounded seen-cache; stale & duplicate rejected |
| 2.2 | Unsolicited compute DoS | **FIXED** | membership check (known peers only) + per-peer rate limit |
| 2.3 | MAD inverts under Sybil majority | **IMPROVED** | membership stops outsider yields; identity *creation* is priced by a proof-of-work birth certificate, and an off-by-default **admission policy** (allowlist / vouching) now *bounds* the population on top of pricing it — non-admitted ids never enter any view and cannot participate. Residuals (transitive-vouching coalition; policy is per-deployment) — see Open |
| 2.4 / 2.5 | UDP size limit / no reliability | **FIXED** | reliable chunked layer (`juvian_reliable.py`): unicast is fragmented + ARQ-retransmitted; a 100 KB message and a full tensor job both survive 20–30% datagram loss in tests |
| 2.6 | Unbounded task spawn | **FIXED** | inbound dispatch is capped (`MAX_INFLIGHT`) in the transport; load is shed under flood |
| 2.7 | Same-host bind collision | **FIXED** | `reuse_port=True` with graceful fallback |
| 2.8 | IPv6 address split | **OPEN** | `addr.split(":")` still breaks IPv6 (low priority) |
| 3.1 | Unvalidated reshape/frombuffer | **FIXED** | `_safe_decode_array` (size cap, dtype, shape match) + handler try/except |
| 3.2 | Memory leaks | **FIXED** | round eviction (512), KEX cap (64), in-memory entry cap (10k) + true-depth counter, seen-cache cap (8k) |
| 4.1 | Single-initiator only | **FIXED** | requests ordered through a deterministic sequencer, now chosen by **verifiable per-epoch rotation** over the chain head digest (no longer a grindable fixed lowest id); concurrent initiators produce identical chains on all nodes, no fork (test asserts equal depth + head fingerprint) |
| 4.2 | "3-of-3" sometimes 4-of-4 | **FIXED** | quorum is counted over exactly the selected verifier set |
| 4.3 | Single-verifier veto / liveness | **FIXED** | k-of-n quorum (default 3-of-5) with early resolution + retry on a disjoint verifier set; a silent verifier no longer stalls a request (test asserts liveness + that a retry occurs) |
| 4.4 | Verifier-selection grinding | **IMPROVED** | impersonation already blocked; the PoW birth certificate now also prices grinding — each accepted identity costs ~2**difficulty work, so cheaply minting ids to land near a target costs work per id. A patient/well-resourced attacker can still pay it — see Open |
| 5.1 | Fractal overhead / lossy | **ACCEPTED** | documented; replace with signed JSON/append-log if exact history matters |

## What remains open (in priority order)

1. **Wrong-but-consistent contributions + admission policy.** Round-**1** *and*
   round-**2** equivocation are now resolved: a member that signs two conflicting
   values (z or X) is proven and evicted and the honest remainder re-keys, instead
   of the whole KEX aborting (see "Byzantine-robust BD" below). What remains a safe
   **no-install abort** — never a fork — is a member that sends a single
   *wrong-but-consistent* value (the same bad X, or z, to everyone): that is not
   equivocation, so the echo cannot prove it, and proving a contribution
   mis-computed would require a zero-knowledge proof of correct BD computation,
   which BD does not carry. The confirmation round (keys diverge → no majority →
   nobody installs) stays the backstop for it and for an in-transit non-member
   MITM. A member that simply stays **silent** (withholds a z, X, or echo) still
   stalls the round to a timeout, exactly as withholding any round always did —
   silence is not a provable offence; this is the baseline liveness class. Plus an
   admission policy for new peers (see item 2).
2. **Residual Sybil cost (2.3 / 4.4) — now BOUNDED when the policy is enabled.**
   Proof-of-work *prices* identity creation but does not *prevent* it. An
   **admission policy** is now implemented on top of PoW (never replacing it,
   and **off by default** so existing deployments are unchanged): an
   **allowlist** admits only configured ids (airtight for a closed fleet, and
   authoritative — neither vouchers nor founder status can widen it), and
   **vouching** admits a newcomer once it presents ≥ `vouch_threshold` signed
   vouchers from *distinct already-admitted* members (founders seed the root of
   trust; vouchers are self-certifying signed objects bound to the subject id
   and carried on its beacons). Admission gates **participation**, not just
   visibility: a non-admitted node never enters a routing view (so it is
   invisible to sequencer rotation, verifier selection, and the attestation
   quorum), is refused sequencing, is not served chain repair, and its
   gossip-marked traffic is not relayed. Honest residuals: vouching is
   **transitive**, so a coalition of `vouch_threshold` careless-or-malicious
   admitted members can still admit Sybils (the allowlist is the airtight
   control); vouchers carry a signed issue time but do not expire by default
   (expiry is a deployment choice the format already supports); and policy
   *consistency* across a zone is the operator's responsibility — nodes with
   divergent policies see divergent memberships, which degrades liveness, never
   key integrity. Sequencer selection is **no longer id-grindable** —
   leadership rotates verifiably per epoch via the chain head digest (see "New
   tradeoff introduced by the 4.1 fix") — with two bounded residuals: the
   genesis epoch is still grindable (one term), and an *incumbent* leader can
   bias the next epoch's seed via boundary-slot payload choice; full
   unbiasability needs commit-reveal / threshold randomness.

## Authenticated KEX — key-confirmation round (1.3, FIXED)

The Burmester–Desmedt messages were already signed envelopes (PoW-gated,
identity-bound, tamper-proof), so an *outsider* man-in-the-middle was already
blocked — it cannot forge or alter a member's element in transit. The two gaps
signing alone left were (a) **no key confirmation** (any disagreement installed
silently) and (b) **insider equivocation** — a legitimate member signs two
different round-1 elements and feeds a different one to a chosen victim (each
validly signed by its own key), classically splitting the group key while the
victim believes it is still in the group.

The fix is a confirmation round modelled on the verify quorum's
proof-of-derivation. Before installing the derived genesis salt, every member
publishes `HMAC(key = SHA-256("JUVIAN_KEXCONF_KEY::" ‖ salt); msg =
"JUVIAN_KEXCONF::" ‖ kex_id ‖ roster_hash ‖ member_id)` — a one-way proof of
*which* key it derived. The tag reveals the MAC, never the salt. A member
installs only once a **majority** of the roster has published a tag that matches
the key *it* derived, verified by recomputing each tag with the verifier's own
salt; a non-matching tag counts as zero, exactly like a wrong proof in the
verify quorum. `roster_hash` is recomputed locally and never transmitted, so a
divergent roster yields a divergent tag and silently fails to count; `kex_id`
makes every tag unique per KEX, so a confirmation cannot be replayed into
another exchange.

Property delivered: **agreement-or-abort.** Because two disjoint majorities
cannot exist, at most one salt can ever be installed network-wide — no
split-brain. An equivocation victim falls short of quorum and *refuses* to
install rather than silently forking (the meaningful change: before this round
it installed on derivation). The honest cost is liveness, not safety: a BD
member re-broadcasts a round-2 value derived from the element it received, so an
equivocation perturbs every member's key and the KEX aborts for everyone — a
forced re-key, never a fork. The Byzantine-robust follow-up — **detect, prove,
evict the equivocator, then re-key the honest remainder** — is now implemented;
see the next section. The confirmation round remains the backstop beneath it for
an in-transit non-member MITM and for a *wrong-but-consistent* contribution
(both equivocation rounds are now caught and evicted by the echoes).

Verified by `test_authkex.py`: a 5-node honest run still converges on one shared
salt with the added echo round (5/5 install, 1 distinct salt); and a 5-node run
where one member signs a corrupt round-1 element only to a chosen victim now ends
in that member's **eviction** — the honest four re-key (4/5 install, 1 distinct
salt, no fork), the steered victim recovers onto the group salt, the initiator
reports ESTABLISHED, and the culprit installs nothing.

## Byzantine-robust BD — round-1 + round-2 echo → eviction → re-key (DONE)

The Burmester–Desmedt equivocation attack has two forms: **round-1** (an insider
signs a different z_i to a chosen victim) and **round-2** (it signs a different
X_i). Each is validly signed by the culprit's own key. The confirmation round
above already makes both *safe* (the perturbed keys never reach a confirming
majority, so nobody installs), but the honest cost was a full abort, and a naive
retry re-admits the equivocator — a persistent equivocator could stall every
attempt forever (a liveness DoS). This section closes that for **both rounds**.

- **Round-1 echo.** Once a member holds every round-1 z, it broadcasts the
  *signed envelopes* it received (`KEX_ECHO`) and withholds its round-2 value
  until the echoes are in and consistent. Round-1 messages are already Ed25519
  envelopes, so a retained envelope is a non-repudiable signed statement of
  "(kex_id, owner) → z".

- **Non-repudiable proof.** Merging echoes yields, per owner, the set of distinct
  signed z values seen. An honest broadcaster sends the *same* envelope to all
  (one z); an equivocator must emit two separately-signed envelopes with different
  z. Two distinct, validly-signed z for one owner+kex_id is therefore a complete
  proof — and a victim that only ever received one of them **harvests** the
  conflicting one from a peer's echo (each carried envelope is verified against
  the owner's own key, never the echoer's say-so). **Framing an honest member is
  impossible**: a second signature under its key cannot be forged.

- **Eviction + victim-only re-key.** On a verified proof the culprit is added to a
  persistent exclusion set and a self-contained `KEX_EVICT` (carrying the two
  envelopes) is broadcast; every receiver **re-verifies the proof itself** before
  excluding. The lowest-id *surviving* member then re-keys the honest remainder on
  the reduced roster (a single re-initiator, so re-keys don't race; bounded —
  each eviction permanently removes one distinct id). The original caller's
  request is carried across the re-key, so it still completes. If equivocations
  shrink the roster below two members, the re-key fails cleanly with an error.

- **Round-2 echo (the same construction over X).** Once a member holds every
  round-2 X it broadcasts the *signed X envelopes* it received (`KEX_ECHO2`) and
  withholds its key-confirmation until that view is consistent. By this point the
  roster and every z are pinned and echo-consistent, so X_i = (z_{i+1}/z_{i-1})^{r_i}
  is a deterministic function of public values and the owner's fixed secret — an
  honest member emits exactly one X_i, and two distinct, validly-signed X for one
  owner+kex_id is a non-repudiable round-2 proof, harvested and acted on exactly
  like round 1. One new subtlety: an honest member legitimately emits both a z
  (round 1) and an X (round 2), so the proof check requires both halves to be the
  **same round** — pairing a z with an X is explicitly rejected, or an honest
  member's own two values would frame it.

- **Honest scope (what is and isn't covered).** Both round-1 and round-2
  *equivocation* (two conflicting signed values) are now proven and evicted. What
  remains a safe no-install abort — **not** an eviction — is a member that sends a
  single **wrong-but-consistent** value (the same bad X, or z, to everyone): that
  is not equivocation, and proving a contribution mis-computed would need a
  zero-knowledge proof of correct BD computation, which BD does not carry, so the
  confirmation round (no quorum → no install) stays the backstop for it and for an
  in-transit non-member MITM. And a member that simply stays **silent** (withholds
  a z, X, or echo) without equivocating still stalls the round to a timeout,
  exactly as withholding any round always did — silence is not a provable offence;
  this is the baseline liveness class, which already required every round-1 z.

Verified by `test_byzkex.py`: an honest N=4 run converges with both echo rounds
and evicts no one; three classes of **forged** round-1 proof (identical-z, a
tampered bad-signature envelope, two-different-owners) are all rejected while a
genuine one is acted on (positive control); an equivocator that is itself the
**initiator** (lowest id) is still evicted, with the honest remainder re-keying
under the next-lowest survivor; and for round 2, a **cross-round** "proof" (a
member's own legitimate z paired with its X) is correctly rejected by the
same-round guard while a genuine two-conflicting-X proof evicts, and a
non-initiator that equivocates only in round 2 (a good X to all, a bad X to one
victim) is proven, evicted, and the honest remainder re-keys to one salt with the
victim recovering and the initiator reporting ESTABLISHED. Both KEX suites were
run repeatedly (node ids are random) and pass every time.

## New tradeoff introduced by the 4.1 fix (be aware of this)

Ordering flows through a **sequencer**. Originally this was the lowest node id —
deterministic and election-free, but **permanently capturable**: one offline grind
of a low `node_id = SHA-256(pubkey)[:40]` bought standing reorder/censor power in
any membership the node joined. The sequencer is now chosen by **verifiable
rotation**: for the epoch containing chain depth `d` (epochs are
`SEQUENCER_TERM = 32` slots), the sequencer is the member minimising
`H(epoch_seed ‖ epoch ‖ id)`, where `epoch_seed` is the verifiable chain **head
digest at the epoch boundary**. Properties, honestly stated:

- **Verifiable & election-free.** The winner is a pure function of public,
  chain-agreed inputs (member ids, boundary digest, epoch index); any third party
  recomputes it. No election traffic, same as before.
- **Degrindable for non-incumbents.** The hash makes the id's *value* irrelevant —
  every member is equally likely to lead each epoch (the adversarial suite sweeps
  400 epochs over 16 members: all 16 lead, top share ~0.08 vs uniform 0.0625, the
  lowest id holds exactly its fair share). Influence is proportional to the number
  of PoW-priced identities — the Sybil cost the system already assumes — instead
  of one cheap offline grind.
- **Ordering power only, time-sliced.** The sequencer still cannot forge keys
  (members adopt only their OWN re-derived, fingerprint-checked key) or fork the
  chain (single-winner-per-slot adoption; repair is quorum-anchored). What
  rotation changes is that reorder/censor power now expires every term instead of
  being permanent.
- **Liveness across a sequencer death — now CLOSED (bounded stall).**
  Membership is liveness-maintained: the beacon loop doubles as a maintenance
  tick (any peer silent beyond `PEER_EXPIRY_S = 30 s` — no beacon, no
  authenticated traffic — is pruned from the view), so a dead sequencer drops
  out of `_members()` everywhere within roughly one expiry window, and the
  hash-argmin **automatically recomputes to the next-ranked live member** — the
  deterministic fallback IS the selection rule over the live set; no extra
  election protocol. Forwarding retries with a recomputed target each attempt
  (and sequences directly if expiry promotes the origin itself), results are
  accepted only from the node a request was actually sent to, and a stopped
  node now truly goes dark instead of lingering as a half-alive ghost. What was
  a **permanent** stall is now bounded by ~`PEER_EXPIRY_S` + one beacon tick;
  a request submitted *inside* that window can still time out, honestly — it is
  a bounded outage, not zero outage. (Verification, KEX, and key derivation
  remain fully distributed.) Two further honest notes on the failover window:
  a **per-depth single-candidate guard** (`DEPTH_GUARD_S`) makes each member
  honor at most one opener per chain slot at a time (same-opener quorum-retries
  pass; a dead claimant's hold lapses after the guard window), which narrows the
  dual-candidate race that a transient expiry-skew view split could open — and a
  candidate that cannot win a member's author check still cannot normally
  complete a round, because the verification quorum applies the majority view.
  But a true network **partition** in which *both* sides retain a verifier
  quorum could still extend two divergent chains; expiry-based failover narrows
  in-zone races, it is **not** partition consensus, and pruned peers' head
  attestations are dropped with them so catch-up anchoring always reflects the
  live membership.
- **Membership-view agreement matters — now sized correctly.** Honest nodes must
  agree on the membership to agree on the winner. The Kademlia bucket capacity was
  raised from k=20 to **k=48** so that a *zone* (the stated ≤40–50-node operating
  envelope) fits entirely in every member's routing view — at N=50, k=20 silently
  dropped ~5 of the ~25 far-half peers per node, which the old fixed-min rule
  mostly tolerated but uniform rotation (whose winner is a random member each
  epoch) exposed as occasional sequencer splits. With zone-complete views,
  agreement is exact; **beyond ~2×k members per zone, partial views return** and
  membership would need to move on-chain / into attested rosters — that is the
  documented federation boundary, not a silent failure mode.

**Honest residuals of the rotation itself:** (a) epoch 0's seed is the public
genesis constant, so the *first* term is still id-grindable — bounded to one term;
(b) an **incumbent** sequencer can bias the next boundary digest by choosing or
placing the payload that lands on the boundary slot (it orders requests), i.e.
seed-grinding by the current leader remains possible — but it is active,
per-epoch, on-chain-visible work available only to a node that already leads,
rather than a one-off offline grind available to anyone; closing it fully needs
commit-reveal or threshold randomness (future work); (c) transient view
disagreement during churn/partition causes rounds to be ignored (safe — they fail
rather than fork) until views reconverge, exactly as before.

## Semantic change introduced by the 4.3 fix (be aware of this)

Consensus moved from **strict 3-of-3** (all three selected verifiers must prove,
or the request fails) to a **k-of-n quorum** — by default 3 valid proofs out of
up to 5 invited verifiers (`verifier_quorum` / `verifier_fanout`, both
per-node-configurable). Two honest points:

- **Why this does not weaken integrity.** Each proof is checked against the
  sequencer's *own* derived key, so an absent or wrong proof simply does not
  count toward the quorum — it can never push a *wrong* key through. The headline
  property ("at least `k` independent devices confirmed this exact key") is
  preserved; only the requirement that one *specific* set of three all agree is
  relaxed. Tolerating up to `n − k` non-responding/misbehaving verifiers is
  therefore a pure liveness gain.
- **What it costs.** The guarantee is now "≥ k of the invited verifiers agreed",
  not "these particular k did". The code never silently drops below
  `verifier_quorum`: if fewer than `k` distinct verifiers can be invited (small
  network, or too many excluded after retries), the request fails rather than
  confirm on a thinner quorum. Liveness slack only exists when the network has
  spare verifiers beyond `k`.

## Sybil pricing via proof-of-work (be aware of this)

To be admitted, a node must present a **proof-of-work birth certificate**: a
nonce whose hash over its public key clears a difficulty target (leading zero
bits). It is checked at the same chokepoint as the signature, so a peer without a
valid cert is never added to routing and its messages never reach a handler.
`node_id` is still `SHA-256(pubkey)`, so routing / XOR / sequencer logic is
unchanged.

- **What it buys.** Identity *creation* now costs ~`2**difficulty` hashes each,
  pricing the damaging case (a Sybil majority drowning the tensor-aggregation MAD
  filter) and the grinding of ids toward a target. One-time per identity,
  verifiable by anyone with a single hash, no coordinator.
- **What it does NOT buy.** PoW *prices* Sybils, it does not *prevent* them: a
  well-resourced attacker can pay the work, and the difficulty must stay low
  enough to be feasible on weak edge devices (the demo default is modest). For a
  closed fleet, layer an allowlist / vouching policy on top. Difficulty is
  per-node configurable (`pow_difficulty`, 0 disables); a node also rejects peers
  whose cert is below its own required difficulty.

## Geo-zones: locality only, not consensus sharding (option A)

`geo_zone` steers **tensor compute** toward same-region peers to cut cross-region
traffic. It deliberately does **not** shard consensus: request ordering still
flows through the single global sequencer, so the key chain stays fork-free
(4.1). Per-zone sequencing would let two zones claim the same chain slot and fork
the chain — only safe with a separate chain per zone, which this does not do. To
avoid starving the Byzantine filter, zone-local sharding keeps a floor
(`MIN_TENSOR_CONTRIBUTORS`, incl. the local contribution): a small zone borrows
the nearest out-of-zone peers rather than shrink the MAD sample.

## Cooperative thermal/battery load-shedding (not a security control)

An optional `HardwareTelemetryMonitor` (off by default) lets a hot or low-battery
node zero its own compute weight — the existing tensor governor then drops
incoming tasks — and emit a signed handover signal so peers down-weight it. This
is a *cooperative* feature for honest nodes only: temperature, battery, and
`hw_weight` are self-reported and unverifiable, and the handover is authenticated
so a node can only down-weight *itself*. It provides no defense against a
malicious node, by design.

## Iterative Kademlia discovery (scale follow-up, DONE — with honest scope)

The scale study flagged the demo's all-to-all beacon flood as O(N²) and
non-converging past a few dozen nodes. Added a real Kademlia lookup:
`FIND_NODE` / `FIND_NODE_REPLY` plus an iterative `_lookup(target)` (queries the
ALPHA closest unqueried contacts, merges replies, until nothing new) and a
`bootstrap(seed)` that joins from a SINGLE seed address — self-lookup for nearby
buckets, random-target lookups for distant ones. FIND_NODE also carries presence
metadata, subsuming the beacon's role.

- **Security.** Routing is populated ONLY from verified replies (every inbound
  message clears signature + PoW + id-match before `_note_peer`). The (id,addr)
  pairs inside a reply are mere hints for whom to query next — an id enters the
  table only once that node proves it with a signed message, so a peer cannot
  inject a claimed id it does not hold the key for.
- **What's proven (tests).** A node joins from one seed (no flood) and reaches
  near-complete coverage (≈38 of 39 on average); an iterative lookup lands on the
  true XOR-closest node — exactly so in ~97% of random targets, and never worse
  than the second-closest (exact-closest is *not* a hard guarantee once buckets
  cannot hold the whole roster, k=20 < N−1); lookup cost grows ~log N (a handful
  of queries — still single digits at N=80 — i.e. sub-linear, not O(N)); and the
  consensus runs unchanged on the lookup-discovered roster (verify round + full
  lockstep).
- **Honest limits.** At demo scale the flood is actually *cheaper* in raw
  messages (iterative's win is asymptotic: O(N·log N) vs O(N²)). The
  COMPLETE-roster property holds only for zone-sized networks (~≤40–50 here);
  beyond that a Kademlia table holds O(k·log N) contacts, not everyone — correct
  DHT behaviour. At large scale the consensus's two costs are *selecting*
  verifiers and *notifying* members: both are now handled — see "Lookup-based
  verifier selection" and "Gossip dissemination" below.

## Lookup-based verifier selection (scale follow-up, DONE)

Verifier selection used to scan the whole roster (`routing.peer_ids()`) for the
payload's XOR-closest verifiers — which presumes a node holds the whole roster,
exactly the assumption that breaks past zone scale. The sequencer now calls
`_verifier_pool(payload_hash)`, which runs an iterative Kademlia `_lookup` toward
the payload and selects from the `closest()` result. The lookup runs on every
request but is cheap when redundant (a complete table returns nothing new and it
terminates in one round); it is deliberately *not* gated on local table size,
because holding `fanout` contacts says nothing about whether they are the ones
near *this* payload — a size gate silently picks far nodes at scale.

- **Behaviour.** At zone scale this is identical to the old full-roster scan:
  the lookup is redundant and `closest()` already returns the true XOR-closest
  fanout (test: selection == full-roster's closest fanout, 6/6 targets). From a
  deliberately sparse table (2 contacts) the lookup re-discovers the payload's
  neighbourhood and selection recovers it — the true closest verifier every
  time, and a majority of the closest set with it (the exact k-set isn't
  guaranteed from a sparse start: same k-bucket boundary as discovery). A request
  still VERIFIES end-to-end through this path.
- **What it does and doesn't do.** It removes the full-roster requirement from
  *choosing* verifiers — the selection half of scaling the verify round. The
  *notification* half — announcing the round to every member — is handled
  separately by gossip dissemination (next section).
- **Security.** Unchanged: selection is XOR-closest-to-payload as before, so the
  payload (not the sequencer's whim) determines the neighbourhood, and PoW still
  prices grinding ids toward it; each verifier is still validated individually by
  its own proof-of-derivation. A stronger property — members independently
  recomputing the expected verifier set and rejecting a sequencer that names
  far-from-payload verifiers — would need an agreed neighbourhood view and is
  noted as a possible hardening, not yet implemented.

## Gossip dissemination (scale follow-up, DONE — with honest scope)

The per-request verify broadcasts (`VERIFY_OPEN`, `VERIFY_RESULT`) used a flat
reliable broadcast: the sequencer sent one copy to **every** known peer. That is
an O(N) fan-out *at the sequencer* on the hot path and the last roster-wide cost
in the verify round. It is now disseminated by **gossip**: the sequencer pushes
to `GOSSIP_FANOUT` (6) peers, and each peer re-forwards the message on first
receipt to its own fan-out, so a message floods the overlay in ~O(log N) hops
with **O(fanout) per-node** cost. The origin's signed envelope (and its nonce) is
forwarded unchanged, so every node authenticates the original author and dedups
network-wide on `(origin, nonce)` — the existing replay cache *is* the gossip
dedup, and it also terminates forwarding loops. KEX broadcasts are deliberately
NOT gossiped: that is a one-time setup/re-key path, and gossip's redundant
delivery would change the equivocation adversary model the authenticated-KEX
confirmation round is analysed against.

- **Size gate (no regression at zone scale).** `_disseminate` keeps the
  GUARANTEED reliable broadcast at or below `GOSSIP_MIN_ROSTER` (32) and only
  switches to gossip above it. So at zone scale delivery — and therefore hard
  lockstep — is unchanged and fully deterministic (the N=3 lossy-lockstep and
  N=24 bootstrapped-consensus tests pass exactly as before). Gossip engages only
  where a flat fan-out is the real cost. The strategy is carried in the message
  (a `_g` marker the origin sets), so relays forward consistently regardless of
  their own local view.
- **Verifiers-only commit.** A companion change: every member still *derives* the
  key on `VERIFY_OPEN` (so it can adopt the result and stay in lockstep), but only
  the **selected verifiers** send a `VERIFY_COMMIT` back. The sequencer counted
  only the selected set anyway, so this drops its *inbound* commit load from O(N)
  to O(fanout) without changing the outcome.
- **Measured (N=50, above the gate).** Per-node fan-out is bounded — the busiest
  node sends ~6 messages, versus 49 for a flat broadcast. A single gossip push
  reaches the mesh with high probability (averaged ~49.9/50, worst ~49/50 across
  pushes). A verify request still VERIFIES end-to-end over the overlay, and every
  member that adopts lands on the **same** chain head (no fork / no split-brain).
- **Honest limit — coverage is high-probability, not guaranteed.** Push-gossip
  cannot guarantee full coverage from fan-out alone; a node missed by a flood
  cannot derive/adopt that round and stays behind. In one-shot end-to-end runs a
  *majority-to-near-all* adopt the head (typically 45–50/50), but the figure dips
  (≈37–40/50) on rounds that hit a **verifier-quorum retry** — a retry is a second
  `VERIFY_OPEN` flood whose coverage is independent of the first, so a member
  reached only by the first flood derived a key for the superseded session and
  stays on genesis. There is never a fork, only a lagging tail. The companion that
  closes the residual tail at large scale is **anti-entropy repair**: a member
  that sees a chain index ahead of its own pulls the slots it missed and re-derives
  them. This is now **implemented** — see "Anti-entropy repair" below — and drives
  the lagging tail back to full/near-full lockstep. (A direct-to-selected-verifiers
  unicast alongside the flood would further cut the retry-induced lag at the source.)
  Below the gate, the reliable broadcast already gives hard lockstep.

## Anti-entropy repair (scale follow-up, DONE — with honest scope)

Gossip (previous section) delivers a verify round to the mesh with *high*
probability, but a node missed by a flood derives nothing for that round and
falls behind — a lagging tail, never a fork. Anti-entropy repair closes that gap
by letting a behind node **pull the slots it missed and re-derive them**, so the
tail converges back to full/near-full lockstep.

- **Detect-ahead, then pull from several holders.** Every `VERIFY_OPEN` /
  `VERIFY_RESULT` carries the round's `chain_index`. A node that sees an index
  ahead of its own depth knows it is behind and (rate-limited, one batch of
  requests at a time) sends `CHAIN_REQUEST{have_depth}` to up to `CATCHUP_FANOUT`
  distinct archive-holders — the current sequencer **and** other peers its signed
  beacons show to be at least as deep — so repair proceeds even if the sequencer
  has departed or is the wrong/ambiguous node mid-handover. A second, *proactive*
  trigger rides the beacon: every `BEACON` / `BEACON_ACK` now advertises
  `chain_depth` **and** `chain_digest`, so a node isolated from the verify floods
  (but still beaconing) still notices a higher peer depth and requests catch-up,
  and simultaneously learns the head digests it will anchor against.

- **Every member serves an archive ring.** When a round VERIFIES, **every**
  member (not just the sequencer) records `{chain_index → (payload, iterations,
  verifiers, fingerprint)}` in a bounded ring (last `CHAIN_ARCHIVE_MAX` slots) —
  on the live adoption path and on the catch-up path alike. On a `CHAIN_REQUEST`
  any member replies with `CHAIN_BATCH{rounds: [...]}` covering the requester's
  missing slots, capped at `CHAIN_BATCH_MAX`. So repair no longer depends on the
  one node that happened to be sequencer when a slot was committed.

- **A verifiable head digest binds the canonical chain.** The chain maintains a
  cumulative `head_digest = H(… H(H(seed ‖ idx₀ ‖ fp₀) ‖ idx₁ ‖ fp₁) …)` over the
  ordered (chain_index, fingerprint) pairs. Two members that adopted the same
  rounds in the same order share the **identical** digest, so it is a compact,
  order-binding commitment to the whole chain. `BEACON` / `BEACON_ACK` now
  advertise `chain_depth` **and** `chain_digest`; since beacons are signed
  envelopes, each is a non-repudiable head attestation by that peer.

- **Catch-up is anchored to a quorum (or the sequencer), so ANY server is safe.**
  This is the key safety point, and it is *stronger* than the previous model. A
  catching-up member re-derives each slot from *its own* salt and refuses any slot
  whose re-derived fingerprint disagrees (a tampered payload is rejected exactly as
  before). But re-derivation alone is only self-consistent, **not** canonical: an
  insider that knows the genesis salt could serve a fully forged-yet-self-consistent
  chain whose every fingerprint re-derives cleanly. The digest closes that: a batch
  from any server is adopted only up to the **longest prefix whose re-derived head
  digest is anchored** — vouched for by a **quorum** of distinct peers' signed
  beacons, or by the current sequencer's beacon. A forged sequence re-derives to a
  head digest no honest quorum (and no honest sequencer) attested, so it is refused.
  For backward compatibility and bootstrap, a batch *from the current sequencer* is
  still trusted directly (the prior single-sequencer model); the quorum-anchored
  path additionally tolerates a malicious server — **or even a malicious sequencer**
  — because honest members attest only the canonical digest. (`test_antientropy.py`
  asserts: a tampered payload, a no-anchor batch, and a *self-consistent forged
  sequence* with correct per-slot fingerprints but the wrong head digest are all
  refused, while the genuine rounds — quorum-anchored, from a non-sequencer — are
  adopted.)

- **Result.** With repair engaged, a deliberately-held laggard — and a dozen
  simultaneous laggards — pulled back to the **exact** sequencer head with no
  fork; the lagging-tail figure from gossip-only runs converges to full/near-full
  lockstep.

- **Honest residuals (liveness, never safety).**
  - *Partial-view anchoring.* Repair now anchors on a quorum of head attestations
    or the current sequencer's, and pulls from several archive-holders. A node
    whose routing view is too sparse to gather a quorum of attestations (and that
    also cannot see/address the sequencer) cannot yet anchor a catch-up, so it
    stays behind until discovery fills its view in. This is a discovery /
    routing-completeness limitation, not a repair flaw — no node ever adopts a
    wrong key as a result; it simply waits. (At zone scale the routing tables
    converge well within a quorum, so this is a cold-start / heavily-partitioned
    edge case.)
  - *Cross-sequencer historical repair — now CLOSED.* Previously only the
    sequencer archived, so a post-handover sequencer had no history for slots
    predating its tenure. Every member now archives and serves, and adoption is
    bound to a quorum/sequencer-attested **head digest**, so a laggard repairs from
    any holder and converges to the canonical head even when the original
    sequencer is gone. `test_failover_repair_without_original_sequencer` drives a
    laggard to the exact head (matched by salt **and** digest, no fork) while the
    sequencer is made unable to serve — the repair is carried entirely by a
    non-sequencer, quorum-anchored.

## Reliable broadcast + a pipelining race it exposed (FIXED)

The critical broadcasts — KEX rounds and VERIFY_OPEN / VERIFY_RESULT — were
chunked but best-effort: one dropped datagram and a member computes the wrong
group key (a permanent split) or misses a chain advance (falls out of lockstep).
They now go out via **reliable broadcast**: the body is signed once and delivered
to every known peer over the reliable unicast path (per-peer selective-repeat
ARQ). BEACON stays best-effort (it repeats every 5 s; loss is self-healing).
Scope is the known roster — a peer we don't yet know isn't reached, the same
limit any broadcast has.

Testing this over a lossy/reordering transport surfaced a **pre-existing
pipelining race** that the synchronous in-memory bus had always hidden: the
transport ACKs a datagram on *receipt*, but the adoption handler runs as a
scheduled task, so the sequencer could open request N+1 before a member had
adopted request N. With a small membership (quorum = all members) that one stale
member derived on the previous salt and the round failed spuriously. Fix: a
member now **buffers a VERIFY_OPEN whose `chain_index` is ahead of its own
chain** and processes it the instant the chain catches up (symmetric to the
existing early-VERIFY_RESULT buffer). A lagging member's vote is simply *pending*
rather than *wrong*, so the round waits briefly and still reaches quorum instead
of rejecting. No effect on the synchronous path.

## How this was verified

`test_adversarial.py` asserts, against the live code: signed-envelope
spoof/tamper/forge rejection; replay freshness + duplicate suppression;
free-rider rejection with the honest path still verifying; BD bad-element
rejection without crashing; malformed/oversized tensor decode returning safely;
tensor membership + rate limiting; round-table eviction; a single canonical
KEX initiator; concurrent initiators producing identical fork-free chains; the
**verifiable rotating sequencer** — every node computing the same winner, the
winner independently recomputable from public inputs alone, leadership spreading
across all 16 members over a 400-epoch sweep with the lowest id holding only its
fair share (so a ground low id buys nothing); a
non-sequencer being unable to drive a round; and — for the quorum (4.3) — a
request still verifying when one selected verifier is silent, plus a request
that has to retry onto a *disjoint* verifier set (because the first set can't
reach quorum) and still succeeds; that a no-PoW Sybil is denied admission to the
mesh while a properly-stamped peer is admitted, and that minting cost scales with
difficulty; that zone-local tensor sharding stays in-zone for a large zone but
borrows neighbours to hold the Byzantine floor for a small one; and that a
simulated overheating node sheds load and peers down-weight it. A separate
`test_reliable.py`
proves the transport layer: a 100 KB message is fragmented and reassembled; a
message and a full distributed tensor job both survive 20–30% datagram loss
(plus duplication and reordering) via retransmission; duplicates are delivered
exactly once; and — for reliable broadcast — a group KEX completes with every
member agreeing on the same secret genesis salt over a 20%-loss network, and
three pipelined verify rounds keep every member in lockstep (same chain depth and
head) over a 30%-loss network. `test_authkex.py` proves the key-confirmation
round and the Byzantine-robust eviction: an honest 5-node KEX still converges on
one shared genesis salt with the added echo round, and an insider that signs a
corrupt round-1 element only to a chosen victim is now proven and **evicted** —
the honest four re-key onto one salt (no fork), the victim recovers, the initiator
reports ESTABLISHED, and the culprit installs nothing. `test_byzkex.py` adds the
edge cases: an honest N=4 run evicts no one; three classes of forged proof are
rejected while a genuine one is acted on; and an equivocator that is itself the
initiator is still evicted, the remainder re-keying under the next-lowest
survivor. `test_discovery.py` proves iterative Kademlia lookup
lands on the true XOR-closest contact (exactly so for the large majority of
targets, never worse than the second) at roughly logarithmic query cost, that
consensus runs unchanged on a bootstrapped roster, and that verifier selection
runs on a lookup — picking the payload's closest fanout (identical to a
full-roster scan at zone scale, and recovered from a deliberately sparse 2-contact
table at the boundary). `test_gossip.py` runs *above* the size gate (N=50) so
gossip is engaged, and proves the per-node fan-out is bounded (busiest node ~6
sends vs 49 for a flat broadcast), that a single gossip push reaches the mesh
with high probability (averaged ~49.9/50), and that a verify request still
VERIFIES end-to-end over the overlay with every adopter on the same head (no
fork) — while being explicit that full one-shot lockstep is high-probability, not
guaranteed for gossip alone. `test_antientropy.py` then verifies the repair that
closes that gap: a member deterministically held behind (verify gossip and its own
catch-up suppressed) is driven to a lower depth than its peers, then — once
released — pulls the missed slots, re-derives each from its own salt, and lands on
the **exact** head (no fork); a dozen simultaneous laggards all converge the same
way; catch-up adoption is **digest-anchored**, so a no-anchor batch, a tampered
payload, and a *self-consistent forged sequence* (right per-slot fingerprints,
wrong head digest) are all refused while the genuine rounds — quorum-anchored, from
a **non-sequencer** — are adopted; and a **failover** case drives a laggard to the
canonical head (matched by salt and digest) from a non-sequencer while the original
sequencer is unable to serve. `test_failover.py` verifies the liveness layer:
prune/touch semantics at the routing table; an idle-but-beaconing cluster never
shrinking across multiple expiry windows; the headline sequencer-death failover —
the dead winner expiring from every survivor's view within ~one expiry window,
all survivors agreeing on the same next-ranked live sequencer (independently
recomputed from public inputs over the survivor set), a post-failover request
verifying end-to-end, and skewed vs. non-skewed pruners landing on ONE chain (no
fork); plus the per-depth single-candidate guard (different-opener race refused,
same-opener quorum-retry allowed, lapsed claim retaken by the successor).
`test_admission.py` verifies the admission policy: voucher primitives (a valid
voucher verifies; subject-swap, forged-issuer, tampered-field, and wrong-key
vouchers are all rejected); off-by-default no-regression; allowlist authority
(an unlisted-but-PoW-valid peer never enters the view, and neither vouchers nor
founder status can widen the list); the vouching bound (≥ threshold
admitted-issuer vouchers admit; under-vouched and *Sybil-only-vouched* newcomers
are refused, and self-vouching never counts); transitive promotion in
voucher-arrival order; participation gating (a non-admitted origin's
gossip-marked traffic is not relayed and its chain-repair requests are not
served, each with an admitted-peer control, and a voucher flood cannot grow the
pending cache past its bound); and end-to-end — a properly vouched newcomer
joins a live mesh and gets a request VERIFIED while a Sybil stays out of every
honest view and cannot.
`demo_local.py` and `demo_ecdh.py` confirm the happy
path (4/4 and 3/3 verified, group salt agreed, eavesdropper defeated).
