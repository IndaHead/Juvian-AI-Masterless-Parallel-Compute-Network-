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
JUVIAN GRID :: NODE DAEMON
The end-to-end node. Ties together:
  * discovery (LAN broadcast beacons + peer table)
  * Kademlia routing
  * pi-Mandelbrot 3-of-3 key verification
  * session key chain (+ fractal-indexed persistence)
  * distributed tensor map/reduce with MAD Byzantine filtering
  * hardware-weight governance

Runs over any BaseTransport (UDP in deployment, in-memory in tests).
==============================================================================
"""

import time
import uuid
import asyncio
import hashlib
import base64
import binascii
import json
import random
import hmac
from typing import Dict, List, Optional, Callable

import numpy as np

from juvian_crypto import (
    PiMandelbrotKeyEngine, ThreeWayVerification, SessionKeyChain,
    SessionBootstrap, JuvianCipher, DEFAULT_ITER,
)
from juvian_ecdh import GroupKeyAgreement, PairwiseECDH
from juvian_identity import Identity, verify_envelope, verify_voucher, make_voucher
from juvian_dht import KademliaRoutingTable, node_id_from_seed
from juvian_tensor import TensorMapWorker, TensorReducer
from juvian_history import JuvianHistoryLedger
from juvian_fractal_storage import FractalPersistenceManager


# message types
MSG_BEACON         = "BEACON"
MSG_BEACON_ACK     = "BEACON_ACK"
MSG_KEX_START      = "KEX_START"        # initiator -> all: begin group key agreement
MSG_KEX_R1         = "KEX_R1"           # member -> all: Burmester-Desmedt z_i
MSG_KEX_R2         = "KEX_R2"           # member -> all: Burmester-Desmedt X_i
MSG_KEX_CONFIRM    = "KEX_CONFIRM"      # member -> all: proof it derived the group key
MSG_KEX_ECHO       = "KEX_ECHO"         # member -> all: signed view of the round-1 z's it received (equivocation cross-check)
MSG_KEX_ECHO2      = "KEX_ECHO2"        # member -> all: signed view of the round-2 X's it received (round-2 equivocation cross-check)
MSG_KEX_EVICT      = "KEX_EVICT"        # member -> all: non-repudiable proof an insider equivocated -> exclude + re-key
MSG_REQUEST_SUBMIT = "REQUEST_SUBMIT"   # non-sequencer -> sequencer: order this request
MSG_REQUEST_RESULT = "REQUEST_RESULT"   # sequencer -> origin: outcome of a forwarded request
MSG_VERIFY_OPEN    = "VERIFY_OPEN"      # initiator -> verifiers: derive & commit
MSG_VERIFY_COMMIT  = "VERIFY_COMMIT"    # verifier  -> initiator: fingerprint
MSG_VERIFY_RESULT  = "VERIFY_RESULT"    # initiator -> verifiers: verified/rejected
MSG_TENSOR_TASK    = "TENSOR_TASK"      # anchor -> worker: compute a slice
MSG_TENSOR_YIELD   = "TENSOR_YIELD"     # worker -> anchor: result
MSG_NODE_HANDOVER  = "NODE_HANDOVER"    # cooperative thermal/battery load-shed signal
MSG_FIND_NODE      = "FIND_NODE"        # iterative Kademlia lookup query
MSG_FIND_NODE_REPLY = "FIND_NODE_REPLY" # ... and its reply (k closest contacts)
MSG_CHAIN_REQUEST  = "CHAIN_REQUEST"    # lagging member -> sequencer: send the chain entries I missed
MSG_CHAIN_BATCH    = "CHAIN_BATCH"      # sequencer -> lagging member: ordered missed rounds (anti-entropy repair)

# bounds for network-supplied arrays (audit 2.2, 3.1)
MAX_TENSOR_ELEMENTS = 4_000_000         # cap decoded array size
TENSOR_RATE_WINDOW  = 10.0              # seconds
TENSOR_RATE_MAX     = 20                # max tensor tasks accepted per peer / window
MAX_RETAINED_KEX    = 64               # cap KEX sessions kept in memory (audit 3.2)
MAX_SEEN_MESSAGES   = 8192             # bounded (from,nonce) cache for replay defence
VERIFIER_FANOUT     = 5                # how many verifiers to invite per request
REQUIRED_QUORUM     = 3                # valid proofs needed to confirm a key
MAX_VERIFY_ATTEMPTS = 3                # retries with a disjoint verifier set (audit 4.3)
POW_DIFFICULTY      = 16               # leading-zero bits required of a peer's birth cert (audit 2.3/4.4)
VOUCH_THRESHOLD     = 2               # admission-policy default: distinct admitted-member vouchers a newcomer needs when vouching is enabled (founders seed the root of trust); the allowlist path ignores this
VOUCHER_CACHE_MAX   = 512             # most PENDING (not-yet-admitted) voucher subjects kept; oldest evicted first, so a Sybil beacon flood cannot grow memory without bound (an evicted genuine subject re-presents on its next beacon)
MIN_TENSOR_CONTRIBUTORS = 4            # incl. local; floor so zone-local sharding doesn't starve the MAD filter
MAX_PENDING_OPENS   = 64              # how far ahead a buffered VERIFY_OPEN may sit while the chain catches up
GOSSIP_FANOUT       = 6               # peers each node forwards a verify-broadcast to (epidemic dissemination)
GOSSIP_MIN_ROSTER   = 32              # only switch to gossip above this roster size; at/below it keep guaranteed reliable broadcast
CHAIN_ARCHIVE_MAX   = 64              # how many recent verified rounds EACH member keeps for anti-entropy repair (so any member can serve, surviving a sequencer handover)
CHAIN_BATCH_MAX     = 32              # most rounds returned in one catch-up batch
CATCHUP_COOLDOWN_S  = 0.5             # min spacing between a lagging node's catch-up requests (also retries a lost batch)
CATCHUP_FANOUT      = 4               # distinct archive-holders a laggard asks per attempt (failover: not only the sequencer)
SEQUENCER_TERM      = 32              # chain slots per sequencer epoch: leadership is stable within an epoch (so request forwarding is unchanged) and rotates verifiably across epochs
BEACON_INTERVAL_S   = 5.0             # periodic discovery/keep-alive beacon; also the membership-maintenance tick (read at call time so tests can compress it)
PEER_EXPIRY_S       = 30.0            # a peer silent for this long (no beacon, no authenticated traffic) is dropped from the membership view; roles recompute over the survivors, which is what un-sticks a dead sequencer
DEPTH_GUARD_S       = 10.0            # refuse a SECOND, different round opened at the same chain depth within this window (narrows the dual-candidate race during expiry skew)
# Per-request broadcasts that are disseminated by gossip (relayed on first
# receipt) rather than a flat O(N) fan-out from the sequencer. KEX messages are
# deliberately NOT gossiped: that is a one-time setup/re-key path, and gossip's
# redundant delivery would change the equivocation adversary model the
# authenticated-KEX confirmation round is analysed against.
GOSSIP_TYPES        = {MSG_VERIFY_OPEN, MSG_VERIFY_RESULT}
KAD_ALPHA           = 3               # iterative-lookup parallelism (Kademlia alpha)
KAD_K               = 8               # contacts returned per FIND_NODE reply
KAD_LOOKUP_ROUNDS   = 12              # safety cap on lookup rounds (>= log2 of any realistic zone)


class HardwareTelemetryMonitor:
    """Cooperative thermal / battery load-shedding for edge hosts.

    NOT a security control. Temperature and battery -- like hw_weight itself --
    are self-reported and unverifiable, so this only helps an *honest* node shed
    work gracefully before it thermally throttles or browns out. A malicious node
    gains nothing here it couldn't get by setting hw_weight directly, and the
    handover signal it emits can only down-weight *itself* in peers' tables
    (the signal is authenticated, so `from` cannot be forged). Off by default;
    opt in with enable_telemetry=True.

    Readings come from `node.simulated_temp` / `node.simulated_battery` when those
    are set (the default, so tests and demos are deterministic and never touch
    real hardware); set them to None to read this host's real sysfs sensors."""

    def __init__(self, node, max_temp_c: float = 85.0,
                 min_battery_pct: float = 15.0):
        self.node = node
        self.max_temp = max_temp_c
        self.min_battery = min_battery_pct
        self._task = None
        self.is_bouncing = False

    def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self):
        while True:
            await self.tick()
            await asyncio.sleep(2.0)

    async def tick(self):
        """One evaluation step. Public so tests can drive it deterministically."""
        try:
            temp = self._read_cpu_temp()
            batt = self._read_battery_level()
            if ((temp >= self.max_temp or batt <= self.min_battery)
                    and not self.is_bouncing):
                self.is_bouncing = True
                await self._bounce(temp, batt)
            elif (self.is_bouncing and temp < self.max_temp - 10.0
                  and batt > self.min_battery + 5.0):
                self.is_bouncing = False
                self.node.hw_weight = self.node.nominal_hw_weight
                self.node.stats["last_report"] = (
                    "telemetry recovered: nominal weight restored")
        except Exception:
            return  # a telemetry hiccup must never crash the node

    def _read_cpu_temp(self) -> float:
        sim = getattr(self.node, "simulated_temp", None)
        if sim is not None:
            return sim
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                return float(f.read().strip()) / 1000.0
        except (FileNotFoundError, ValueError, PermissionError, OSError):
            return 45.0

    def _read_battery_level(self) -> float:
        sim = getattr(self.node, "simulated_battery", None)
        if sim is not None:
            return sim
        try:
            with open("/sys/class/power_supply/BAT0/capacity", "r") as f:
                return float(f.read().strip())
        except (FileNotFoundError, ValueError, PermissionError, OSError):
            return 100.0

    async def _bounce(self, temp: float, batt: float):
        """Shed load: zero our own compute weight (the tensor governor then drops
        incoming tasks) and advise peers to prefer a healthy successor."""
        n = self.node
        n.hw_weight = 0.0
        peers = list(n.routing.all_peers().values())
        healthy = [p for p in peers
                   if p.get("geo_zone") == n.geo_zone
                   and p.get("device_type") == "ANCHOR"
                   and p.get("weight", 0.0) > 0.5]
        if not healthy:
            healthy = n.routing.closest(n.node_id, count=5)
        successor = (max(healthy, key=lambda p: p.get("weight", 1.0))["id"]
                     if healthy else "")
        n.stats["last_report"] = (
            f"thermal bounce (temp={temp}C batt={batt}%) -> {successor[:8]}")
        await n._broadcast({
            "type": MSG_NODE_HANDOVER,
            "from": n.node_id,
            "from_addr": n.address,
            "designated_successor": successor,
            "geo_zone": n.geo_zone,
        })


# ---- authenticated-KEX key-confirmation primitives ---------------------------
# A member proves it derived the group key by publishing a MAC under that key
# (not the key itself -- HMAC is one-way, so the salt never crosses the wire).
# The tag is bound to (kex_id, roster, member) so it cannot be replayed into a
# different KEX or a different roster, and a member who derived a DIFFERENT key
# (e.g. an equivocation victim) produces a tag that nobody else's key matches.
def _roster_digest(roster) -> str:
    """Order-independent fingerprint of the participant set (roster is already
    sorted at the call sites; we hash the canonical join)."""
    return hashlib.sha256(("|".join(roster)).encode()).hexdigest()[:32]


def _kex_conf_tag(salt: bytes, kex_id: str, roster_hash: str,
                  member_id: str) -> str:
    """HMAC proving knowledge of the derived genesis `salt`, bound to this KEX,
    this roster, and the confirming member. Verified by recomputing it with the
    verifier's OWN salt: a match means both sides derived the identical key."""
    conf_key = hashlib.sha256(b"JUVIAN_KEXCONF_KEY::" + salt).digest()
    ctx = ("JUVIAN_KEXCONF::" + kex_id + "::" + roster_hash
           + "::" + member_id).encode()
    return hmac.new(conf_key, ctx, hashlib.sha256).hexdigest()


class JuvianNode:
    def __init__(self, name: str, address: str, transport,
                 session_secret: str = "juvian-default-session",
                 device_type: str = "ANCHOR",
                 hw_weight: float = 5.0,
                 geo_zone: str = "ZONE_0",
                 mandelbrot_iter: int = DEFAULT_ITER,
                 history_path: Optional[str] = None,
                 chain_path: Optional[str] = None,
                 identity: Optional["Identity"] = None,
                 identity_path: Optional[str] = None,
                 pow_difficulty: int = POW_DIFFICULTY,
                 allowlist: Optional[set] = None,
                 founders: Optional[set] = None,
                 vouch_threshold: int = VOUCH_THRESHOLD,
                 enable_telemetry: bool = False):
        self.name = name
        self.address = address
        self.transport = transport
        self.device_type = device_type
        self.hw_weight = hw_weight
        self.nominal_hw_weight = hw_weight     # restored after a thermal bounce
        # Geo-zone is a LOCALITY hint only (option A): it steers tensor compute
        # toward same-region peers. It does NOT shard consensus -- request
        # ordering still flows through the single global sequencer, so the key
        # chain stays fork-free (audit 4.1).
        self.geo_zone = geo_zone
        self.mandelbrot_iter = mandelbrot_iter
        # Sybil pricing: this node mints its own birth certificate at this
        # difficulty, and requires the same of every peer it admits (audit
        # 2.3/4.4). 0 disables. A supplied `identity` is used as-is.
        self.pow_difficulty = pow_difficulty
        # --- admission policy (Sybil BOUNDING; OFF by default) ---
        # Layered ON TOP of PoW, never replacing it. With both off (the default)
        # behaviour is exactly as before, so existing deployments/tests are
        # unchanged. Two independent, combinable controls:
        #   * allowlist  -- if non-empty, ONLY ids in it are ever admitted
        #                   (closed fleet; airtight). `None`/empty disables.
        #   * vouching   -- `founders` seed the root of trust (admitted
        #                   unconditionally, like seeds). Any already-admitted
        #                   member may sign a voucher; a newcomer is admitted once
        #                   it shows >= `vouch_threshold` vouchers from DISTINCT
        #                   admitted members. Enabled iff `founders` is non-empty.
        # (The admitted-set / voucher caches are initialised below, once node_id
        # is known, since this node always trusts itself.)
        self.allowlist = set(allowlist) if allowlist else set()
        self.founders = set(founders) if founders else set()
        self.vouch_threshold = max(1, vouch_threshold)
        # verifier quorum config (overridable per-node, e.g. in tests).
        # A request needs `verifier_quorum` valid proofs out of up to
        # `verifier_fanout` invited verifiers, retried on a disjoint set if the
        # quorum can't be met (audit 4.3).
        self.verifier_fanout = VERIFIER_FANOUT
        self.verifier_quorum = REQUIRED_QUORUM

        # self-certifying identity: node_id is bound to an Ed25519 public key,
        # so the `from` on a signed message cannot be spoofed (audit 2.1), and
        # carries a proof-of-work birth certificate to price Sybils (audit 2.3).
        if identity is None:
            identity = (Identity.load_or_create(identity_path, pow_difficulty)
                        if identity_path else Identity.generate(pow_difficulty))
        self.identity = identity
        self.node_id = identity.node_id
        # admission caches (node_id is known now; a node always trusts itself,
        # and founders seed the root of trust for vouching).
        self._admitted: set = {self.node_id} | self.founders
        self._vouchers: Dict[str, dict] = {}   # subject -> {issuer -> voucher}
        self._my_vouchers: List[dict] = []      # vouchers issued FOR this node
        self.routing = KademliaRoutingTable(self.node_id)

        # shared genesis salt for the session (all participants must match)
        self.genesis_salt = SessionBootstrap.pre_shared(session_secret)
        self.genesis_source = "PSK"   # becomes "GROUP_DH" after key agreement
        self.chain = SessionKeyChain(self.node_id, self.genesis_salt)
        self.verifier = ThreeWayVerification(self.node_id)
        self.cipher = JuvianCipher(self.chain)

        self.history_path = history_path or f"history_{name}.npy"
        self.chain_path = chain_path or f"chain_{name}.frac"

        # in-flight collectors
        self._pending_tensor: Dict[str, dict] = {}   # job_id -> collector
        self._verify_waits: Dict[str, asyncio.Future] = {}
        self._find_waits: Dict[str, asyncio.Future] = {}   # qid -> reply future
        # forwarded-request futures: request_id -> future (non-sequencer side)
        self._request_waits: Dict[str, asyncio.Future] = {}
        # request_id -> ids this node actually forwarded the request to; a
        # REQUEST_RESULT is honored only from one of them (failover-correct: the
        # answering node may have become sequencer after we sent).
        self._request_targets: Dict[str, set] = {}
        # (depth, session_id, opener_id, ts) of the round currently open at our
        # head slot; a SECOND round for the same depth from a DIFFERENT opener
        # within DEPTH_GUARD_S is refused, narrowing the dual-candidate window
        # during expiry skew (same-opener retries pass freely).
        self._depth_open = None
        self._request_lock: Optional[asyncio.Lock] = None
        # VERIFY_RESULT broadcasts that arrive before this node has opened its
        # own round for that session. Buffered here and applied the instant the
        # round opens, so non-verifier members never miss a chain advance.
        self._pending_results: Dict[str, dict] = {}
        # VERIFY_OPENs for a chain slot we haven't caught up to yet (the prior
        # slot's VERIFY_RESULT is still in flight). Buffered by chain_index and
        # processed once our chain reaches that depth, so pipelined requests over
        # an async/lossy network stay in lockstep instead of deriving on a stale
        # salt and spuriously failing the round.
        self._pending_opens: Dict[int, List[dict]] = {}
        # --- anti-entropy repair (closes the gossip coverage gap) ---
        # EVERY member stashes per-session payload/iter at VERIFY_OPEN time and
        # keeps a bounded ring of committed rounds keyed by chain_index, so ANY
        # member -- not just the sequencer -- can serve the slots a laggard missed
        # (this is what lets repair survive a sequencer handover / failover).
        self._round_meta: Dict[str, dict] = {}            # session_id -> {payload_b64, iter}
        self._round_archive: Dict[int, dict] = {}         # chain_index -> {payload_b64, iter, verifiers, fingerprint}
        # Signed head attestations harvested from peers' beacons: peer_id ->
        # {depth, digest, ts}. A catch-up batch from any server is adopted only up
        # to the longest prefix whose re-derived head digest is anchored here by
        # the current sequencer OR by a quorum of distinct peers -- so a malicious
        # server cannot fork a laggard onto a forged chain.
        self._head_attest: Dict[str, dict] = {}
        # Member-side: rate-limit catch-up requests (also retries a lost batch).
        self._last_catchup_req: float = 0.0
        # group-key-agreement (KEX) sessions: kex_id -> state dict
        self._kex: Dict[str, dict] = {}
        # node ids PROVEN to have equivocated in a KEX (two validly-signed,
        # conflicting round-1 values). Persists across re-keys so an evicted
        # culprit cannot be re-admitted to a fresh attempt (Byzantine-robust BD).
        self._kex_excluded: set = set()
        # the future of an in-flight establish_session() call, resolved when THIS
        # node installs a genesis salt via ANY kex_id -- so a re-key triggered by
        # an eviction still completes the original caller's request.
        self._kex_caller_future: Optional[asyncio.Future] = None
        # bounded replay-defence cache of (from, nonce) we have already accepted
        self._seen_msgs: set = set()
        self._seen_order: List[tuple] = []
        self._kex_order: List[str] = []       # insertion order, for eviction
        self._installed_kex_id: Optional[str] = None
        # rate limiting for unsolicited compute (audit 2.2): per-peer timestamps
        self._tensor_hits: Dict[str, List[float]] = {}

        # stats for dashboards
        self.stats = {
            "keys_verified": 0,
            "keys_rejected": 0,
            "tensor_jobs": 0,
            "byzantine_purged": 0,
            "last_report": "standing by",
            "last_energy": 0.0,
            "peak_z": 0.0,
        }
        self._running = False

        # cooperative thermal/battery load-shedding (off unless opted in).
        # simulated_* default to safe nominal values so demos/tests never read
        # real hardware; set to None to use this host's sysfs sensors.
        self.enable_telemetry = enable_telemetry
        self.simulated_temp = 45.0
        self.simulated_battery = 100.0
        self.telemetry = HardwareTelemetryMonitor(self)

    # ------------------------------------------------------------------ boot --
    async def start(self):
        self.transport.set_handler(self._on_message)
        await self.transport.start()
        self._running = True
        JuvianHistoryLedger.initialize(self.history_path)
        if self.enable_telemetry:
            self.telemetry.start()
        asyncio.create_task(self._beacon_loop())

    async def stop(self):
        self._running = False
        # Go DARK: a stopped node must neither beacon (the loop exits on the
        # flag) nor keep receiving/answering -- otherwise it lingers as a
        # half-alive ghost that still sequences forwarded requests but never
        # expires from anyone's view.
        self.transport.set_handler(None)
        await self.telemetry.stop()
        await self.transport.stop()
        self.persist_chain()

    # ------------------------------------------------- authenticated I/O ------
    async def _send(self, addr: str, body: dict):
        """Sign and send a message body to a single peer."""
        await self.transport.send(addr, self.identity.wrap(body))

    async def _broadcast(self, body: dict):
        """Sign and broadcast a message body to all peers (BEST-EFFORT: a single
        datagram per peer, no retransmission). Used for periodic/idempotent
        traffic like discovery beacons, where loss is self-healing."""
        await self.transport.broadcast(self.identity.wrap(body))

    async def _reliable_broadcast(self, body: dict):
        """RELIABLE broadcast: sign once, then deliver to every known peer over
        the reliable unicast path (per-peer selective-repeat ARQ) instead of one
        best-effort datagram. A dropped packet otherwise desyncs a member -- a
        missed KEX round makes it compute the wrong group key (permanent split).
        Each peer receives one copy (deduped by the reliable layer on
        retransmit). Scope is the *known* roster: a peer we do not yet know is not
        reached -- the same limit any broadcast has.

        Used for the one-time KEX setup/re-key path. The per-request verify
        broadcasts use `_gossip_broadcast` instead, so the sequencer's fan-out
        does not grow O(N) on the hot path."""
        envelope = self.identity.wrap(body)
        addrs = [p["address"] for p in self.routing.all_peers().values()
                 if p.get("address")]
        for addr in addrs:
            await self.transport.send(addr, envelope)

    def _mark_seen(self, from_id, nonce):
        """Record an (origin, nonce) in the seen-cache so a later copy is treated
        as a duplicate. Used so the *origin* of a gossip never re-processes or
        re-forwards its own message if a forward loops back to it (the relay path
        in `_on_message` already records every message it forwards)."""
        if nonce is None:
            return
        key = (from_id, nonce)
        if key in self._seen_msgs:
            return
        self._seen_msgs.add(key)
        self._seen_order.append(key)
        while len(self._seen_order) > MAX_SEEN_MESSAGES:
            old = self._seen_order.pop(0)
            self._seen_msgs.discard(old)

    def _gossip_peers(self, exclude_addrs: set) -> List[str]:
        """Pick up to GOSSIP_FANOUT peer addresses to forward a broadcast to,
        excluding addresses we should not echo back to (the peer we received it
        from, and the origin). A random sample gives standard epidemic coverage;
        with <= GOSSIP_FANOUT eligible peers it returns all of them (so at small
        scale gossip degenerates to sending to everyone)."""
        addrs = [p["address"] for p in self.routing.all_peers().values()
                 if p.get("address") and p["address"] not in exclude_addrs]
        if len(addrs) <= GOSSIP_FANOUT:
            return addrs
        return random.sample(addrs, GOSSIP_FANOUT)

    async def _gossip(self, envelope: dict, exclude_addrs: set):
        """Forward an ALREADY-SIGNED envelope to a bounded fan-out of peers over
        reliable unicast. The origin's signature and nonce are preserved, so any
        node can authenticate the original author and dedup network-wide on
        (origin, nonce) -- which also stops forwarding loops. This replaces the
        O(N) origin fan-out of a flat broadcast with O(GOSSIP_FANOUT) per node;
        total messages are still O(N) (you cannot inform N nodes with fewer), but
        no single node is the bottleneck."""
        for addr in self._gossip_peers(exclude_addrs):
            await self.transport.send(addr, envelope)

    async def _gossip_broadcast(self, body: dict):
        """Disseminate a message to the whole mesh by gossip: sign once, push to
        GOSSIP_FANOUT peers, and let each peer re-forward on first receipt (see
        the relay hook in `_on_message`). The message floods the overlay in
        ~O(log N) hops with bounded per-node fan-out. Used for the per-request
        verify broadcasts every member must see to stay in lockstep, once the
        roster is large enough that a flat fan-out is the real cost.

        Reliability note: each hop is reliable unicast (ARQ), so per-hop loss is
        recovered; coverage is the probabilistic part (does the overlay reach
        every node), which holds with high probability but is NOT a hard
        guarantee from push-gossip alone. A partition/coverage backstop
        (anti-entropy: a lagging node pulls the chain entry it missed) is the
        companion that restores a hard guarantee at scale -- noted as future
        work. Below GOSSIP_MIN_ROSTER, `_disseminate` keeps the guaranteed
        reliable broadcast instead, so lockstep is hard-guaranteed at zone
        scale."""
        marked = dict(body)
        marked["_g"] = 1                           # tells relays to re-forward
        envelope = self.identity.wrap(marked)
        # record our own (id, nonce) so a forward looping back is deduped and we
        # never re-process or re-forward our own broadcast
        self._mark_seen(self.node_id, envelope["body"].get("_n"))
        await self._gossip(envelope, exclude_addrs={self.address})

    async def _disseminate(self, body: dict):
        """Choose the dissemination strategy for a per-request broadcast. At zone
        scale (roster <= GOSSIP_MIN_ROSTER) use the GUARANTEED reliable broadcast
        (every known peer, ARQ) so every member adopts in hard lockstep -- this
        is the unchanged, fully-tested behaviour. Above that, switch to
        bounded-fan-out GOSSIP so the sequencer's send does not grow O(N) on the
        hot path, trading the hard delivery guarantee for high-probability
        coverage (see _gossip_broadcast; anti-entropy repair is the companion
        that would restore the guarantee at large scale)."""
        if (body.get("type") in GOSSIP_TYPES
                and self.routing.count() > GOSSIP_MIN_ROSTER):
            await self._gossip_broadcast(body)
        else:
            await self._reliable_broadcast(body)

    # -------------------------------------------------------------- discovery --
    async def _beacon_loop(self):
        """Periodic keep-alive + membership maintenance. Every tick: (1) prune
        peers silent beyond PEER_EXPIRY_S, so the membership view -- and every
        role computed from it, the rotating sequencer above all -- reflects who
        is actually alive; (2) beacon, so live-but-quiet nodes never expire
        (broadcast reaches the whole zone). Globals are read each iteration so
        tests can compress the timeline."""
        while self._running:
            expired = self.routing.prune(PEER_EXPIRY_S)
            if expired:
                self.stats["peers_expired"] = (
                    self.stats.get("peers_expired", 0) + len(expired))
                # an expired peer's head attestation must not keep counting
                # toward the catch-up anchoring quorum (and must not accumulate
                # unboundedly across churn): anchoring reflects LIVE membership.
                for pid in expired:
                    self._head_attest.pop(pid, None)
            await self.announce()
            await asyncio.sleep(BEACON_INTERVAL_S)

    async def announce(self):
        """Broadcast a signed discovery beacon. When this node was admitted by
        vouching, the beacon carries the vouchers issued for it so a peer running
        an admission policy can fold them and admit us on first contact."""
        beacon = {
            "type": MSG_BEACON,
            "from": self.node_id,
            "from_addr": self.address,
            "name": self.name,
            "device_type": self.device_type,
            "weight": self.hw_weight,
            "geo_zone": self.geo_zone,
            "chain_depth": self.chain.depth(),
            "chain_digest": self.chain.head_digest(),
        }
        if self._my_vouchers:
            beacon["vouchers"] = self._my_vouchers
        await self._broadcast(beacon)

    # --------------------------------------------------------------- dispatch --
    async def _on_message(self, envelope: dict, sender_addr: str):
        """Verify the signed envelope, then dispatch. Any malformed or
        unauthenticated message is dropped here, never reaching a handler
        (audit 2.1, 3.1). Replays are dropped via a bounded seen-nonce cache."""
        body = verify_envelope(envelope, pow_difficulty=self.pow_difficulty)
        if body is None:
            return  # unsigned / forged / id-spoofed / no-PoW / stale -> drop
        if self._is_replay(body):
            return  # duplicate (from, nonce) within the freshness window -> drop
        # Liveness: any authenticated message from a known peer counts as
        # evidence it is alive -- refresh its last_seen so active protocol
        # traffic (not only beacons) keeps it from expiring out of the view.
        sender_id = body.get("from")
        if isinstance(sender_id, str) and sender_id:
            self.routing.touch(sender_id)
        # Byzantine-robust BD: keep every signed round-1 envelope we see. An
        # equivocator's second, conflicting value arrives (at us, or harvested
        # from a peer's echo) as another validly-signed envelope for the same
        # (kex_id, owner); two distinct z for one owner is a non-repudiable proof.
        if body.get("type") == MSG_KEX_R1:
            self._retain_kex_r1(body, envelope)
        # Same for round-2: keep every signed X envelope; two distinct, validly-
        # signed X for one owner is a round-2 equivocation proof (cross-checked
        # via the round-2 echo, exactly as round 1).
        elif body.get("type") == MSG_KEX_R2:
            self._retain_kex_r2(body, envelope)
        # Gossip relay: on FIRST receipt of a message the origin marked for
        # gossip (_g), re-forward the original signed envelope to our own
        # fan-out, so it floods the overlay with bounded per-node cost. We
        # exclude the peer we got it from and the origin to cut backflow;
        # duplicates from other paths are dropped by the replay check above, so
        # forwarding loops terminate. Messages sent by reliable broadcast (zone
        # scale) carry no _g and are not relayed -- the origin reached everyone.
        # Under an admission policy, only ADMITTED senders' traffic is relayed:
        # otherwise a non-admitted Sybil (kept out of every view) could still use
        # the honest mesh as a free N x fanout bandwidth amplifier for arbitrary
        # signed messages. Admission gates amplification, not just visibility.
        if body.get("_g") and (not self._admission_active()
                               or self._is_admitted(sender_id)):
            asyncio.ensure_future(self._gossip(
                envelope, exclude_addrs={sender_addr, body.get("from_addr")}))
        try:
            await self._dispatch(body, sender_addr)
        except Exception:
            # a malformed-but-signed message must not crash the node (audit 3.1)
            return

    def _is_replay(self, body: dict) -> bool:
        """True if this exact (sender, nonce) was already seen. Bounded FIFO
        cache; combined with the freshness window in verify_envelope this gives
        practical replay protection without unbounded state."""
        key = (body.get("from"), body.get("_n"))
        if key[1] is None:
            return False
        if key in self._seen_msgs:
            return True
        self._seen_msgs.add(key)
        self._seen_order.append(key)
        while len(self._seen_order) > MAX_SEEN_MESSAGES:
            old = self._seen_order.pop(0)
            self._seen_msgs.discard(old)
        return False

    async def _dispatch(self, msg: dict, sender_addr: str):
        t = msg.get("type")
        if t == MSG_BEACON:
            await self._handle_beacon(msg)
        elif t == MSG_BEACON_ACK:
            self._note_peer(msg)
        elif t == MSG_NODE_HANDOVER:
            await self._handle_handover(msg)
        elif t == MSG_FIND_NODE:
            await self._handle_find_node(msg)
        elif t == MSG_FIND_NODE_REPLY:
            self._handle_find_node_reply(msg)
        elif t == MSG_KEX_START:
            await self._handle_kex_start(msg)
        elif t == MSG_KEX_R1:
            await self._handle_kex_r1(msg)
        elif t == MSG_KEX_R2:
            await self._handle_kex_r2(msg)
        elif t == MSG_KEX_CONFIRM:
            await self._handle_kex_confirm(msg)
        elif t == MSG_KEX_ECHO:
            await self._handle_kex_echo(msg)
        elif t == MSG_KEX_ECHO2:
            await self._handle_kex_echo2(msg)
        elif t == MSG_KEX_EVICT:
            await self._handle_kex_evict(msg)
        elif t == MSG_REQUEST_SUBMIT:
            await self._handle_request_submit(msg)
        elif t == MSG_REQUEST_RESULT:
            await self._handle_request_result(msg)
        elif t == MSG_VERIFY_OPEN:
            await self._handle_verify_open(msg)
        elif t == MSG_VERIFY_COMMIT:
            await self._handle_verify_commit(msg)
        elif t == MSG_VERIFY_RESULT:
            await self._handle_verify_result(msg)
        elif t == MSG_TENSOR_TASK:
            await self._handle_tensor_task(msg)
        elif t == MSG_TENSOR_YIELD:
            await self._handle_tensor_yield(msg)
        elif t == MSG_CHAIN_REQUEST:
            await self._handle_chain_request(msg)
        elif t == MSG_CHAIN_BATCH:
            await self._handle_chain_batch(msg)

    def _admission_active(self) -> bool:
        """True iff any admission policy is in force (allowlist or vouching).
        With neither configured, admission is open (PoW remains the only gate)
        and behaviour is exactly as before."""
        return bool(self.allowlist) or bool(self.founders)

    def _is_admitted(self, pid: str) -> bool:
        """Policy decision for a peer id, evaluated on top of (never instead of)
        the PoW + signature checks already done in verify_envelope.

        * No policy configured -> everyone who cleared PoW is admitted.
        * Allowlist configured -> the id must be on it (airtight, closed fleet).
          The allowlist is AUTHORITATIVE: neither vouchers nor founder status can
          widen it (self is the one exception, so a misconfigured node fails to
          admit peers rather than silently self-rejecting -- loud, not weird).
        * Else vouching -> self and founders are always admitted; any other id is
          admitted once it has accumulated >= vouch_threshold vouchers from
          DISTINCT already-admitted issuers."""
        if not self._admission_active():
            return True
        if pid == self.node_id:
            return True
        if self.allowlist:
            return pid in self.allowlist
        if pid in self.founders or pid in self._admitted:
            return True
        # vouching: count distinct vouchers whose issuer is itself admitted
        vmap = self._vouchers.get(pid, {})
        good = sum(1 for iss in vmap if iss in self._admitted)
        if good >= self.vouch_threshold:
            self._admitted.add(pid)
            self._vouchers.pop(pid, None)  # decision cached; map no longer needed
            self._reeval_vouchees()        # this admission may unblock others
            return True
        return False

    def _reeval_vouchees(self):
        """A newly-admitted issuer can push other subjects over the threshold.
        Re-scan pending vouchees and promote any that now qualify (bounded
        fixpoint: each pass admits >=1 or stops, and the id set is finite)."""
        changed = True
        while changed:
            changed = False
            for sub, vmap in list(self._vouchers.items()):
                if sub in self._admitted:
                    self._vouchers.pop(sub, None)
                    continue
                if sum(1 for iss in vmap if iss in self._admitted) >= self.vouch_threshold:
                    self._admitted.add(sub)
                    self._vouchers.pop(sub, None)
                    changed = True

    def _fold_vouchers(self, vouchers, subject_id: str):
        """Record valid vouchers carried by a peer's beacon FOR that peer. Each
        voucher is self-verified (signature + issuer-id binding + subject match);
        only well-formed ones for THIS subject are kept. Storing them lets the
        subject become admissible the moment enough of its issuers are admitted,
        even if they are admitted later. The pending-subject cache is BOUNDED
        (oldest non-admitted subject evicted past VOUCHER_CACHE_MAX): a Sybil
        swarm beaconing self-signed vouchers must not grow this dict without
        limit -- an evicted genuine pending subject merely re-presents its
        vouchers on its next beacon, so eviction costs liveness-of-memory only,
        never admission correctness."""
        if not isinstance(vouchers, list):
            return
        vmap = self._vouchers.setdefault(subject_id, {})
        for v in vouchers[:64]:                       # bound work per message
            iss = verify_voucher(v, subject_id)
            if iss and iss != subject_id:             # no self-vouching
                vmap[iss] = v
        while len(self._vouchers) > VOUCHER_CACHE_MAX:
            oldest = next(iter(self._vouchers))
            if oldest == subject_id:                  # never evict the live one
                break
            self._vouchers.pop(oldest, None)

    def my_voucher_for(self, subject_id: str) -> dict:
        """Issue a signed voucher from THIS node for `subject_id` (helper for
        operators / onboarding flows; a node vouches only for peers it chooses
        to)."""
        return make_voucher(self.identity, subject_id)

    def present_voucher(self, voucher: dict) -> bool:
        """Onboarding: accept a voucher issued FOR this node (by a sponsor) so we
        will advertise it on our beacons. Kept only if it validates for our id."""
        if verify_voucher(voucher, self.node_id):
            # de-dup by issuer
            self._my_vouchers = [v for v in self._my_vouchers
                                 if v.get("iss") != voucher.get("iss")]
            self._my_vouchers.append(voucher)
            return True
        return False

    def _note_peer(self, msg: dict):
        pid = msg.get("from")
        if not pid or pid == self.node_id:
            return
        # Admission policy (Sybil bounding), evaluated on top of the PoW +
        # signature checks already passed in verify_envelope. A beacon may carry
        # the peer's own vouchers; fold them first so a newcomer presenting
        # enough admitted-issuer vouchers is admitted on the same message.
        if self._admission_active():
            self._fold_vouchers(msg.get("vouchers"), pid)
            if not self._is_admitted(pid):
                self.stats["admission_denied"] = (
                    self.stats.get("admission_denied", 0) + 1)
                return                  # not admitted -> never enters the view
        self.routing.update(
            pid, msg.get("from_addr", ""),
            nat_type=msg.get("nat_type", "UNKNOWN"),
            weight=msg.get("weight", 1.0),
            device_type=msg.get("device_type", "UNKNOWN"),
            geo_zone=msg.get("geo_zone", "UNKNOWN"),
        )
        # Proactive anti-entropy: a beacon that advertises a chain deeper than
        # ours means we missed one or more rounds entirely (we may never have
        # received their VERIFY_* gossip). Pull the gap. This catches a node
        # isolated from the verify floods but still exchanging beacons.
        peer_depth = msg.get("chain_depth", -1)
        peer_digest = msg.get("chain_digest")
        # Record the peer's signed head attestation (the beacon is an
        # authenticated envelope, so this (depth,digest) is non-repudiably
        # theirs). Used to anchor catch-up adoption to a quorum / the sequencer.
        if (isinstance(peer_depth, int) and peer_depth >= 0
                and isinstance(peer_digest, str) and peer_digest):
            self._head_attest[pid] = {"depth": peer_depth,
                                      "digest": peer_digest, "ts": time.time()}
        if isinstance(peer_depth, int) and peer_depth > self.chain.depth():
            asyncio.ensure_future(self._maybe_request_catchup(peer_depth))

    async def _handle_beacon(self, msg: dict):
        self._note_peer(msg)
        # reply so the beacon sender learns about us too
        if msg.get("from_addr"):
            await self._send(msg["from_addr"], {
                "type": MSG_BEACON_ACK,
                "from": self.node_id,
                "from_addr": self.address,
                "name": self.name,
                "device_type": self.device_type,
                "weight": self.hw_weight,
                "geo_zone": self.geo_zone,
                "chain_depth": self.chain.depth(),
                "chain_digest": self.chain.head_digest(),
            })

    async def _handle_handover(self, msg: dict):
        """A peer announced it is thermally/battery bouncing. Down-weight it in
        our table so we stop sending it compute. The signal is authenticated, so
        a node can only down-weight ITSELF this way (it cannot touch others)."""
        pid = msg.get("from")
        if pid and pid != self.node_id and pid in self.routing.peer_ids():
            self.routing.set_weight(pid, 0.0)
            succ = (msg.get("designated_successor") or "")[:8]
            self.stats["last_report"] = f"peer {pid[:8]} bounced -> prefer {succ}"

    # ----------------------------------------- iterative Kademlia discovery ---
    # Replaces the O(N^2) beacon flood: a node joins by *looking up* its own id
    # (and a few random ids to fill distant buckets) through the DHT, learning
    # O(k.log N) contacts in O(log N) hops instead of being flooded by everyone.
    # FIND_NODE also carries presence metadata, so it subsumes the beacon's role.
    #
    # Security: routing is populated ONLY from verified replies (every inbound
    # message passes signature + PoW + id-match in verify_envelope before
    # _note_peer runs). The (id,address) contacts inside a reply are treated as
    # mere hints for WHOM to query next -- a node's id enters our table only once
    # that node has proven it with a signed message, so a peer cannot inject a
    # claimed id it does not hold the key for.
    def _self_descriptor(self) -> dict:
        d = {"from": self.node_id, "from_addr": self.address,
             "name": self.name, "device_type": self.device_type,
             "weight": self.hw_weight, "geo_zone": self.geo_zone}
        if self._my_vouchers:
            d["vouchers"] = self._my_vouchers
        return d

    async def _handle_find_node(self, msg: dict):
        """Reply to a lookup with the k closest contacts we know to the target,
        and learn the (verified) querier -- FIND_NODE doubles as presence."""
        self._note_peer(msg)
        target = msg.get("target", self.node_id)
        contacts = [[p["id"], p["address"]]
                    for p in self.routing.closest(target, KAD_K)
                    if p["id"] != msg.get("from") and p["address"]]
        reply = self._self_descriptor()
        reply.update({"type": MSG_FIND_NODE_REPLY, "qid": msg.get("qid"),
                      "contacts": contacts})
        if msg.get("from_addr"):
            await self._send(msg["from_addr"], reply)

    def _handle_find_node_reply(self, msg: dict):
        """A verified reply: learn the responder (real id + metadata) and hand
        the contact hints to the waiting lookup."""
        self._note_peer(msg)
        fut = self._find_waits.get(msg.get("qid"))
        if fut is not None and not fut.done():
            fut.set_result(msg.get("contacts", []))

    async def _find_node(self, addr: str, target: str,
                         timeout: float = 2.0) -> List[list]:
        """Send one FIND_NODE(target) to an address and await its contact list
        ([] on timeout). The responder is learned via the verified reply path."""
        qid = uuid.uuid4().hex[:12]
        fut = asyncio.get_running_loop().create_future()
        self._find_waits[qid] = fut
        q = self._self_descriptor()
        q.update({"type": MSG_FIND_NODE, "qid": qid, "target": target})
        try:
            await self._send(addr, q)
            return await asyncio.wait_for(fut, timeout)
        except Exception:
            return []
        finally:
            self._find_waits.pop(qid, None)

    async def _lookup(self, target: str, timeout: float = 2.0) -> int:
        """Iterative Kademlia node lookup for `target`: repeatedly query the
        ALPHA closest unqueried addresses and merge the contacts they return,
        until a round discovers nothing new (or a round cap). Every verified
        node reached lands in routing. Returns the number of addresses queried
        (a proxy for lookup cost; expected ~O(log N))."""
        tgt = int(target[:40], 16)
        order = lambda idhex: tgt ^ int(idhex[:40], 16)
        shortlist: Dict[str, str] = {p["address"]: p["id"]
                                     for p in self.routing.closest(target, KAD_K)
                                     if p["address"]}
        queried: set = set()
        for _ in range(KAD_LOOKUP_ROUNDS):
            cand = sorted((a for a in shortlist if a not in queried),
                          key=lambda a: order(shortlist[a]))[:KAD_ALPHA]
            if not cand:
                break
            queried.update(cand)
            replies = await asyncio.gather(
                *(self._find_node(a, target, timeout) for a in cand))
            grew = False
            for contacts in replies:
                for entry in contacts:
                    try:
                        cid, caddr = entry[0], entry[1]
                    except (TypeError, IndexError, KeyError):
                        continue
                    if caddr and caddr != self.address and caddr not in shortlist:
                        shortlist[caddr] = cid
                        grew = True
            if not grew:
                break
        return len(queried)

    async def bootstrap(self, seeds, refresh: int = 6,
                        timeout: float = 2.0) -> int:
        """Join the mesh by iterative lookup instead of an O(N^2) flood: ask the
        seed(s) for nodes near us, run a self-lookup to fill nearby buckets, then
        a few random-target lookups to populate distant buckets (Kademlia bucket
        refresh). Returns the total addresses queried."""
        if isinstance(seeds, str):
            seeds = [seeds]
        total = 0
        for s in seeds:                                   # learn from the seed(s)
            await self._find_node(s, self.node_id, timeout)
            total += 1
        total += await self._lookup(self.node_id, timeout)   # nearby buckets
        for _ in range(max(0, refresh)):                     # distant buckets
            total += await self._lookup("%040x" % random.getrandbits(160), timeout)
        return total

    # ================================================================ KEX ======
    # Group key agreement (Burmester-Desmedt) to establish a *confidential*
    # genesis salt. Two broadcast rounds; no secret ever leaves a node. Once the
    # group key is computed it is installed as the chain's genesis salt, so every
    # subsequent pi-Mandelbrot key depends on a secret an eavesdropper lacks.

    def _kex_state(self, kex_id: str) -> dict:
        """Fetch or create the per-KEX state, bounding how many we retain so a
        flood of KEX_START ids cannot grow memory without limit (audit 3.2)."""
        st = self._kex.get(kex_id)
        if st is None:
            st = {"bd": None, "pending_r1": {}, "pending_r2": {},
                  "future": None, "r2_sent": False, "installed": False,
                  "confs": {}, "confirmed": False,
                  "roster": None, "roster_hash": None,
                  # --- Byzantine-robust BD: round-1 echo / equivocation state ---
                  "echo_sent": False,          # have we broadcast our R1 view yet
                  "echoed_by": set(),          # member ids whose echo we have folded in (incl. self)
                  # owner_id -> { z_hex -> signed R1 envelope } : >1 z_hex for an
                  # owner is a non-repudiable equivocation proof (two signed values)
                  "r1_seen": {},
                  "pending_echo": [],          # echoes that arrived before our bd existed
                  # --- round-2 echo: the exact analog over the X values ---
                  "echo2_sent": False,         # have we broadcast our R2 (X) view yet
                  "echoed2_by": set(),         # member ids whose round-2 echo we folded (incl. self)
                  "r2_seen": {},               # owner_id -> { x_hex -> signed R2 envelope }
                  "pending_echo2": [],         # round-2 echoes that arrived before our bd existed
                  "aborted": False,            # set when this KEX is abandoned for a re-key
                  "rekey_launched": False}     # guard: at most one re-key per aborted KEX
            self._kex[kex_id] = st
            self._kex_order.append(kex_id)
            while len(self._kex_order) > MAX_RETAINED_KEX:
                old = self._kex_order.pop(0)
                if old != self._installed_kex_id:
                    self._kex.pop(old, None)
        return st

    def _lowest_member_id(self) -> str:
        """The lexicographically smallest node id among (self + known peers).
        Used as the deterministic KEX initiator, so every honest node agrees on
        who re-keys the group without any election traffic. (The request
        SEQUENCER is no longer this fixed id -- see `_sequencer_for` -- so a node
        cannot grind one low id into permanent ordering power.)"""
        return min([self.node_id] + self.routing.peer_ids())

    def _members(self) -> List[str]:
        """Sorted ids of (self + known peers) -- the membership as this node sees
        it. The same partial-view caveat as the old lowest-id rule applies: a
        node whose routing table is missing peers sees a smaller set."""
        return sorted(set([self.node_id] + self.routing.peer_ids()))

    def _epoch_seed(self, depth: int) -> bytes:
        """Public, agreed per-epoch randomness: the verifiable chain head digest at
        the epoch boundary (depth // TERM * TERM). Every member caught up to the
        boundary agrees on it, and it is unknowable before that epoch's rounds are
        committed. Far from the boundary (digest trimmed, or a laggard that hasn't
        reached it) it falls back to the current head digest -- such a node may
        misjudge the live sequencer, which is the documented partial-view/lag
        residual (liveness only; repair is quorum-anchored and does not depend on
        naming the sequencer correctly)."""
        boundary = (depth // SEQUENCER_TERM) * SEQUENCER_TERM
        seed_hex = self.chain.digest_at(boundary) or self.chain.head_digest()
        return bytes.fromhex(seed_hex)

    def _sequencer_for(self, depth: int) -> str:
        """The verifiably-chosen request sequencer for the epoch containing
        `depth`: the member minimising H(epoch_seed ‖ epoch ‖ id) over ALL members
        in view. Properties, honestly stated:

        * AGREEMENT -- a deterministic function of the member set and the epoch
          seed, so no election traffic; views that contain the epoch's winner all
          agree. Sensitivity to an incomplete view is the same single-critical-
          element profile as the old lowest-id rule (you must know the one winner),
          except the critical element rotates per epoch instead of being one fixed
          node forever -- so a stale view causes occasional, self-healing
          disagreement rather than a permanent one.
        * DEGRINDING -- the hash makes the id's VALUE irrelevant: every member is
          equally likely to win each epoch, so offline-grinding a low (or any
          special) id no longer buys leadership. Influence is proportional to the
          number of PoW-priced identities, exactly the Sybil cost the system
          already assumes.
        * ROTATION -- the winner changes with the epoch index and the boundary
          digest, so ordering/censorship power is time-sliced rather than
          permanently capturable.

        Honest residuals: (a) epoch 0's seed is the public genesis constant, so
        the very first term is still id-grindable -- bounded to one term; (b) an
        INCUMBENT sequencer can bias the next boundary digest by choosing/placing
        the payload that lands on the boundary slot (it orders requests), i.e.
        seed-grinding by the current leader remains possible -- but it is active,
        per-epoch, on-chain-visible work rather than a one-off offline grind, and
        only an existing leader can do it. Closing that fully needs commit-reveal
        or threshold randomness (future work). (c) Partial-view minority
        disagreement persists exactly as before -- a minority-view node degrades
        in liveness, and a minority self-believed 'sequencer' cannot normally
        complete rounds because the verification quorum applies the majority
        view; adoption stays single-winner-per-slot, fingerprint-checked, and
        repair quorum-anchored.

        The sequencer still cannot forge keys (every member adopts its OWN
        re-derived, fingerprint-checked key) -- selection confers ordering power
        only."""
        members = self._members()
        if not members:
            return self.node_id
        seed = self._epoch_seed(depth)
        epoch = (depth // SEQUENCER_TERM).to_bytes(8, "big")
        return min(members, key=lambda pid: hashlib.sha256(
            seed + epoch + pid.encode()).digest())

    def _current_sequencer(self) -> str:
        """The sequencer for this node's current chain depth/epoch."""
        return self._sequencer_for(self.chain.depth())

    def is_sequencer(self) -> bool:
        """True iff this node is the verifiably-chosen request sequencer for the
        current epoch. All requests are ordered through this single node so the key
        chain is a totally-ordered log and two concurrent initiators cannot fork
        it (audit 4.1). The role is deterministic and verifiable (see
        `_sequencer_for`) and ROTATES per epoch, so it confers ordering power only
        and cannot be permanently captured by grinding a low node id -- the
        sequencer still cannot forge keys, since every member adopts its OWN
        independently-derived key, fingerprint-checked."""
        return self.node_id == self._current_sequencer()

    def should_initiate_kex(self) -> bool:
        """True iff this node is the canonical KEX initiator for the current
        membership -- the lexicographically smallest node id among (self + known
        peers). A single deterministic initiator prevents two concurrent KEXs
        from installing different genesis salts and forking the group (audit
        1.4)."""
        return self.node_id == self._lowest_member_id()

    async def establish_session(self, timeout: float = 8.0,
                                force: bool = False) -> dict:
        """Drive a group key agreement across all currently-discovered peers
        and install the resulting shared secret as this session's genesis salt.

        Any node may call this (it is masterless overall -- the caller is just
        the per-KEX initiator). All members derive the identical genesis salt.
        Must be run before the first request, while the chain is still empty."""
        if self.chain.depth() > 0:
            return {"status": "ERROR",
                    "reason": "cannot re-key: chain already has entries"}

        roster = sorted(set(self.routing.peer_ids() + [self.node_id])
                        - self._kex_excluded)
        if len(roster) < 2:
            return {"status": "ERROR",
                    "reason": "need >=2 members for group key agreement"}

        # Only the canonical initiator should start a KEX, so two nodes cannot
        # concurrently install different genesis salts (audit 1.4). Override with
        # force=True if you are intentionally coordinating a single initiator.
        if not force and not self.should_initiate_kex():
            return {"status": "DEFERRED",
                    "reason": "not the canonical initiator (lowest node id); "
                              "another node should lead this KEX, or pass force=True"}

        kex_id = uuid.uuid4().hex[:12]
        fut = asyncio.get_running_loop().create_future()
        self._kex_caller_future = fut   # resolved on install, surviving re-keys

        # announce the roster so every member uses the identical ring ordering
        await self._reliable_broadcast({
            "type": MSG_KEX_START,
            "from": self.node_id,
            "from_addr": self.address,
            "kex_id": kex_id,
            "roster": roster,
        })
        # begin locally too (we don't receive our own broadcast)
        await self._kex_begin(kex_id, roster, fut)

        try:
            salt_fp = await asyncio.wait_for(fut, timeout)
            return {"status": "ESTABLISHED", "kex_id": kex_id,
                    "members": len(roster), "genesis_fingerprint": salt_fp,
                    "source": "GROUP_DH"}
        except asyncio.TimeoutError:
            return {"status": "TIMEOUT", "kex_id": kex_id,
                    "reason": "group key agreement did not complete in time"}

    async def _kex_begin(self, kex_id: str, roster: List[str],
                         fut: Optional[asyncio.Future] = None):
        """Create our Burmester-Desmedt state for a KEX and broadcast round 1.
        Applies any round-1/round-2/echo values that raced ahead of the start.
        A proven equivocator is filtered out of the ring here, so a crafted
        KEX_START cannot re-admit it (Byzantine-robust BD)."""
        roster = sorted(set(roster) - self._kex_excluded)
        if self.node_id not in roster or len(roster) < 2:
            return
        st = self._kex_state(kex_id)
        if st.get("bd") is not None:
            return  # already begun
        if fut is not None:
            st["future"] = fut

        st["bd"] = GroupKeyAgreement(self.node_id, roster)
        st["roster"] = list(roster)
        st["roster_hash"] = _roster_digest(roster)

        # broadcast our round-1 value z_i
        await self._reliable_broadcast({
            "type": MSG_KEX_R1,
            "from": self.node_id,
            "from_addr": self.address,
            "kex_id": kex_id,
            "z": st["bd"].round1_public(),
        })

        # apply anything buffered before we had a bd object (never from an
        # excluded culprit)
        for mid, zhex in st["pending_r1"].items():
            if mid not in self._kex_excluded:
                st["bd"].set_round1(mid, zhex)
        st["pending_r1"].clear()
        for mid, xhex in st["pending_r2"].items():
            if mid not in self._kex_excluded:
                st["bd"].set_round2(mid, xhex)
        st["pending_r2"].clear()
        for echo in st["pending_echo"]:
            self._kex_fold_echo(kex_id, echo)
        st["pending_echo"].clear()
        for echo in st["pending_echo2"]:
            self._kex_fold_echo2(kex_id, echo)
        st["pending_echo2"].clear()
        await self._kex_advance(kex_id)

    async def _handle_kex_start(self, msg: dict):
        await self._kex_begin(msg["kex_id"], msg["roster"], fut=None)

    async def _handle_kex_r1(self, msg: dict):
        kex_id, mid, zhex = msg["kex_id"], msg["from"], msg["z"]
        if mid in self._kex_excluded:
            return  # a proven equivocator's contributions are not folded in
        st = self._kex.get(kex_id)
        if st is None or st.get("bd") is None:
            # round-1 value arrived before our KEX_START -- buffer it
            st = self._kex_state(kex_id)
            st["pending_r1"][mid] = zhex
            return
        st["bd"].set_round1(mid, zhex)
        await self._kex_advance(kex_id)

    async def _handle_kex_r2(self, msg: dict):
        kex_id, mid, xhex = msg["kex_id"], msg["from"], msg["x"]
        if mid in self._kex_excluded:
            return
        st = self._kex.get(kex_id)
        if st is None or st.get("bd") is None:
            st = self._kex_state(kex_id)
            st["pending_r2"][mid] = xhex
            return
        st["bd"].set_round2(mid, xhex)
        await self._kex_advance(kex_id)

    async def _kex_advance(self, kex_id: str):
        """Move a KEX forward through the Byzantine-robust schedule:
        round-1  ->  ECHO our signed round-1 view  ->  (cross-check)  ->
        round-2  ->  key-confirmation  ->  install. The echo step turns an
        insider equivocation into a non-repudiable proof BEFORE any key is
        computed, so the culprit is evicted and the honest members re-key
        (see _handle_kex_echo / _kex_handle_equivocation) rather than the whole
        group aborting."""
        st = self._kex.get(kex_id)
        if not st or st.get("bd") is None or st.get("aborted"):
            return
        bd = st["bd"]

        # Step 1: once we hold every round-1 z, broadcast the SIGNED round-1
        # envelopes we received so every member can cross-check them. We withhold
        # round 2 until the echoes are in and the view is consistent.
        if bd.ready_for_round2() and not st["echo_sent"]:
            st["echo_sent"] = True
            st["echoed_by"].add(self.node_id)
            envs = [env for zmap in st["r1_seen"].values()
                    for env in zmap.values()]
            await self._reliable_broadcast({
                "type": MSG_KEX_ECHO,
                "from": self.node_id,
                "from_addr": self.address,
                "kex_id": kex_id,
                "envs": envs,
            })

        # An equivocation may already be provable from what we hold. If so,
        # eviction + re-key takes over and this KEX is abandoned.
        if self._kex_scan_equivocation(kex_id):
            return

        # Step 2: send round 2 only once the round-1 view is confirmed consistent
        # across the whole (non-excluded) roster -- so an equivocation is caught
        # before it can perturb anyone's key.
        if (bd.ready_for_round2() and st["echo_sent"]
                and not st["r2_sent"] and self._kex_echo_consistent(kex_id)):
            st["r2_sent"] = True
            await self._reliable_broadcast({
                "type": MSG_KEX_R2,
                "from": self.node_id,
                "from_addr": self.address,
                "kex_id": kex_id,
                "x": bd.round2_public(),
            })
            # a buffered X may now be applicable
            for mid, xhex in st["pending_r2"].items():
                if mid not in self._kex_excluded:
                    bd.set_round2(mid, xhex)
            st["pending_r2"].clear()

        # Step 3: once we hold every round-2 X, ECHO the signed X envelopes we
        # received and withhold confirmation until that view is consistent --
        # the round-2 analog of the round-1 echo. This turns an X-equivocation
        # into the same non-repudiable proof + eviction + re-key, instead of the
        # confirmation round merely failing to reach quorum (a safe abort).
        if bd.ready_for_key() and not st["echo2_sent"]:
            st["echo2_sent"] = True
            st["echoed2_by"].add(self.node_id)
            envs = [env for vmap in st["r2_seen"].values()
                    for env in vmap.values()]
            await self._reliable_broadcast({
                "type": MSG_KEX_ECHO2,
                "from": self.node_id,
                "from_addr": self.address,
                "kex_id": kex_id,
                "envs": envs,
            })

        # a round-1 OR round-2 equivocation may now be provable (the scan checks
        # both rounds); if so, eviction + re-key takes over and this KEX aborts.
        if self._kex_scan_equivocation(kex_id):
            return

        # Step 4: key-confirmation + install, only once the round-2 view is
        # confirmed consistent across the (non-excluded) roster -- so an
        # X-equivocation is caught and evicted before anyone confirms a key.
        if (bd.ready_for_key() and st["echo2_sent"] and not st["confirmed"]
                and self._kex_echo2_consistent(kex_id)):
            # Authenticated KEX -- key-confirmation round. We publish a proof we
            # derived the salt and install only once a MAJORITY publishes the
            # SAME proof (see _kex_try_install). The round-1 and round-2 echoes
            # above turn an equivocation in either round into an eviction; this
            # confirmation round remains the backstop for an active in-transit
            # MITM that perturbs values -- the disagreeing parties never reach
            # quorum, so no node installs a salt the majority did not confirm.
            st["confirmed"] = True
            salt = bd.group_salt()
            my_conf = _kex_conf_tag(salt, kex_id, st["roster_hash"], self.node_id)
            st["confs"][self.node_id] = my_conf
            await self._reliable_broadcast({
                "type": MSG_KEX_CONFIRM,
                "from": self.node_id,
                "from_addr": self.address,
                "kex_id": kex_id,
                "conf": my_conf,
            })
            await self._kex_try_install(kex_id)

    async def _handle_kex_confirm(self, msg: dict):
        """Record a member's key-confirmation proof and (re)check whether a
        confirming majority has now been reached. Confirmations may arrive
        before we have finished deriving our own key; we store them and
        _kex_try_install simply no-ops until our Burmester-Desmedt state is
        ready."""
        kex_id = msg.get("kex_id")
        mid = msg.get("from")
        conf = msg.get("conf")
        if kex_id is None or mid is None or conf is None:
            return
        if mid in self._kex_excluded:
            return
        st = self._kex.get(kex_id)
        if st is None:
            st = self._kex_state(kex_id)
        st["confs"][mid] = conf
        await self._kex_try_install(kex_id)

    async def _kex_try_install(self, kex_id: str):
        """Install the group genesis salt iff a MAJORITY of the roster has
        confirmed the *same* key we derived. Confirmations that do not match our
        key (a different roster, or a value an equivocator fed only to us) are
        counted as zero -- exactly like a wrong proof in the verify quorum -- so
        an equivocation victim falls short of quorum and refuses to install,
        while the honest majority converges on one salt. A bare majority also
        guarantees uniqueness: two disjoint majorities cannot exist, so at most
        one salt can ever be installed network-wide."""
        st = self._kex.get(kex_id)
        if not st or st.get("bd") is None or st["installed"]:
            return
        bd = st["bd"]
        if not bd.ready_for_key():
            return  # cannot confirm a key we have not derived yet
        roster = st["roster"] or []
        roster_hash = st["roster_hash"]
        salt = bd.group_salt()

        matches = 0
        for mid, tag in st["confs"].items():
            if mid not in roster:
                continue
            expected = _kex_conf_tag(salt, kex_id, roster_hash, mid)
            if hmac.compare_digest(expected, tag):
                matches += 1
        quorum = len(roster) // 2 + 1
        if matches < quorum:
            return  # not enough agreement (yet, or ever if equivocated)

        # First-wins guard (defense-in-depth for 1.4 alongside the canonical
        # initiator); install_genesis_salt also refuses a non-empty chain.
        if self._installed_kex_id is not None:
            st["installed"] = True
            return
        if self.chain.install_genesis_salt(salt):
            self.genesis_salt = salt
            self.genesis_source = "GROUP_DH"
            self._installed_kex_id = kex_id
            st["installed"] = True
            fp = hashlib.sha256(salt).hexdigest()[:16]
            self.stats["last_report"] = (
                f"group genesis salt installed+confirmed by "
                f"{matches}/{len(roster)} ({fp})")
            fut = st.get("future")
            if fut and not fut.done():
                fut.set_result(fp)
            if (self._kex_caller_future is not None
                    and not self._kex_caller_future.done()):
                self._kex_caller_future.set_result(fp)

    # ---- Byzantine-robust BD: round-1 echo, equivocation proof, eviction -----
    # The canonical Burmester-Desmedt attack is round-1 EQUIVOCATION: a member
    # signs a different z_i to different members. Because each member folds its
    # neighbours' z into its own X_i, the keys diverge and the confirmation round
    # (above) refuses to install -- safe, but a persistent equivocator can stall
    # every retry forever (a liveness DoS). These methods turn that into a
    # non-repudiable proof (two validly-signed, conflicting z for one owner),
    # evict the culprit, and re-key the honest remainder. Framing an honest node
    # is impossible: a second signature under its key cannot be forged.

    def _retain_kex_r1(self, body: dict, envelope: dict):
        """Keep a signed round-1 envelope (the evidence), indexed by
        kex_id -> owner -> z_hex -> envelope. A second, conflicting z from the
        same owner lands beside the first and the pair is a complete proof."""
        kex_id = body.get("kex_id")
        owner = body.get("from")
        z = body.get("z")
        if not (isinstance(kex_id, str) and isinstance(owner, str)
                and isinstance(z, str)):
            return
        if owner in self._kex_excluded:
            return
        st = self._kex_state(kex_id)
        st["r1_seen"].setdefault(owner, {})[z] = envelope

    def _verify_r1_evidence(self, env: dict, kex_id: str):
        """Verify one round-1 envelope offered as evidence; return (owner, z_hex)
        iff it is a validly-signed KEX_R1 for THIS kex_id. Freshness is
        intentionally not enforced (evidence is older than a live message by the
        time it is echoed), but the signature, the id<->key binding and the
        proof-of-work are checked exactly as on the live path."""
        body = verify_envelope(env, max_age_s=float("inf"),
                               pow_difficulty=self.pow_difficulty)
        if body is None:
            return None
        if body.get("type") != MSG_KEX_R1 or body.get("kex_id") != kex_id:
            return None
        owner = body.get("from")
        z = body.get("z")
        if not (isinstance(owner, str) and isinstance(z, str)):
            return None
        return owner, z

    def _kex_fold_echo(self, kex_id: str, echo: dict):
        """Fold a peer's echoed round-1 view into our evidence map. Each carried
        envelope is independently verified -- we trust the owner's own signature,
        never the echoer's say-so -- so harvesting a conflicting value from an
        echo is exactly as sound as having received it directly."""
        st = self._kex.get(kex_id)
        if st is None:
            return
        for env in (echo.get("envs") or []):
            ev = self._verify_r1_evidence(env, kex_id)
            if ev is None:
                continue
            owner, z = ev
            if owner in self._kex_excluded:
                continue
            st["r1_seen"].setdefault(owner, {})[z] = env
        echoer = echo.get("from")
        if isinstance(echoer, str):
            st["echoed_by"].add(echoer)

    def _kex_scan_equivocation(self, kex_id: str) -> bool:
        """If any owner now shows two distinct, validly-signed values for the
        SAME round (round-1 z OR round-2 X), that is a non-repudiable
        equivocation proof: schedule eviction + re-key and return True so the
        caller stops advancing this (doomed) KEX."""
        st = self._kex.get(kex_id)
        if st is None:
            return False
        if st.get("aborted"):
            return True
        for seen in (st["r1_seen"], st["r2_seen"]):
            for owner, vmap in seen.items():
                if owner in self._kex_excluded:
                    return True
                if len(vmap) >= 2:
                    envs = list(vmap.values())
                    asyncio.ensure_future(self._kex_handle_equivocation(
                        kex_id, owner, envs[0], envs[1]))
                    return True
        return False

    def _kex_echo_consistent(self, kex_id: str) -> bool:
        """True iff we have folded an echo from every (non-excluded) roster
        member AND no owner shows a conflicting round-1 value -- the precondition
        for safely computing and broadcasting round 2."""
        st = self._kex.get(kex_id)
        if st is None:
            return False
        active = {m for m in (st.get("roster") or [])
                  if m not in self._kex_excluded}
        if not active <= set(st["echoed_by"]):
            return False
        return all(len(zmap) <= 1 for zmap in st["r1_seen"].values())

    async def _handle_kex_echo(self, msg: dict):
        """A member echoes the signed round-1 envelopes it received. We verify
        each, harvest any conflict (an equivocation proof), and -- once the view
        is complete and consistent -- proceed to round 2."""
        kex_id = msg.get("kex_id")
        mid = msg.get("from")
        if not isinstance(kex_id, str) or mid in self._kex_excluded:
            return
        st = self._kex.get(kex_id)
        if st is None or st.get("bd") is None:
            # echo raced ahead of our KEX_START -- buffer; folded in at _kex_begin
            st = self._kex_state(kex_id)
            st["pending_echo"].append(msg)
            return
        self._kex_fold_echo(kex_id, msg)
        await self._kex_advance(kex_id)

    # ---- round-2 echo: the exact analog of the round-1 path, over the X values.
    # X_i = (z_{i+1}/z_{i-1})^{r_i} is a deterministic function of the (now
    # pinned + echo-consistent) round-1 z's and the owner's fixed secret, so an
    # honest member emits exactly ONE X_i. Two distinct, validly-signed X for one
    # owner+kex_id is therefore a non-repudiable round-2 equivocation proof,
    # cross-checked the same way: each member echoes the signed X envelopes it
    # received, a victim harvests the conflicting one, and the culprit is evicted.
    def _retain_kex_r2(self, body: dict, envelope: dict):
        """Round-2 analog of _retain_kex_r1: index a signed X envelope by
        kex_id -> owner -> x_hex -> envelope. A second, conflicting X lands
        beside the first and the pair is a complete proof."""
        kex_id = body.get("kex_id")
        owner = body.get("from")
        x = body.get("x")
        if not (isinstance(kex_id, str) and isinstance(owner, str)
                and isinstance(x, str)):
            return
        if owner in self._kex_excluded:
            return
        st = self._kex_state(kex_id)
        st["r2_seen"].setdefault(owner, {})[x] = envelope

    def _verify_r2_evidence(self, env: dict, kex_id: str):
        """Round-2 analog of _verify_r1_evidence: return (owner, x_hex) iff env
        is a validly-signed KEX_R2 for THIS kex_id (sig/id-binding/PoW checked,
        freshness intentionally disabled)."""
        body = verify_envelope(env, max_age_s=float("inf"),
                               pow_difficulty=self.pow_difficulty)
        if body is None:
            return None
        if body.get("type") != MSG_KEX_R2 or body.get("kex_id") != kex_id:
            return None
        owner = body.get("from")
        x = body.get("x")
        if not (isinstance(owner, str) and isinstance(x, str)):
            return None
        return owner, x

    def _kex_fold_echo2(self, kex_id: str, echo: dict):
        """Fold a peer's echoed round-2 view into our X evidence map. Each
        carried envelope is independently verified against the owner's own key,
        never the echoer's say-so."""
        st = self._kex.get(kex_id)
        if st is None:
            return
        for env in (echo.get("envs") or []):
            ev = self._verify_r2_evidence(env, kex_id)
            if ev is None:
                continue
            owner, x = ev
            if owner in self._kex_excluded:
                continue
            st["r2_seen"].setdefault(owner, {})[x] = env
        echoer = echo.get("from")
        if isinstance(echoer, str):
            st["echoed2_by"].add(echoer)

    def _kex_echo2_consistent(self, kex_id: str) -> bool:
        """True iff we have folded a round-2 echo from every (non-excluded)
        roster member AND no owner shows a conflicting X -- the precondition for
        safely confirming and installing the group key."""
        st = self._kex.get(kex_id)
        if st is None:
            return False
        active = {m for m in (st.get("roster") or [])
                  if m not in self._kex_excluded}
        if not active <= set(st["echoed2_by"]):
            return False
        return all(len(vmap) <= 1 for vmap in st["r2_seen"].values())

    async def _handle_kex_echo2(self, msg: dict):
        """A member echoes the signed round-2 X envelopes it received. We verify
        each, harvest any conflict (a round-2 equivocation proof), and -- once
        the view is complete and consistent -- proceed to confirmation."""
        kex_id = msg.get("kex_id")
        mid = msg.get("from")
        if not isinstance(kex_id, str) or mid in self._kex_excluded:
            return
        st = self._kex.get(kex_id)
        if st is None or st.get("bd") is None:
            # echo raced ahead of our KEX_START -- buffer; folded in at _kex_begin
            st = self._kex_state(kex_id)
            st["pending_echo2"].append(msg)
            return
        self._kex_fold_echo2(kex_id, msg)
        await self._kex_advance(kex_id)

    def _verify_kex_value_envelope(self, env: dict, kex_id: str):
        """Verify a signed KEX_R1 (z) or KEX_R2 (X) envelope offered as
        evidence; return (msg_type, owner, value_hex) iff it is validly signed
        for THIS kex_id. Freshness is intentionally not enforced (evidence is
        older than a live message by the time it is echoed), but the signature,
        the id<->key binding and the proof-of-work are checked exactly as on the
        live path. Returning the message TYPE lets the proof check require both
        halves to be the SAME round."""
        body = verify_envelope(env, max_age_s=float("inf"),
                               pow_difficulty=self.pow_difficulty)
        if body is None:
            return None
        mt = body.get("type")
        field = {MSG_KEX_R1: "z", MSG_KEX_R2: "x"}.get(mt)
        if field is None or body.get("kex_id") != kex_id:
            return None
        owner = body.get("from")
        val = body.get(field)
        if not (isinstance(owner, str) and isinstance(val, str)):
            return None
        return mt, owner, val

    def _verify_equivocation_proof(self, culprit: str, env_a: dict,
                                   env_b: dict, kex_id: str) -> bool:
        """A sound proof against `culprit` is two key-agreement envelopes that
        both verify under the culprit's key, are both for this kex_id, are both
        attributed to the culprit, are for the SAME round, and carry DIFFERENT
        values. The unforgeable second signature is what makes framing an honest
        member impossible; the same-round check is what stops an honest member's
        own (legitimately different) round-1 z and round-2 X from being paired
        into a bogus 'proof'."""
        a = self._verify_kex_value_envelope(env_a, kex_id)
        b = self._verify_kex_value_envelope(env_b, kex_id)
        if a is None or b is None:
            return False
        mt_a, owner_a, val_a = a
        mt_b, owner_b, val_b = b
        return (mt_a == mt_b                       # same round (not a z paired with an X)
                and owner_a == culprit and owner_b == culprit
                and val_a != val_b)

    async def _kex_handle_equivocation(self, kex_id: str, culprit: str,
                                       env_a: dict, env_b: dict):
        """Exclude a proven equivocator and drive a re-key on the honest
        remainder. Idempotent and bounded: each call permanently removes one
        distinct id, so re-keys cannot loop."""
        if culprit in self._kex_excluded:
            return
        if not self._verify_equivocation_proof(culprit, env_a, env_b, kex_id):
            return  # never act on an unproven accusation
        self._kex_excluded.add(culprit)
        self.stats["last_report"] = (
            f"KEX equivocation proven: evicting {culprit[:12]}")
        self.stats["kex_evictions"] = self.stats.get("kex_evictions", 0) + 1
        # broadcast the self-contained proof so every member verifies it itself
        await self._reliable_broadcast({
            "type": MSG_KEX_EVICT,
            "from": self.node_id,
            "from_addr": self.address,
            "kex_id": kex_id,
            "culprit": culprit,
            "proof": [env_a, env_b],
        })
        await self._kex_rekey_after_eviction(kex_id)

    async def _handle_kex_evict(self, msg: dict):
        """Receive an eviction proof, RE-VERIFY it independently (trust no
        accuser), and on success exclude the culprit and join the re-key. A
        forged or framing proof verifies to False and is ignored."""
        kex_id = msg.get("kex_id")
        culprit = msg.get("culprit")
        proof = msg.get("proof") or []
        if not (isinstance(kex_id, str) and isinstance(culprit, str)
                and isinstance(proof, list) and len(proof) == 2):
            return
        if not self._verify_equivocation_proof(culprit, proof[0], proof[1], kex_id):
            return
        if culprit in self._kex_excluded:
            return
        self._kex_excluded.add(culprit)
        self.stats["kex_evictions"] = self.stats.get("kex_evictions", 0) + 1
        await self._kex_rekey_after_eviction(kex_id)

    async def _kex_rekey_after_eviction(self, kex_id: str):
        """Abandon the poisoned KEX and re-key on the honest remainder. The
        original caller's future is carried (node-level) so it still resolves.
        Only the lowest-id survivor broadcasts the new start, so re-keys never
        race (the audit 1.4 single-initiator principle)."""
        st = self._kex.get(kex_id)
        if st is None or st.get("rekey_launched"):
            return  # a concurrent detector (self-scan + inbound EVICT) already did this
        st["aborted"] = True
        st["rekey_launched"] = True
        reduced = sorted(set(st.get("roster") or []) - self._kex_excluded)
        if len(reduced) < 2:
            fut = st.get("future") or self._kex_caller_future
            if fut is not None and not fut.done():
                fut.set_exception(RuntimeError(
                    "KEX failed: too many equivocators to form a group"))
            return
        # the lowest-id surviving member leads; everyone else awaits its KEX_START
        if self.node_id != min(reduced):
            return
        new_kex_id = uuid.uuid4().hex[:12]
        await self._reliable_broadcast({
            "type": MSG_KEX_START,
            "from": self.node_id,
            "from_addr": self.address,
            "kex_id": new_kex_id,
            "roster": reduced,
        })
        await self._kex_begin(new_kex_id, reduced, st.get("future"))

    # ------------------------------------------------ 3-of-3 verification ------
    async def submit_request(self, payload: bytes,
                             timeout: float = 5.0) -> dict:
        """Submit a compute-request. To keep the key chain a single totally-
        ordered log, every request is ordered through the deterministic, verifiably
        -chosen sequencer for the current epoch (see `_sequencer_for`). If this node
        is the sequencer it drives the round directly; otherwise it forwards the
        request to the sequencer and awaits the outcome. This makes the chain
        fork-free even when several nodes submit at the same moment (audit 4.1)."""
        if self.is_sequencer():
            return await self._sequence_request(payload, timeout)
        return await self._forward_to_sequencer(payload, timeout)

    async def _forward_to_sequencer(self, payload: bytes,
                                    timeout: float) -> dict:
        """Non-sequencer path: hand the request to the sequencer and wait for it
        to report the outcome. The wait is split into attempts, and the target
        is RECOMPUTED each attempt: if the sequencer died, peer expiry removes
        it from the membership view mid-wait and the selection argmin lands on
        the next-ranked live member -- so the ladder deterministically follows
        the failover instead of re-dialing a ghost. If expiry promotes THIS node
        mid-ladder, it sequences directly. A request submitted inside the expiry
        window can still time out (the dead winner is not yet pruned anywhere);
        ordering resumes within ~PEER_EXPIRY_S + one beacon tick -- a bounded
        stall instead of the previous permanent one."""
        request_id = uuid.uuid4().hex[:12]
        fut = asyncio.get_running_loop().create_future()
        self._request_waits[request_id] = fut
        self._request_targets[request_id] = set()
        attempts = 3
        per_wait = max(1.0, (timeout + 1.0) / attempts)
        try:
            for _ in range(attempts):
                if self.is_sequencer():
                    # expiry promoted us mid-ladder: order it ourselves
                    return await self._sequence_request(payload, timeout)
                seq_id = self._current_sequencer()
                seq = self.routing.all_peers().get(seq_id)
                if seq and seq.get("address"):
                    self._request_targets[request_id].add(seq_id)
                    await self._send(seq["address"], {
                        "type": MSG_REQUEST_SUBMIT,
                        "from": self.node_id,
                        "from_addr": self.address,
                        "request_id": request_id,
                        "payload_b64": base64.b64encode(payload).decode(),
                    })
                try:
                    return await asyncio.wait_for(
                        asyncio.shield(fut), per_wait)
                except asyncio.TimeoutError:
                    continue   # recompute the sequencer and try again
            return {"status": "TIMEOUT", "request_id": request_id,
                    "reason": "sequencer did not return a result in time "
                              "(if it died, ordering resumes after peer expiry)"}
        finally:
            self._request_waits.pop(request_id, None)
            self._request_targets.pop(request_id, None)
            if not fut.done():
                fut.cancel()

    async def _handle_request_submit(self, msg: dict):
        """Sequencer path: a peer forwarded a request to be ordered. Drive it
        through the same serialized pipeline as our own requests, then report
        the outcome back to the origin. Under an admission policy the origin must
        itself be admitted -- otherwise a Sybil that is kept out of honest views
        could still get honest verifiers to derive/confirm a key by forwarding a
        request to its (honest) sequencer. Admission gates PARTICIPATION, not
        just routing visibility."""
        if not self.is_sequencer():
            return  # not our job; ignore (the real sequencer will handle it)
        origin = msg.get("from")
        if self._admission_active() and not self._is_admitted(origin):
            self.stats["admission_denied"] = (
                self.stats.get("admission_denied", 0) + 1)
            return  # refuse to sequence for a non-admitted origin
        payload = base64.b64decode(msg["payload_b64"])
        result = await self._sequence_request(payload, timeout=5.0)
        await self._send(msg["from_addr"], {
            "type": MSG_REQUEST_RESULT,
            "from": self.node_id,
            "from_addr": self.address,
            "request_id": msg["request_id"],
            "result": result,
        })

    async def _handle_request_result(self, msg: dict):
        """Non-sequencer path: a node reported the outcome of a request we
        forwarded. Only accept it from a node we ACTUALLY sent this request_id
        to -- tighter than recomputing the sequencer (which may have rotated
        since we sent) and correct across a failover."""
        rid = msg.get("request_id")
        if msg.get("from") not in self._request_targets.get(rid, set()):
            return
        fut = self._request_waits.get(rid)
        if fut and not fut.done():
            fut.set_result(msg.get("result", {"status": "ERROR"}))

    async def _verifier_pool(self, payload_hash: str) -> List[str]:
        """Find the payload's XOR-closest neighbourhood by an iterative Kademlia
        lookup, returning candidate verifier ids (self included, closest-first).

        This is the scale-relevant change to verifier selection. Instead of
        scanning the whole roster -- which presumes we hold it -- we actively
        fetch only the nodes nearest the payload in ~O(log N) queries, then pick
        `verifier_fanout` of them by XOR proximity (with retry headroom for a
        disjoint set). Past the complete-roster boundary this re-discovers the
        payload's closest nodes from a partial table (the true closest is
        recovered reliably; the exact k-set is not guaranteed once buckets cannot
        hold everyone -- same near-closest property as `_lookup`).

        The lookup runs every request, but it is cheap when it is redundant: with
        a complete table the first round's queries return only already-known
        contacts, so it terminates after one round (~one RTT). We deliberately do
        NOT gate it on local table size -- holding `fanout` contacts says nothing
        about whether they are the ones *near this payload*, so a size gate would
        silently pick far nodes at scale. (Caching a converged neighbourhood per
        key-region is a possible future optimization.)

        Scope note: the round is still *announced* to every member via reliable
        broadcast so all chains adopt the key in lockstep. That O(N) fan-out is a
        separate concern (gossip dissemination). This method removes the
        full-roster requirement from *choosing* verifiers, not yet from
        *notifying* members."""
        await self._lookup(payload_hash)
        pool_size = max(self.verifier_fanout * MAX_VERIFY_ATTEMPTS, KAD_K)
        ids = [p["id"] for p in self.routing.closest(payload_hash, pool_size)]
        if self.node_id not in ids:
            ids.append(self.node_id)               # we can verify too
        return ids

    async def _sequence_request(self, payload: bytes,
                                timeout: float = 5.0) -> dict:
        """Sequencer-only: invite up to `verifier_fanout` verifiers by XOR
        proximity, broadcast the request to ALL members (each derives with its
        OWN salt), and confirm the key once `verifier_quorum` of them return a
        valid proof-of-derivation. The round resolves early on quorum, so a slow
        or offline verifier no longer stalls it. If a quorum can't be gathered,
        retry with a DISJOINT verifier set (excluding non-responders / bad
        provers) up to MAX_VERIFY_ATTEMPTS (audit 4.3).

        The salt (previous key) is NEVER transmitted. The round is bound to an
        explicit chain_index so adoption is single-winner-per-slot. The whole
        body runs under a serialization lock, so each request fully completes
        before the next opens -- every node advances in deterministic lockstep."""
        if self._request_lock is None:
            self._request_lock = asyncio.Lock()

        async with self._request_lock:
            if self.routing.count() < 2:
                return {"status": "ERROR",
                        "reason": "need >=2 other peers to form a quorum"}

            payload_hash = hashlib.sha256(payload).hexdigest()
            # Lookup-based verifier selection (scale): actively fetch the
            # payload's XOR-closest neighbourhood rather than scanning the whole
            # roster, so selection works from a partial routing table. Identical
            # to a full-roster scan at zone scale (see _verifier_pool).
            candidates = await self._verifier_pool(payload_hash)
            if len(set(candidates)) < self.verifier_quorum:
                return {"status": "ERROR",
                        "reason": "fewer than a quorum of verifiers reachable "
                                  "near the payload"}
            chain_index = self.chain.depth()
            salt = self.chain.current_salt()       # local only -- not sent
            b64payload = base64.b64encode(payload).decode()

            excluded: set = set()
            result = {"status": "TIMEOUT",
                      "reason": "no verifier quorum could be reached"}

            for attempt in range(MAX_VERIFY_ATTEMPTS):
                verifiers = ThreeWayVerification.select_verifiers(
                    payload_hash, candidates, self.verifier_fanout, exclude=excluded)
                # never silently weaken the guarantee: we must be able to invite
                # at least `verifier_quorum` distinct verifiers
                if len(verifiers) < self.verifier_quorum:
                    break
                k = self.verifier_quorum

                session_id = uuid.uuid4().hex[:12]
                rnd, _ = self.verifier.open_round(
                    session_id, payload, salt, verifiers,
                    quorum=k, iterations=self.mandelbrot_iter)

                fut = asyncio.get_running_loop().create_future()
                self._verify_waits[session_id] = fut

                # remember how to reconstruct this round for anti-entropy repair
                # BEFORE disseminating: on the in-memory path a verifier's commit
                # can return (and the round resolve) during the await below, so
                # the meta must already be present for _archive_round to find it.
                self._round_meta[session_id] = {
                    "payload_b64": b64payload, "iter": self.mandelbrot_iter}
                if len(self._round_meta) > CHAIN_ARCHIVE_MAX * 2:
                    self._round_meta.pop(next(iter(self._round_meta)))

                await self._disseminate({
                    "type": MSG_VERIFY_OPEN,
                    "from": self.node_id,
                    "from_addr": self.address,
                    "session_id": session_id,
                    "payload_b64": b64payload,
                    "verifiers": verifiers,
                    "quorum": k,
                    "iter": self.mandelbrot_iter,
                    "chain_index": chain_index,
                })

                try:
                    result = await asyncio.wait_for(fut, timeout)
                except asyncio.TimeoutError:
                    result = {"status": "TIMEOUT", "session_id": session_id,
                              "reason": "verifier quorum not reached in time"}
                finally:
                    self._verify_waits.pop(session_id, None)

                if result.get("status") == "VERIFIED":
                    return result

                # failed attempt: drop this round so a late proof can't resolve
                # it, then exclude everyone who didn't give a valid proof and try
                # a disjoint set on the next attempt.
                good = self.verifier.valid_provers(session_id)
                self.verifier.abandon(session_id)
                excluded |= (set(verifiers) - good)
                if len(set(candidates) - excluded) < self.verifier_quorum:
                    break                          # no fresh verifiers left to try

            return result

    async def _handle_verify_open(self, msg: dict):
        """Every member independently derives the key using its OWN chain salt
        (never a transmitted salt) and returns an identity-bound PROOF that it
        actually possesses the derived key (audit 1.1) -- not a fingerprint a
        free-rider could copy.

        Only rounds opened by the verifiably-chosen sequencer for the current
        epoch are honored. This stops a rogue node from driving its own round and
        getting members to adopt a key into a slot the sequencer did not order
        (audit 4.1)."""
        # If this open is for a chain slot ahead of us, the prior slot's
        # VERIFY_RESULT hasn't been adopted yet (it's still in flight). Deriving
        # now would use a stale salt and fail verification, so buffer it and
        # process it the instant our chain catches up -- keeps members in
        # lockstep when requests are pipelined over an async / lossy network. This
        # is checked BEFORE the sequencer test because the sequencer for a future
        # epoch can only be computed once we hold that epoch's boundary digest; a
        # buffered open is re-checked against the right sequencer when it drains.
        chain_index = msg.get("chain_index")
        if (chain_index is not None
                and 0 <= self.chain.depth() < chain_index
                and chain_index - self.chain.depth() <= MAX_PENDING_OPENS):
            self._pending_opens.setdefault(chain_index, []).append(msg)
            # a buffered open more than one slot ahead means we actually MISSED a
            # round (not just out-of-order pipelining): pull the gap so we don't
            # wait on a VERIFY_RESULT that never arrives.
            if chain_index > self.chain.depth() + 1:
                await self._maybe_request_catchup(chain_index)
            return

        # We are at our current depth/epoch here, so the epoch seed is local and
        # every honest member computes the same expected sequencer for this slot.
        if msg.get("from") != self._sequencer_for(self.chain.depth()):
            return  # not the sequencer for this epoch -- ignore unsanctioned round

        # Single-CANDIDATE-per-slot guard: if a DIFFERENT NODE's round is
        # already open at this exact depth (and is still fresh), refuse this
        # one. Membership changes (peer expiry) can transiently flip the
        # computed sequencer across nodes by a beacon tick; without this, two
        # candidates racing the same slot inside that skew could each gather
        # proofs from a different subset. First-accepted-candidate-wins per
        # member narrows that window to nothing in-zone, while the SAME
        # sequencer re-opening the slot (the quorum-retry path mints a fresh
        # session per attempt) passes freely. A candidate that died mid-flight
        # unblocks after DEPTH_GUARD_S so the legitimate successor can retake
        # the slot.
        d = self.chain.depth()
        session_id = msg["session_id"]
        opener = msg.get("from")
        g = self._depth_open
        if (g and g[0] == d and g[1] != session_id and g[2] != opener
                and time.time() - g[3] < DEPTH_GUARD_S):
            return  # another candidate holds this slot; let it resolve or expire
        self._depth_open = (d, session_id, opener, time.time())

        payload = base64.b64decode(msg["payload_b64"])
        verifiers = msg["verifiers"]
        iterations = msg.get("iter", self.mandelbrot_iter)
        salt = self.chain.current_salt()   # local only

        # Stash how to reconstruct this round so that, once we adopt it, WE can
        # also serve it to a future laggard (every member archives, not just the
        # sequencer -- this is what makes repair survive a sequencer handover).
        self._round_meta[session_id] = {
            "payload_b64": msg["payload_b64"], "iter": iterations}
        if len(self._round_meta) > CHAIN_ARCHIVE_MAX * 2:
            self._round_meta.pop(next(iter(self._round_meta)))

        rnd, my_proof = self.verifier.open_round(
            session_id, payload, salt,
            [v for v in verifiers if v != self.node_id],
            iterations=iterations)
        # Every member derives the key (above) so it can adopt the result and
        # stay in lockstep -- but only the SELECTED verifiers need to commit a
        # proof to the sequencer. Non-verifiers derive silently and wait for the
        # VERIFY_RESULT, which keeps the sequencer's inbound commits O(fanout),
        # not O(N). (The sequencer counts only the selected verifiers anyway, so
        # this changes load, not the outcome.)
        if self.node_id in verifiers:
            await self._send(msg["from_addr"], {
                "type": MSG_VERIFY_COMMIT,
                "from": self.node_id,
                "from_addr": self.address,
                "session_id": session_id,
                "proof": my_proof,
                "payload_hash": rnd.local_key.payload_hash,
            })

        # A VERIFY_RESULT for this session may have raced ahead of us and
        # arrived before this round existed. Apply it now if so.
        buffered = self._pending_results.pop(session_id, None)
        if buffered is not None:
            self._adopt_verified_key(session_id, buffered)
            await self._drain_pending_opens()

    async def _drain_pending_opens(self):
        """Process a buffered VERIFY_OPEN that our chain has now caught up to
        (its chain_index == our current depth). Called after each adoption, so
        pipelined opens unblock one slot at a time as the chain advances."""
        msgs = self._pending_opens.pop(self.chain.depth(), None)
        if msgs:
            for m in msgs:
                await self._handle_verify_open(m)

    # ----------------------------------------------- anti-entropy repair ------
    def _archive_round(self, chain_index: int, session_id: str,
                       verifiers: List[str], fingerprint: str):
        """Keep a bounded ring of committed rounds so we can later serve a
        lagging member exactly the slots it missed. Called by EVERY member on
        adoption (not only the sequencer), which is what lets repair survive a
        sequencer handover. Stores only what is needed to RE-DERIVE and
        re-confirm the key (payload, iter, verifiers, fingerprint) -- never the
        key itself."""
        meta = self._round_meta.pop(session_id, None)
        if not meta:
            return
        self._round_archive[chain_index] = {
            "chain_index": chain_index,
            "payload_b64": meta["payload_b64"],
            "iter": meta["iter"],
            "verifiers": list(verifiers),
            "fingerprint": fingerprint,
        }
        while len(self._round_archive) > CHAIN_ARCHIVE_MAX:
            self._round_archive.pop(min(self._round_archive))

    async def _maybe_request_catchup(self, target_index: int):
        """Member-side: we've seen a chain_index ahead of our own, so we missed
        one or more rounds. Ask several archive-holders -- the current sequencer
        AND other peers known (from signed beacons) to be at least as deep as the
        target -- so repair proceeds even if the sequencer has departed or is the
        wrong/ambiguous node during a handover. Whatever they return is adopted
        only up to a head digest anchored by a quorum or the sequencer (see
        _handle_chain_batch), so asking an untrusted peer is safe. Rate-limited,
        which also retries a lost batch."""
        if target_index <= self.chain.depth():
            return
        now = time.time()
        if now - self._last_catchup_req < CATCHUP_COOLDOWN_S:
            return
        self._last_catchup_req = now
        peers = self.routing.all_peers()

        def addr_of(pid):
            p = peers.get(pid)
            return p.get("address") if p else None

        targets: List[str] = []
        seq_id = self._current_sequencer()
        if seq_id != self.node_id and addr_of(seq_id):
            targets.append(seq_id)
        # then other peers attested at a depth beyond ours, lowest-id first
        # (deterministic), until we have a fan-out's worth of servers to ask
        for pid in sorted(self._head_attest):
            if len(targets) >= CATCHUP_FANOUT:
                break
            if pid == self.node_id or pid in targets:
                continue
            if self._head_attest[pid].get("depth", -1) > self.chain.depth() and addr_of(pid):
                targets.append(pid)

        for pid in targets[:CATCHUP_FANOUT]:
            await self._send(addr_of(pid), {
                "type": MSG_CHAIN_REQUEST,
                "from": self.node_id,
                "from_addr": self.address,
                "have_depth": self.chain.depth(),
            })

    async def _handle_chain_request(self, msg: dict):
        """Server-side (ANY member, not just the sequencer): a lagging member
        asked for the slots it missed. Reply with our archived rounds in
        [have_depth, our_depth), oldest first, capped at CHAIN_BATCH_MAX. A node
        with no relevant archive simply sends nothing. The requester re-derives
        and anchors what we send to a quorum / the sequencer before adopting, so
        serving carries no special trust -- it only needs to be a node that
        retained the rounds (which every member now does). Under an admission
        policy, only ADMITTED requesters are served -- repair is a member
        service, and round payloads / serving bandwidth are not owed to a node
        the policy keeps out."""
        if self._admission_active() and not self._is_admitted(msg.get("from")):
            return
        have = msg.get("have_depth", 0)
        if not isinstance(have, int) or have < 0:
            return
        rounds = []
        for ci in sorted(self._round_archive):
            if have <= ci < self.chain.depth():
                rounds.append(self._round_archive[ci])
                if len(rounds) >= CHAIN_BATCH_MAX:
                    break
        if not rounds:
            return
        await self._send(msg["from_addr"], {
            "type": MSG_CHAIN_BATCH,
            "from": self.node_id,
            "from_addr": self.address,
            "rounds": rounds,
        })

    def _digest_anchored(self, depth: int, digest_hex: str) -> bool:
        """True iff the head digest `digest_hex` at `depth` is vouched for by a
        trusted source: the sequencer for that depth's epoch (its signed beacon),
        OR a quorum of distinct peers' signed beacons. This is the binding that
        lets us accept a catch-up batch from ANY server: a forged sequence
        re-derives to a digest no honest quorum (and no honest sequencer)
        attested, so it is refused. (The quorum path is the workhorse during
        catch-up, where the far-ahead epoch seed may not yet be local; a misjudged
        sequencer id can only fail to anchor, never anchor a forgery, since honest
        peers attest only canonical digests.)"""
        seq_id = self._sequencer_for(depth)
        a = self._head_attest.get(seq_id)
        if a and a.get("depth") == depth and a.get("digest") == digest_hex:
            return True
        votes = sum(1 for at in self._head_attest.values()
                    if at.get("depth") == depth and at.get("digest") == digest_hex)
        return votes >= REQUIRED_QUORUM

    async def _handle_chain_batch(self, msg: dict):
        """Member-side: apply missed rounds pulled from a peer. Each slot is
        RE-DERIVED locally from our own salt (the server never sends a key), so a
        tampered payload yields a different fingerprint and is refused. Slots are
        applied strictly in ascending order. The batch may come from ANY member;
        what bounds trust is the *head digest*: we adopt only the longest prefix
        whose re-derived head digest is anchored by a quorum (or the sequencer's
        beacon). For backward compatibility and bootstrap, a batch from the
        current sequencer is trusted directly (the prior single-sequencer model);
        the quorum-anchored path additionally tolerates a malicious server -- or
        even a malicious sequencer -- and survives a sequencer handover."""
        rounds = msg.get("rounds") or []
        if not isinstance(rounds, list) or not rounds:
            return
        from_seq = (msg.get("from") == self._current_sequencer())

        # Dry-run: re-derive the contiguous run WITHOUT mutating the chain,
        # tracking the running head digest at each step so we can decide how far
        # we may safely commit before touching the chain at all.
        salt = self.chain.current_salt()
        digest = bytes.fromhex(self.chain.head_digest())
        depth = self.chain.depth()
        plan = []  # (derived, verifiers, resulting_depth, resulting_digest_hex, raw_round)
        for r in sorted(rounds, key=lambda x: x.get("chain_index", -1)):
            ci = r.get("chain_index")
            if not isinstance(ci, int):
                break
            if ci < depth:
                continue            # already have this slot -- skip, keep scanning
            if ci != depth:
                break               # a gap we can't bridge yet -> stop
            try:
                payload = base64.b64decode(r["payload_b64"], validate=True)
                iters = int(r.get("iter", self.mandelbrot_iter))
            except (binascii.Error, ValueError, TypeError, KeyError):
                break
            derived = PiMandelbrotKeyEngine.derive(payload, salt, iters)
            if derived.fingerprint != r.get("fingerprint"):
                break               # our salt/payload disagree -> refuse to corrupt
            salt = derived.fernet_key
            digest = hashlib.sha256(
                digest + depth.to_bytes(8, "big") + derived.fingerprint.encode()).digest()
            depth += 1
            plan.append((derived, r.get("verifiers", []), depth, digest.hex(), r))
        if not plan:
            return

        # Commit the LONGEST prefix that is trusted: from the sequencer directly,
        # or whose resulting head digest is quorum/sequencer-anchored.
        commit_upto = 0
        for i, (_, _, d, dhex, _) in enumerate(plan):
            if from_seq or self._digest_anchored(d, dhex):
                commit_upto = i + 1
        if commit_upto == 0:
            return

        applied = 0
        for derived, verifiers, d, dhex, r in plan[:commit_upto]:
            entry = self.chain.append(derived, verifiers)
            # retain what we adopted so we can serve it to the next laggard
            self._round_archive[entry.chain_index] = r
            self.stats["keys_verified"] += 1
            self.stats["keys_caught_up"] = self.stats.get("keys_caught_up", 0) + 1
            applied += 1
        if applied:
            while len(self._round_archive) > CHAIN_ARCHIVE_MAX:
                self._round_archive.pop(min(self._round_archive))
            # we advanced: drop now-stale buffered opens and let the rest proceed
            for k in [k for k in self._pending_opens if k < self.chain.depth()]:
                self._pending_opens.pop(k, None)
            await self._drain_pending_opens()

    async def _handle_verify_commit(self, msg: dict):
        """Initiator collects a verifier's proof of derivation and checks it
        against its own key."""
        session_id = msg["session_id"]
        result = self.verifier.submit_commitment(
            session_id, msg["from"], msg.get("proof", ""),
            payload_hash=msg.get("payload_hash"))
        if result is None:
            return  # still collecting

        # round resolved
        verifiers = self.verifier.rounds[session_id].required_verifiers
        if result["status"] == "VERIFIED":
            entry = self.chain.append(result["derived"], result["verifiers"])
            self.stats["keys_verified"] += 1
            self.stats["last_report"] = (
                f"key verified | chain[{entry.chain_index}] "
                f"fp={result['fingerprint'][:12]}")
            self._archive_round(entry.chain_index, session_id,
                                result["verifiers"], result.get("fingerprint", ""))
        else:
            self.stats["keys_rejected"] += 1
            self.stats["last_report"] = f"key REJECTED: {result.get('reason')}"

        # tell ALL members the outcome (so every chain adopts in lockstep)
        if result["status"] == "VERIFIED":
            await self._disseminate({
                "type": MSG_VERIFY_RESULT,
                "from": self.node_id,
                "from_addr": self.address,
                "session_id": session_id,
                "status": result["status"],
                "fingerprint": result.get("fingerprint", ""),
                "chain_index": entry.chain_index,
            })

        fut = self._verify_waits.get(session_id)
        if fut and not fut.done():
            # don't leak the raw key object through the public result
            public = {k: v for k, v in result.items() if k != "fernet_key" and k != "derived"}
            public["chain_depth"] = self.chain.depth()
            fut.set_result(public)

    async def _handle_verify_result(self, msg: dict):
        """A member adopts the verified key into its own chain so the next
        round's salt stays synchronized. Only results from the epoch's sequencer
        are honored (audit 4.1). If this node hasn't opened its round yet (the
        broadcast outran the open), buffer the result; the open handler applies
        it. Adopts exactly once per session."""
        session_id = msg["session_id"]
        # A result for a slot beyond our chain means we missed one or more rounds
        # (a gossip coverage gap). This is just a "we're behind" signal from a
        # signed peer, so trigger catch-up regardless of who sent it (the pull is
        # rate-limited and its adoption is independently anchored).
        if msg.get("chain_index", -1) > self.chain.depth():
            await self._maybe_request_catchup(msg["chain_index"])
        # Adoption (and stashing for the open-after-result race) is gated on the
        # epoch's verifiably-chosen sequencer. We adopt only at our current depth,
        # where the epoch seed is local and every honest member agrees on it.
        if msg.get("from") != self._sequencer_for(self.chain.depth()):
            return  # not the sequencer for this epoch -- ignore unsanctioned result
        rnd = self.verifier.rounds.get(session_id)
        if not rnd or not rnd.local_key:
            # round not open yet -- stash and let _handle_verify_open apply it
            self._pending_results[session_id] = msg
            return
        self._adopt_verified_key(session_id, msg)
        await self._drain_pending_opens()

    def _adopt_verified_key(self, session_id: str, msg: dict):
        """Adopt a globally-verified key into the local chain exactly once.
        Only adopts if our independently-derived fingerprint matches the
        verified one -- a node whose salt had drifted will refuse to adopt
        rather than corrupt its chain."""
        rnd = self.verifier.rounds.get(session_id)
        if not rnd or not rnd.local_key:
            return
        if rnd.status != "PENDING":
            return  # already adopted / resolved
        if msg.get("status") == "VERIFIED" and \
                rnd.local_key.fingerprint == msg.get("fingerprint"):
            rnd.status = "VERIFIED"
            entry = self.chain.append(rnd.local_key, rnd.required_verifiers)
            self.stats["keys_verified"] += 1
            # archive this round so we can serve it to a future laggard
            self._archive_round(entry.chain_index, session_id,
                                rnd.required_verifiers, rnd.local_key.fingerprint)

    # ----------------------------------------------- distributed tensor -------
    def _select_compute_peers(self) -> List[dict]:
        """Pick worker peers for a tensor job, PREFERRING the local geo-zone
        (option A locality). Strict per-zone sharding would shrink the set the
        coordinate-wise MAD filter sees, weakening Byzantine rejection and making
        a zone-local Sybil cheaper -- so we keep a floor: if same-zone peers
        (plus our own local contribution) are fewer than MIN_TENSOR_CONTRIBUTORS,
        we top up with the XOR-nearest out-of-zone peers. Honest tradeoff:
        big zones stay fully local; small zones borrow a few neighbours to stay
        Byzantine-safe."""
        allp = list(self.routing.all_peers().values())
        if not allp:
            return []
        same = [p for p in allp if p.get("geo_zone", "UNKNOWN") == self.geo_zone]
        need_peers = max(0, MIN_TENSOR_CONTRIBUTORS - 1)   # -1: local counts too
        if len(same) >= need_peers or len(same) == len(allp):
            return same
        local_int = int(self.node_id[:40], 16)
        others = [p for p in allp if p.get("geo_zone", "UNKNOWN") != self.geo_zone]
        others.sort(key=lambda p: local_int ^ int(p["id"][:40], 16))
        return same + others[:need_peers - len(same)]

    async def run_tensor_job(self, tensor: np.ndarray, rank: int = 4,
                             timeout: float = 5.0,
                             projection: Optional[np.ndarray] = None) -> dict:
        """Anchor-side: shard the first axis across worker peers, collect
        yields, MAD-filter, reduce. Falls back to local compute if no peers.
        Workers are chosen zone-locally where possible (see
        _select_compute_peers)."""
        self.stats["tensor_jobs"] += 1
        if projection is None:
            rng = np.random.default_rng(42)
            projection = rng.standard_normal((rank, tensor.shape[0]))

        peers = self._select_compute_peers()
        job_id = uuid.uuid4().hex[:12]

        if not peers:
            # local fallback -- still exercises the engine
            mapped = TensorMapWorker.map_slice(tensor, projection, self.hw_weight)
            res = TensorReducer.reduce({self.node_id: mapped},
                                       {self.node_id: self.hw_weight})
            self._record_reduction(res)
            return res

        # collector
        collector = {
            "future": asyncio.get_running_loop().create_future(),
            "yields": {}, "weights": {},
            "expected": len(peers), "projection": projection,
            "dispatched": {p["id"] for p in peers},   # only these may yield
        }
        self._pending_tensor[job_id] = collector

        b64proj = base64.b64encode(projection.astype(np.float64).tobytes()).decode()
        proj_shape = list(projection.shape)
        b64tensor = base64.b64encode(tensor.astype(np.float64).tobytes()).decode()
        tensor_shape = list(tensor.shape)

        for p in peers:
            await self._send(p["address"], {
                "type": MSG_TENSOR_TASK,
                "from": self.node_id,
                "from_addr": self.address,
                "job_id": job_id,
                "tensor_b64": b64tensor,
                "tensor_shape": tensor_shape,
                "proj_b64": b64proj,
                "proj_shape": proj_shape,
            })

        # also compute locally
        local = TensorMapWorker.map_slice(tensor, projection, self.hw_weight)
        collector["yields"][self.node_id] = local
        collector["weights"][self.node_id] = self.hw_weight

        try:
            await asyncio.wait_for(collector["future"], timeout)
        except asyncio.TimeoutError:
            pass

        res = TensorReducer.reduce(collector["yields"], collector["weights"])
        self._pending_tensor.pop(job_id, None)
        self._record_reduction(res)
        return res

    @staticmethod
    def _safe_decode_array(b64: str, shape) -> Optional[np.ndarray]:
        """Decode a network-supplied float64 array, rejecting anything whose
        shape is malformed, oversized, or inconsistent with the byte length
        (audit 3.1). Returns None on any problem instead of raising."""
        if not isinstance(shape, list) or not shape:
            return None
        dims = []
        total = 1
        for d in shape:
            if not isinstance(d, int) or d <= 0:
                return None
            total *= d
            if total > MAX_TENSOR_ELEMENTS:
                return None
            dims.append(d)
        try:
            raw = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError, TypeError):
            return None
        if len(raw) != total * 8:        # float64 == 8 bytes
            return None
        try:
            return np.frombuffer(raw, dtype=np.float64).reshape(tuple(dims))
        except ValueError:
            return None

    def _tensor_rate_ok(self, peer_id: str) -> bool:
        """Simple per-peer token check to bound unsolicited compute (audit 2.2)."""
        now = time.time()
        hits = [t for t in self._tensor_hits.get(peer_id, []) if now - t < TENSOR_RATE_WINDOW]
        if len(hits) >= TENSOR_RATE_MAX:
            self._tensor_hits[peer_id] = hits
            return False
        hits.append(now)
        self._tensor_hits[peer_id] = hits
        return True

    async def _handle_tensor_task(self, msg: dict):
        """Worker-side: compute the assigned slice and return the yield.
        Only authenticated, known session members may task this node, at a
        bounded rate, with size-validated input (audit 2.2, 3.1)."""
        if self.hw_weight <= 0.0:
            return  # data governor: cellular / low battery -> skip
        requester = msg.get("from")
        # membership: ignore tensor work requests from non-peers
        if requester not in self.routing.all_peers():
            return
        if not self._tensor_rate_ok(requester):
            return  # rate-limited
        tensor = self._safe_decode_array(msg.get("tensor_b64", ""), msg.get("tensor_shape"))
        proj = self._safe_decode_array(msg.get("proj_b64", ""), msg.get("proj_shape"))
        if tensor is None or proj is None:
            return  # malformed / oversized -> drop
        if tensor.ndim != 3 or proj.ndim != 2 or proj.shape[1] != tensor.shape[0]:
            return  # shape contract not met
        mapped = TensorMapWorker.map_slice(tensor, proj, self.hw_weight)
        await self._send(msg["from_addr"], {
            "type": MSG_TENSOR_YIELD,
            "from": self.node_id,
            "from_addr": self.address,
            "job_id": msg["job_id"],
            "yield_b64": base64.b64encode(mapped.astype(np.float64).tobytes()).decode(),
            "yield_shape": list(mapped.shape),
            "weight": self.hw_weight,
        })

    async def _handle_tensor_yield(self, msg: dict):
        c = self._pending_tensor.get(msg.get("job_id"))
        if not c:
            return
        sender = msg.get("from")
        # only count yields from peers we actually dispatched this job to
        if sender not in c["dispatched"]:
            return
        arr = self._safe_decode_array(msg.get("yield_b64", ""), msg.get("yield_shape"))
        if arr is None:
            return
        c["yields"][sender] = arr
        try:
            c["weights"][sender] = float(msg.get("weight", 1.0))
        except (TypeError, ValueError):
            c["weights"][sender] = 1.0
        # +1 for the local contribution already present
        if len(c["yields"]) >= c["expected"] + 1 and not c["future"].done():
            c["future"].set_result(True)

    def _record_reduction(self, res: dict):
        if res.get("status") == "SUCCESS":
            self.stats["last_energy"] = res["energy"]
            self.stats["peak_z"] = res.get("peak_z", 0.0)
            self.stats["byzantine_purged"] += len(res.get("purged", []))
            self.stats["last_report"] = res["report"]
            JuvianHistoryLedger.log(
                res["energy"], res.get("dominant_modes", []),
                len(res.get("valid_ids", [])) + len(res.get("purged", [])),
                len(res.get("purged", [])), self.history_path)

    # ------------------------------------------------------------ persistence -
    def persist_chain(self):
        cid_dict, ref = self.chain.to_cid_dict()
        if cid_dict:
            FractalPersistenceManager.compress_and_save(
                cid_dict, ref, output_file=self.chain_path)

    def snapshot(self) -> dict:
        """A JSON-safe snapshot for dashboards."""
        return {
            "name": self.name,
            "node_id": self.node_id[:16],
            "address": self.address,
            "device_type": self.device_type,
            "hw_weight": self.hw_weight,
            "peers": self.routing.count(),
            "chain_depth": self.chain.depth(),
            "chain_digest": self.chain.head_digest(),
            "genesis_source": self.genesis_source,
            "stats": dict(self.stats),
            "latest_fingerprint": (self.chain.entries[-1].fingerprint[:16]
                                   if self.chain.entries else "GENESIS"),
        }
