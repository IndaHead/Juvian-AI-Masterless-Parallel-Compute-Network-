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
JUVIAN GRID :: CRYPTO CORE
pi-Mandelbrot per-request key derivation + strict 3-of-3 verification
==============================================================================

DERIVATION (deterministic, never transmitted):
    payload_bytes
        |
        v  SHA-256
    32-byte digest
        |
        v  extract two uint64 segments, scale through pi into Mandelbrot bounds
    c = complex(real, imag)
        |
        v  iterate z = z^2 + c for N steps from z=0, record the escape path
    escape_path (the cryptographic material)
        |
        v  SHA-256( escape_path || final_z || escape_iter || SALT )
    32-byte key  ->  base64  ->  Fernet key

CHAINING:
    SALT for request k = the verified key from request k-1.
    The genesis salt (request 0) is a session bootstrap secret that all
    participants share (see SessionBootstrap). After each successful 3-of-3
    round every participant holds the new key, so the next round's salt is
    shared automatically.

3-OF-3 VERIFICATION:
    Three devices, selected by XOR proximity to the payload hash, each derive
    the key independently and broadcast only its fingerprint (SHA-256 of the
    key -- never the key itself). All three fingerprints must match exactly or
    the round is rejected.

SECURITY NOTE (read the README): this scheme guarantees (a) keys are never put
on the wire, (b) multiple independent devices agree on the key, and (c) any
tampering with the payload or salt produces an instantly-visible fingerprint
mismatch. Confidentiality against an outside eavesdropper depends on the
genesis salt being secret -- that is what SessionBootstrap is for.
==============================================================================
"""

import os
import math
import time
import json
import hmac
import struct
import hashlib
import base64
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

from cryptography.fernet import Fernet

PI = math.pi


def derivation_proof(fernet_key: bytes, session_id: str, node_id: str) -> str:
    """Identity-bound proof that the holder actually possesses the derived key.

    proof = HMAC-SHA256( key , "JUVIAN-POD:" + session_id + ":" + node_id )

    Binding the proof to the prover's own node_id is what defeats free-riding
    (audit 1.1): the proof is different for every node, so a verifier cannot
    pass by copying a proof it observed on the wire -- a copied proof is bound
    to someone else's id and fails the check. Producing a *valid* proof for
    one's own id requires the key, which a non-deriving node does not have."""
    msg = f"JUVIAN-POD:{session_id}:{node_id}".encode()
    return hmac.new(fernet_key, msg, hashlib.sha256).hexdigest()

# Mandelbrot iteration depth. 1000 in the daemon; callers may lower for speed.
DEFAULT_ITER = 1000

# Valid Mandelbrot coordinate window
R_MIN, R_MAX = -2.5, 1.0
I_MIN, I_MAX = -1.25, 1.25


# ==============================================================================
# 1. PI-MANDELBROT KEY ENGINE
# ==============================================================================

@dataclass
class DerivedKey:
    fernet_key: bytes          # the live key (kept local, never transmitted)
    fingerprint: str           # SHA-256(fernet_key) hex -- safe to share
    c_real: float
    c_imag: float
    escape_iter: int
    escape_signature: str      # short checksum of the escape path
    payload_hash: str


class PiMandelbrotKeyEngine:

    @staticmethod
    def _payload_to_coordinates(payload: bytes) -> complex:
        digest = hashlib.sha256(payload).digest()
        real_int = struct.unpack(">Q", digest[0:8])[0]
        imag_int = struct.unpack(">Q", digest[8:16])[0]
        max_u64 = (2 ** 64) - 1
        rn = real_int / max_u64
        inn = imag_int / max_u64
        # pi modulation -> non-linear spread across the valid window
        real = R_MIN + (math.sin(PI * rn) * 0.5 + 0.5) * (R_MAX - R_MIN)
        imag = I_MIN + (math.sin(PI * inn) * 0.5 + 0.5) * (I_MAX - I_MIN)
        return complex(real, imag)

    @staticmethod
    def _iterate(c: complex, iterations: int) -> Tuple[complex, List[float], int]:
        z = complex(0, 0)
        path: List[float] = []
        escape_iter = iterations
        escaped = False
        for i in range(iterations):
            z = z * z + c
            mag = abs(z)
            # guard against overflow to inf on fast-diverging points
            if math.isinf(mag) or math.isnan(mag):
                mag = 1e308
                path.append(mag)
                if not escaped:
                    escape_iter = i
                    escaped = True
                # fill remaining deterministically and stop real math
                path.extend([mag] * (iterations - i - 1))
                break
            path.append(mag)
            if mag > 2.0 and not escaped:
                escape_iter = i
                escaped = True
        return z, path, escape_iter

    @classmethod
    def derive(cls, payload: bytes, salt: bytes,
               iterations: int = DEFAULT_ITER) -> DerivedKey:
        c = cls._payload_to_coordinates(payload)
        final_z, path, escape_iter = cls._iterate(c, iterations)

        path_bytes = struct.pack(f">{len(path)}d", *path)
        material = (
            path_bytes
            + struct.pack(">dd", final_z.real, final_z.imag)
            + struct.pack(">I", escape_iter)
            + salt
        )
        digest = hashlib.sha256(material).digest()
        fernet_key = base64.urlsafe_b64encode(digest)
        fingerprint = hashlib.sha256(fernet_key).hexdigest()
        escape_sig = hashlib.sha256(path_bytes).hexdigest()[:16]

        return DerivedKey(
            fernet_key=fernet_key,
            fingerprint=fingerprint,
            c_real=c.real,
            c_imag=c.imag,
            escape_iter=escape_iter,
            escape_signature=escape_sig,
            payload_hash=hashlib.sha256(payload).hexdigest(),
        )


# ==============================================================================
# 2. SESSION BOOTSTRAP  (genesis salt)
# ==============================================================================

class SessionBootstrap:
    """
    Produces the genesis salt shared by all participants of a session.

    Two modes:
      * pre_shared(secret):  everyone is configured with the same secret.
      * from_participants(): salt = SHA-256(sorted participant ids || nonce).
        The nonce is public; this gives a *deterministic shared* genesis salt
        so 3-of-3 can pass, but it is NOT confidential. Use pre_shared (or wire
        in ECDH -- see README) when confidentiality against eavesdroppers
        matters.
    """

    @staticmethod
    def pre_shared(secret: str) -> bytes:
        return hashlib.sha256(("JUVIAN_PSK::" + secret).encode()).digest()

    @staticmethod
    def from_participants(participant_ids: List[str], nonce: str) -> bytes:
        joined = "|".join(sorted(participant_ids))
        return hashlib.sha256(f"JUVIAN_GENESIS::{joined}::{nonce}".encode()).digest()


# ==============================================================================
# 3. SESSION KEY CHAIN
# ==============================================================================

@dataclass
class ChainEntry:
    chain_index: int
    fingerprint: str
    c_real: float
    c_imag: float
    escape_signature: str
    escape_iter: int
    verifier_ids: List[str]
    payload_hash: str
    previous_fingerprint: str
    timestamp: float


class SessionKeyChain:
    # Keep the chain functional indefinitely without unbounded memory growth:
    # only the current key is needed to operate; older entries are audit history.
    # We retain a bounded recent window in memory (the full chain should be
    # checkpointed to disk by the caller if complete history is required).
    MAX_IN_MEMORY_ENTRIES = 10000

    def __init__(self, device_id: str, genesis_salt: bytes):
        self.device_id = device_id
        self._genesis_salt = genesis_salt
        self._current_key: Optional[bytes] = None
        self.entries: List[ChainEntry] = []
        self._total_depth = 0          # true depth, even after trimming
        self._lock = threading.Lock()
        # Verifiable head digest: a cumulative hash chain over the ordered
        # (chain_index, fingerprint) pairs. Two members that adopted the same
        # rounds in the same order share the identical digest, so it is a compact,
        # order-binding commitment to the whole chain. A lagging member can verify
        # that a catch-up batch it re-derived reproduces a head digest attested by
        # a trusted source (the sequencer, or a quorum), which lets ANY member --
        # not just the original sequencer -- safely serve anti-entropy repair.
        self._DIGEST_SEED = hashlib.sha256(b"juvian-chain-digest-v1").digest()
        self._head_digest = self._DIGEST_SEED
        self._digest_at: dict = {0: self._head_digest.hex()}  # depth -> digest hex
        self._DIGEST_RETAIN = 512

    def current_salt(self) -> bytes:
        with self._lock:
            return self._current_key if self._current_key is not None else self._genesis_salt

    def install_genesis_salt(self, salt: bytes) -> bool:
        """Replace the genesis salt with a freshly-agreed one (e.g. from group
        key agreement). Only permitted before any key is chained, so it cannot
        rewrite history. Returns True if installed."""
        with self._lock:
            if self.entries or self._current_key is not None:
                return False
            self._genesis_salt = salt
            return True

    def current_key(self) -> Optional[bytes]:
        with self._lock:
            return self._current_key

    def depth(self) -> int:
        with self._lock:
            return self._total_depth

    def append(self, derived: DerivedKey, verifier_ids: List[str]) -> ChainEntry:
        with self._lock:
            prev_fp = self.entries[-1].fingerprint if self.entries else (
                "GENESIS" if self._total_depth == 0 else "TRIMMED")
            entry = ChainEntry(
                chain_index=self._total_depth,
                fingerprint=derived.fingerprint,
                c_real=derived.c_real,
                c_imag=derived.c_imag,
                escape_signature=derived.escape_signature,
                escape_iter=derived.escape_iter,
                verifier_ids=list(verifier_ids),
                payload_hash=derived.payload_hash,
                previous_fingerprint=prev_fp,
                timestamp=time.time(),
            )
            self.entries.append(entry)
            self._total_depth += 1
            self._current_key = derived.fernet_key
            # advance the verifiable head digest with this slot (order-binding)
            self._head_digest = hashlib.sha256(
                self._head_digest
                + entry.chain_index.to_bytes(8, "big")
                + entry.fingerprint.encode()).digest()
            self._digest_at[self._total_depth] = self._head_digest.hex()
            if len(self._digest_at) > self._DIGEST_RETAIN:
                self._digest_at.pop(min(self._digest_at))
            # bound memory: drop oldest entries beyond the window
            if len(self.entries) > self.MAX_IN_MEMORY_ENTRIES:
                self.entries = self.entries[-self.MAX_IN_MEMORY_ENTRIES:]
            return entry

    def head_digest(self) -> str:
        """Hex digest committing to the entire ordered chain at the current
        depth. Identical across members that adopted the same rounds in order."""
        with self._lock:
            return self._head_digest.hex()

    def digest_at(self, depth: int) -> Optional[str]:
        """The head digest as it stood at `depth` slots, if still retained
        (bounded window). Lets a member confirm a past head a peer is repairing
        toward, even if the member has since advanced further."""
        with self._lock:
            return self._digest_at.get(depth)

    def to_cid_dict(self) -> Tuple[Dict[str, str], Dict[str, dict]]:
        """Serialise the chain into the (cid_dict, reference_universe) shape
        used by FractalPersistenceManager."""
        with self._lock:
            cid_dict, ref = {}, {}
            for e in self.entries:
                key = f"chain:{e.chain_index}"
                cid_dict[key] = e.fingerprint
                ref[key] = {
                    "chain_index": e.chain_index,
                    "fingerprint": e.fingerprint,
                    "c_coord": [e.c_real, e.c_imag],
                    "escape_signature": e.escape_signature,
                    "escape_iter": e.escape_iter,
                    "verifier_ids": e.verifier_ids,
                    "payload_hash": e.payload_hash,
                    "previous_fingerprint": e.previous_fingerprint,
                    "timestamp": e.timestamp,
                }
            return cid_dict, ref


# ==============================================================================
# 4. STRICT 3-OF-3 VERIFICATION
# ==============================================================================

@dataclass
class VerificationRound:
    session_id: str
    payload_hash: str
    required_verifiers: List[str]
    salt: bytes
    iterations: int
    quorum: int = 0                                        # min valid proofs to verify
    local_key: Optional[DerivedKey] = None
    proofs: Dict[str, str] = field(default_factory=dict)   # node_id -> proof
    status: str = "PENDING"
    created_at: float = field(default_factory=time.time)


# cap on retained rounds, to bound memory in a long-running daemon (audit 3.2)
MAX_RETAINED_ROUNDS = 512


class ThreeWayVerification:
    """Quorum consensus by proof-of-derivation. Verifiers are chosen by XOR
    proximity to the payload hash; any device type is eligible.

    Consensus is not 'the same fingerprint string was sent' (which a free-rider
    could copy) but 'at least `quorum` selected verifiers each produced a valid
    identity-bound proof of possessing the SAME key this node derived'. A round
    resolves as soon as the quorum of valid proofs arrives (so a slow or absent
    verifier no longer stalls it -- audit 4.3), and is rejected only once the
    quorum is provably unreachable. Because every proof is checked against this
    node's own key, an absent or wrong proof can never force a wrong key through;
    it simply does not count -- so tolerating up to (n - quorum) failures is a
    pure liveness gain, not a weakening of integrity."""

    def __init__(self, device_id: str):
        self.device_id = device_id
        self.rounds: Dict[str, VerificationRound] = {}
        self._order: List[str] = []          # insertion order for eviction
        self._lock = threading.Lock()

    # -- verifier selection -------------------------------------------------
    @staticmethod
    def select_verifiers(payload_hash_hex: str, peer_ids: List[str],
                         count: int = 3, exclude: Optional[set] = None) -> List[str]:
        exclude = exclude or set()
        pool = [p for p in peer_ids if p not in exclude]
        if len(pool) <= count:
            return list(pool)
        target = int(payload_hash_hex[:40], 16)
        scored = []
        for pid in pool:
            try:
                pv = int(pid[:40], 16)
            except ValueError:
                pv = int(hashlib.sha256(pid.encode()).hexdigest()[:40], 16)
            scored.append((target ^ pv, pid))
        scored.sort(key=lambda x: x[0])
        return [pid for _, pid in scored[:count]]

    def _register(self, session_id: str, rnd: VerificationRound):
        """Store a round and evict the oldest if we are over the cap."""
        self.rounds[session_id] = rnd
        self._order.append(session_id)
        while len(self._order) > MAX_RETAINED_ROUNDS:
            old = self._order.pop(0)
            self.rounds.pop(old, None)

    # -- round lifecycle ----------------------------------------------------
    def open_round(self, session_id: str, payload: bytes, salt: bytes,
                   required_verifiers: List[str], quorum: int = 0,
                   iterations: int = DEFAULT_ITER) -> Tuple[VerificationRound, str]:
        """Open a round, derive this node's key, and return (round, proof) where
        proof is this node's identity-bound proof of derivation to broadcast.
        `quorum` is the number of valid proofs required to verify (0 -> require
        all of `required_verifiers`, i.e. strict unanimity)."""
        derived = PiMandelbrotKeyEngine.derive(payload, salt, iterations)
        my_proof = derivation_proof(derived.fernet_key, session_id, self.device_id)
        with self._lock:
            rnd = VerificationRound(
                session_id=session_id,
                payload_hash=derived.payload_hash,
                required_verifiers=list(required_verifiers),
                salt=salt,
                iterations=iterations,
                quorum=quorum if quorum > 0 else len(required_verifiers),
                local_key=derived,
            )
            rnd.proofs[self.device_id] = my_proof
            self._register(session_id, rnd)
        return rnd, my_proof

    def submit_commitment(self, session_id: str, node_id: str,
                          proof: str,
                          payload_hash: Optional[str] = None) -> Optional[dict]:
        """Record a verifier's proof of derivation and check it against
        HMAC(our_key, session_id, that_verifier_id). Resolves the round as soon
        as `quorum` selected verifiers have produced VALID proofs (early
        success, so stragglers don't stall it); rejects only once the quorum is
        provably unreachable. Resolves exactly once; late proofs are ignored."""
        with self._lock:
            rnd = self.rounds.get(session_id)
            if rnd is None or rnd.local_key is None:
                return None
            if rnd.status != "PENDING":
                return None  # already resolved / abandoned -- ignore
            if payload_hash is not None and payload_hash != rnd.payload_hash:
                rnd.status = "REJECTED_PAYLOAD_MISMATCH"
                return {"status": "REJECTED", "reason": "payload hash mismatch",
                        "node_id": node_id, "session_id": session_id}

            rnd.proofs[node_id] = proof          # record this proof

            need = set(rnd.required_verifiers)
            if not need:
                return None                       # malformed round
            k = rnd.quorum if rnd.quorum > 0 else len(need)

            # tally valid / invalid proofs among the SELECTED verifier set only
            key = rnd.local_key.fernet_key
            valid_ids, invalid = [], 0
            for nid in need:
                p = rnd.proofs.get(nid)
                if p is None:
                    continue                      # not yet responded
                if hmac.compare_digest(derivation_proof(key, session_id, nid), p):
                    valid_ids.append(nid)
                else:
                    invalid += 1

            # early success: quorum of independent valid proofs reached
            if len(valid_ids) >= k:
                rnd.status = "VERIFIED"
                return {"status": "VERIFIED", "session_id": session_id,
                        "fingerprint": rnd.local_key.fingerprint,
                        "fernet_key": key, "derived": rnd.local_key,
                        "verifiers": sorted(valid_ids), "quorum": k}

            # provably unreachable: even if every not-yet-seen verifier is valid,
            # we still couldn't reach the quorum
            pending = len(need) - len(valid_ids) - invalid
            if len(valid_ids) + pending < k:
                rnd.status = "REJECTED_QUORUM_UNREACHABLE"
                return {"status": "REJECTED",
                        "reason": (f"quorum {k} unreachable: {len(valid_ids)} valid, "
                                   f"{invalid} invalid of {len(need)} verifiers"),
                        "session_id": session_id}
            return None                            # still collecting

    def valid_provers(self, session_id: str) -> set:
        """The set of selected verifiers whose recorded proof is valid against
        our own key. Used by the sequencer to exclude non-responders / bad
        provers when retrying with a different verifier set (audit 4.3)."""
        with self._lock:
            rnd = self.rounds.get(session_id)
            if rnd is None or rnd.local_key is None:
                return set()
            key = rnd.local_key.fernet_key
            return {nid for nid in rnd.required_verifiers
                    if hmac.compare_digest(
                        derivation_proof(key, session_id, nid),
                        rnd.proofs.get(nid, ""))}

    def abandon(self, session_id: str):
        """Mark a round as abandoned so a late-arriving proof can never resolve
        it after the sequencer has moved on to a retry (prevents a stale attempt
        from appending a duplicate chain entry -- audit 4.3)."""
        with self._lock:
            rnd = self.rounds.get(session_id)
            if rnd is not None and rnd.status == "PENDING":
                rnd.status = "ABANDONED"

    def drop_key(self, session_id: str):
        """Forget the derived key once the round is fully handled (audit 3.2)."""
        with self._lock:
            rnd = self.rounds.get(session_id)
            if rnd is not None:
                rnd.local_key = None

    def status(self, session_id: str) -> Optional[str]:
        with self._lock:
            r = self.rounds.get(session_id)
            return r.status if r else None


# ==============================================================================
# 5. CIPHER (uses the live verified key)
# ==============================================================================

class JuvianCipher:
    def __init__(self, chain: SessionKeyChain):
        self.chain = chain

    def encrypt(self, obj: dict) -> str:
        key = self.chain.current_key()
        if key is None:
            raise RuntimeError("no verified key yet (3-of-3 incomplete)")
        return Fernet(key).encrypt(json.dumps(obj).encode()).decode()

    def decrypt(self, token: str) -> dict:
        key = self.chain.current_key()
        if key is None:
            raise RuntimeError("no verified key yet (3-of-3 incomplete)")
        return json.loads(Fernet(key).decrypt(token.encode()).decode())
