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
JUVIAN GRID :: IDENTITY & MESSAGE AUTHENTICATION
==============================================================================

Closes audit finding 2.1 (no message authentication) and the spoofing/forgery
family that flows from it.

Design: SELF-CERTIFYING node identifiers (the libp2p / content-address idea).
A node's id IS a hash of its Ed25519 public key:

    node_id = SHA-256(public_key_raw)[:40]      # 160-bit, Kademlia-compatible

Every message is wrapped in a signed envelope:

    { "pub": <hex pubkey>, "sig": <hex signature>, "body": { ... } }

On receipt a node checks, before doing anything else:
    1. the signature verifies under `pub`, and
    2. body["from"] == node_id_for(pub)

So a sender cannot claim an id it does not hold the private key for, and cannot
tamper with a signed body. No PKI, no key directory needed for verification --
the id self-certifies the key, and the key verifies the message.

Identity *creation* is priced by a proof-of-work birth certificate (see
`mint_pow` / `verify_pow` below): to be admitted, a node must present a nonce
whose hash over its public key clears a difficulty target. This raises the cost
of minting many Sybil identities (the damaging case being a Sybil majority that
overwhelms the tensor-aggregation outlier filter) and of grinding a privileged
id, without a coordinator. It does NOT make Sybils impossible -- a well-resourced
attacker can still pay the work, and the difficulty must stay low enough to be
feasible on weak edge devices -- so for closed device fleets an allowlist /
vouching admission policy is the stronger complement.
==============================================================================
"""

import os
import json
import time
import hashlib
from typing import Optional, Tuple

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature


def node_id_for(public_raw: bytes) -> str:
    """Self-certifying 160-bit node id derived from a raw Ed25519 public key."""
    return hashlib.sha256(public_raw).hexdigest()[:40]


def canonical(body: dict) -> bytes:
    """Deterministic serialization of a message body for signing/verifying."""
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


# ----------------------------------------------------------------------------
# Proof-of-work birth certificate (audit 2.3 / 4.4: Sybil pricing)
# ----------------------------------------------------------------------------
# Identity creation was free: anyone could mint unlimited keypairs and flood the
# mesh (the damaging case is drowning the tensor-aggregation MAD/median filter
# with a Sybil majority). We can't *prevent* Sybils without an external scarce
# resource, but we can *price* them: to be admitted, a node must exhibit a nonce
# whose hash over its public key has at least `difficulty` leading zero bits.
# This costs ~2**difficulty hashes per identity, is a one-time cost the node pays
# at birth, and is verifiable by every peer independently with a single hash --
# no coordinator, no directory. The cert is bound to the public key, so it can't
# be transplanted onto another identity (that identity's signature wouldn't
# verify). node_id is still SHA-256(pubkey), so routing/XOR/sequencer logic is
# unchanged. See README/REMEDIATION_STATUS for the honest limits (PoW does not
# stop a well-resourced attacker, and is awkward on weak edge devices).
POW_DOMAIN = b"JUVIAN_POW_V1::"


def _leading_zero_bits(digest: bytes) -> int:
    """Number of leading zero BITS in a byte string."""
    bits = 0
    for byte in digest:
        if byte == 0:
            bits += 8
            continue
        bits += 8 - byte.bit_length()
        break
    return bits


def pow_stamp(public_raw: bytes, nonce: bytes) -> bytes:
    return hashlib.sha256(POW_DOMAIN + public_raw + nonce).digest()


def verify_pow(public_raw: bytes, nonce_hex: str, difficulty: int) -> bool:
    """True iff `nonce_hex` is a valid birth certificate for `public_raw` at the
    given difficulty. difficulty <= 0 disables the check (returns True)."""
    if difficulty <= 0:
        return True
    if not isinstance(nonce_hex, str) or not nonce_hex:
        return False
    try:
        nonce = bytes.fromhex(nonce_hex)
    except ValueError:
        return False
    return _leading_zero_bits(pow_stamp(public_raw, nonce)) >= difficulty


def mint_pow(public_raw: bytes, difficulty: int,
             max_attempts: int = 1 << 32) -> Tuple[str, int]:
    """Search for a nonce whose stamp has >= difficulty leading zero bits.
    Returns (nonce_hex, attempts); `attempts` lets callers reason about the cost
    (expected ~2**difficulty). Raises if no nonce is found within max_attempts."""
    if difficulty <= 0:
        return "", 0
    counter = 0
    while counter < max_attempts:
        nonce = counter.to_bytes(8, "big")
        counter += 1
        if _leading_zero_bits(pow_stamp(public_raw, nonce)) >= difficulty:
            return nonce.hex(), counter
    raise RuntimeError(
        f"could not mint PoW at difficulty {difficulty} in {max_attempts} tries")


# ----------------------------------------------------------------------------
# Admission vouchers (allowlist / vouching policy: Sybil BOUNDING, not pricing)
# ----------------------------------------------------------------------------
# Proof-of-work *prices* identities but does not *bound* them: a well-resourced
# attacker pays the work and mints as many as it likes, and a Sybil majority
# still breaks every honest-majority quorum the system relies on (verifier
# quorum, attestation quorum, the tensor outlier filter's Byzantine floor). An
# admission policy bounds the population instead of pricing it:
#
#   * allowlist  -- a node admits only ids on a configured list (closed fleet:
#                   airtight, but needs out-of-band provisioning);
#   * vouching   -- an ALREADY-ADMITTED member signs a voucher for a newcomer;
#                   a newcomer is admitted once it presents >= threshold vouchers
#                   from DISTINCT admitted members (founders seed the root of
#                   trust). Minting N Sybils now costs N*threshold vouches from
#                   genuine members, i.e. a social/real cost, not just compute.
#
# A voucher is a self-certifying object: the issuer signs over the SUBJECT's
# node id (plus a domain tag and issue time), and carries the issuer's pubkey.
# Anyone verifies it with one signature check -- no directory. The subject's own
# signed envelope proves it holds the key for the vouched id, so a voucher can
# never be replayed onto a different identity. Honest limit (documented, not
# hidden): vouching is transitive -- a vouched member can itself vouch -- so a
# coalition of `threshold` careless-or-malicious members can still admit Sybils;
# the allowlist is the airtight control, vouching is bounded-but-not-airtight.
VOUCH_DOMAIN = b"JUVIAN_VOUCH_V1::"


def voucher_message(subject_id: str, issued_at: float) -> bytes:
    """Canonical bytes an issuer signs to vouch for `subject_id`. The issue time
    is bound in so a deployment may later choose to expire vouchers; it is part
    of the signed object, so it cannot be altered."""
    return VOUCH_DOMAIN + canonical({"sub": subject_id, "iat": round(issued_at, 3)})


def make_voucher(issuer: "Identity", subject_id: str) -> dict:
    """Issue a signed voucher: `issuer` attests that `subject_id` should be
    admitted. Returns a self-verifying dict {iss, sub, iat, pub, sig}."""
    iat = time.time()
    sig = issuer.sign(voucher_message(subject_id, iat))
    return {"iss": issuer.node_id, "sub": subject_id, "iat": round(iat, 3),
            "pub": issuer.public_hex(), "sig": sig.hex()}


def verify_voucher(voucher: dict, subject_id: str) -> Optional[str]:
    """Validate a voucher FOR `subject_id`. Returns the issuer's node id if the
    signature verifies under the embedded pubkey, that pubkey hashes to the
    claimed issuer id (self-certifying), and the voucher's subject matches the
    id being admitted. Returns None on any failure. Does NOT decide whether the
    issuer is itself admitted -- that is the node's policy (see JuvianNode)."""
    if not isinstance(voucher, dict):
        return None
    iss = voucher.get("iss")
    sub = voucher.get("sub")
    pub_hex = voucher.get("pub")
    sig_hex = voucher.get("sig")
    iat = voucher.get("iat")
    if not (isinstance(iss, str) and isinstance(sub, str)
            and isinstance(pub_hex, str) and isinstance(sig_hex, str)
            and isinstance(iat, (int, float))):
        return None
    if sub != subject_id:
        return None                      # voucher is for a different identity
    try:
        pub_raw = bytes.fromhex(pub_hex)
        sig = bytes.fromhex(sig_hex)
        pub = Ed25519PublicKey.from_public_bytes(pub_raw)
        pub.verify(sig, voucher_message(sub, iat))
    except (ValueError, InvalidSignature):
        return None
    if node_id_for(pub_raw) != iss:      # issuer id must own the signing key
        return None
    return iss


class Identity:
    """A node's signing identity. Persistable so a node keeps its id across
    restarts."""

    def __init__(self, private_key: Ed25519PrivateKey,
                 pow_nonce: str = "", pow_difficulty: int = 0):
        self._priv = private_key
        self._pub_raw = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.node_id = node_id_for(self._pub_raw)
        # proof-of-work birth certificate (empty when difficulty 0 / disabled)
        self.pow_nonce = pow_nonce
        self.pow_difficulty = pow_difficulty

    # ---- construction ----
    @classmethod
    def generate(cls, pow_difficulty: int = 0) -> "Identity":
        priv = Ed25519PrivateKey.generate()
        nonce = ""
        if pow_difficulty > 0:
            pub_raw = priv.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            nonce, _ = mint_pow(pub_raw, pow_difficulty)
        return cls(priv, pow_nonce=nonce, pow_difficulty=pow_difficulty)

    @classmethod
    def load_or_create(cls, path: str, pow_difficulty: int = 0) -> "Identity":
        """Load a 32-byte raw private key from `path`, or create+persist one. The
        PoW birth certificate is cached alongside it in `<path>.pow` so a node
        does not re-mint on every restart (and is re-minted if the required
        difficulty has since increased)."""
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                raw = f.read()
            priv = Ed25519PrivateKey.from_private_bytes(raw)
        else:
            priv = Ed25519PrivateKey.generate()
            if path:
                raw = priv.private_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PrivateFormat.Raw,
                    encryption_algorithm=serialization.NoEncryption(),
                )
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "wb") as f:
                    f.write(raw)

        pub_raw = priv.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        nonce = ""
        if pow_difficulty > 0:
            powpath = (path + ".pow") if path else None
            if powpath and os.path.exists(powpath):
                with open(powpath, "r") as f:
                    cand = f.read().strip()
                if verify_pow(pub_raw, cand, pow_difficulty):
                    nonce = cand
            if not nonce:
                nonce, _ = mint_pow(pub_raw, pow_difficulty)
                if powpath:
                    with open(powpath, "w") as f:
                        f.write(nonce)
        return cls(priv, pow_nonce=nonce, pow_difficulty=pow_difficulty)

    # ---- primitives ----
    def public_hex(self) -> str:
        return self._pub_raw.hex()

    def sign(self, data: bytes) -> bytes:
        return self._priv.sign(data)

    # ---- envelope ----
    def wrap(self, body: dict) -> dict:
        """Produce a signed envelope around a message body. A per-message nonce
        (`_n`) and timestamp (`_t`) are folded into the signed body so receivers
        can reject stale messages (freshness) and duplicates (replay). Both are
        covered by the signature, so neither can be altered (audit: replay)."""
        enriched = dict(body)
        enriched["_n"] = os.urandom(12).hex()
        enriched["_t"] = round(time.time(), 3)
        sig = self._priv.sign(canonical(enriched))
        return {"pub": self._pub_raw.hex(), "sig": sig.hex(),
                "pow": self.pow_nonce, "body": enriched}


# how far out of date a signed message may be before it is rejected outright.
# Generous to tolerate clock skew across devices; the per-node seen-nonce cache
# handles duplicate suppression within this window.
REPLAY_MAX_AGE_S = 90.0


def verify_envelope(envelope: dict,
                    max_age_s: float = REPLAY_MAX_AGE_S,
                    pow_difficulty: int = 0) -> Optional[dict]:
    """Verify a signed envelope. Returns the body if (a) the signature verifies
    under the embedded public key, (b) body['from'] matches the id derived from
    that key, (c) the sender presents a valid proof-of-work birth certificate at
    >= pow_difficulty (Sybil pricing; 0 disables), and (d) the timestamp is fresh
    (within max_age_s). Returns None on any failure (caller drops the message).
    Duplicate suppression within the freshness window is the caller's
    responsibility (see JuvianNode seen-cache)."""
    if not isinstance(envelope, dict):
        return None
    pub_hex = envelope.get("pub")
    sig_hex = envelope.get("sig")
    body = envelope.get("body")
    if not (isinstance(pub_hex, str) and isinstance(sig_hex, str)
            and isinstance(body, dict)):
        return None
    try:
        pub_raw = bytes.fromhex(pub_hex)
        sig = bytes.fromhex(sig_hex)
        pub = Ed25519PublicKey.from_public_bytes(pub_raw)
        pub.verify(sig, canonical(body))           # raises on bad signature
    except (ValueError, InvalidSignature):
        return None
    # the claimed sender must own this key
    if body.get("from") != node_id_for(pub_raw):
        return None
    # Sybil pricing: the sender must present a valid proof-of-work cert for this
    # key (audit 2.3 / 4.4). The cert is self-verifying against the key, so it
    # need not be inside the signed body.
    if not verify_pow(pub_raw, envelope.get("pow", ""), pow_difficulty):
        return None
    # freshness: reject stale or implausibly-future messages (replay defence)
    ts = body.get("_t")
    if not isinstance(ts, (int, float)) or abs(time.time() - ts) > max_age_s:
        return None
    return body
