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
JUVIAN GRID :: KEY-AGREEMENT LAYER  (the "ECDH layer")
==============================================================================

WHY THIS EXISTS
---------------
The pi-Mandelbrot 3-of-3 scheme proves that independent devices agree on a key
and transmits no key material -- but on its own it is *integrity*, not
*confidentiality*. Every key is derived from (payload || salt); if an
eavesdropper knows the payload and the salt chain, they can derive the key.

This layer makes the GENESIS SALT a real shared secret that an eavesdropper
cannot derive. Once the genesis salt is secret, every chained pi-Mandelbrot key
depends on a secret the eavesdropper lacks -> the whole chain gains
confidentiality against outsiders.

WHY A *GROUP* PROTOCOL (not plain pairwise ECDH)
------------------------------------------------
Strict 3-of-3 picks ANY three members (by XOR proximity) per request, and they
must all derive the SAME key, hence the SAME salt. A pairwise ECDH secret is
shared by only two parties. Naively combining pairwise secrets fails too: a
member cannot compute the secret shared between two *other* members. The correct
tool is a CONTRIBUTORY group key agreement where every member computes one
identical group secret. We use Burmester-Desmedt (BD):

  * masterless -- no coordinator chooses the secret; it emerges from all r_i
  * only public DH values (z_i, X_i) ever travel the wire -- no secret is sent
  * every member computes the identical group key K

K is hashed to 32 bytes and becomes the genesis salt fed into SessionKeyChain.

TWO PRIMITIVES PROVIDED
-----------------------
  1. GroupKeyAgreement  -- Burmester-Desmedt contributory group key (the one
     wired into the node for the genesis salt). Implemented over a standard
     2048-bit MODP safe-prime group (RFC 3526 Group 14) using exact big-int
     arithmetic -- sound, dependency-free, and fully testable. The same
     construction ports directly to an elliptic curve.
  2. PairwiseECDH       -- genuine elliptic-curve DH (X25519) for the simple
     two-party confidential link (e.g. a direct input<->output channel).

BD KEY MATH
-----------
Group G of prime order q, generator g, N members in a cycle (indices mod N):
  round 1:  z_i = g^{r_i}
  round 2:  X_i = (z_{i+1} * z_{i-1}^{-1})^{r_i}
  key:      K_i = z_{i-1}^{N*r_i} * X_i^{N-1} * X_{i+1}^{N-2} * ... * X_{i+N-2}^1
All members obtain K = g^{r_0 r_1 + r_1 r_2 + ... + r_{N-1} r_0}.
==============================================================================
"""

import os
import hashlib
import secrets
from typing import Dict, List, Optional

# --- pairwise ECDH (X25519) dependencies -------------------------------------
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey, X25519PublicKey,
)
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


# ==============================================================================
# MODP group -- RFC 3526 Group 14 (2048-bit safe prime), generator g = 2
# ==============================================================================

_P_HEX = (
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
    "29024E088A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
    "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D"
    "C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F"
    "83655D23DCA3AD961C62F356208552BB9ED529077096966D"
    "670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
    "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9"
    "DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
    "15728E5A8AACAA68FFFFFFFFFFFFFFFF"
)
P = int(_P_HEX, 16)
G = 2
Q = (P - 1) // 2   # order of the prime-order (quadratic-residue) subgroup


def _valid_group_element(z: int) -> bool:
    """Reject elements outside the prime-order subgroup (audit 1.2): blocks 0,
    1, p-1, and small-order elements that would allow key confinement or a
    division-by-zero crash."""
    if z <= 1 or z >= P - 1:
        return False
    # must lie in the order-Q subgroup: z^Q == 1 (mod P)
    return pow(z, Q, P) == 1


def _rand_exponent() -> int:
    """Random secret exponent r in [2, Q-1]."""
    return 2 + secrets.randbelow(Q - 3)


# ==============================================================================
# 1. BURMESTER-DESMEDT CONTRIBUTORY GROUP KEY AGREEMENT
# ==============================================================================

class GroupKeyAgreement:
    """One member's view of a Burmester-Desmedt group key agreement.

    Lifecycle:
        bd = GroupKeyAgreement(my_id, roster)
        send  bd.round1_public()            # z_i  -> broadcast
        for each peer: bd.set_round1(pid, z_peer)
        when bd.ready_for_round2():
            send bd.round2_public()         # X_i  -> broadcast
        for each peer: bd.set_round2(pid, X_peer)
        when bd.ready_for_key():
            salt = bd.group_salt()          # 32-byte genesis salt
    """

    def __init__(self, member_id: str, roster: List[str]):
        if member_id not in roster:
            raise ValueError("member_id must be in roster")
        # deterministic cyclic ordering all members agree on
        self.roster = sorted(set(roster))
        self.n = len(self.roster)
        self.member_id = member_id
        self.index = self.roster.index(member_id)

        self._r = _rand_exponent()
        self._z_self = pow(G, self._r, P)
        self.z: Dict[int, int] = {self.index: self._z_self}   # index -> z
        self.x: Dict[int, int] = {}                            # index -> X
        self._key_int: Optional[int] = None

    # -- helpers ----------------------------------------------------------
    def _idx_of(self, member_id: str) -> Optional[int]:
        try:
            return self.roster.index(member_id)
        except ValueError:
            return None

    @property
    def left(self) -> int:
        return (self.index - 1) % self.n

    @property
    def right(self) -> int:
        return (self.index + 1) % self.n

    # -- round 1 ----------------------------------------------------------
    def round1_public(self) -> str:
        """This member's z_i as hex (broadcast it)."""
        return format(self._z_self, "x")

    def set_round1(self, member_id: str, z_hex: str) -> bool:
        idx = self._idx_of(member_id)
        if idx is None:
            return False
        try:
            z = int(z_hex, 16)
        except (ValueError, TypeError):
            return False
        if not _valid_group_element(z):
            return False          # audit 1.2: reject out-of-subgroup elements
        self.z[idx] = z
        return True

    def ready_for_round2(self) -> bool:
        # need both neighbours' z to form X_i; for safety require all z
        return len(self.z) >= self.n

    # -- round 2 ----------------------------------------------------------
    def round2_public(self) -> str:
        """This member's X_i = (z_{i+1} * z_{i-1}^{-1})^{r_i}, as hex."""
        if self.left not in self.z or self.right not in self.z:
            raise RuntimeError("missing neighbour z values for round 2")
        z_left = self.z[self.left]
        z_right = self.z[self.right]
        inv_left = pow(z_left, -1, P)
        x_self = pow((z_right * inv_left) % P, self._r, P)
        self.x[self.index] = x_self
        return format(x_self, "x")

    def set_round2(self, member_id: str, x_hex: str) -> bool:
        idx = self._idx_of(member_id)
        if idx is None:
            return False
        try:
            x = int(x_hex, 16)
        except (ValueError, TypeError):
            return False
        if not _valid_group_element(x):
            return False          # audit 1.2
        self.x[idx] = x
        return True

    def ready_for_key(self) -> bool:
        return len(self.x) >= self.n and self.index in self.x

    # -- key derivation ---------------------------------------------------
    def _compute_key_int(self) -> int:
        n = self.n
        # term0 = z_{i-1}^{N * r_i}
        key = pow(self.z[self.left], (n * self._r) % Q, P)
        # product term: X_{i+j}^{N-1-j} for j = 0..N-2, exponents N-1..1
        for j in range(0, n - 1):
            idx = (self.index + j) % n
            exp = (n - 1 - j)
            key = (key * pow(self.x[idx], exp, P)) % P
        return key

    def group_key_int(self) -> int:
        if self._key_int is None:
            self._key_int = self._compute_key_int()
        return self._key_int

    def group_salt(self) -> bytes:
        """SHA-256 of the shared group key -> 32-byte genesis salt."""
        k = self.group_key_int()
        k_bytes = k.to_bytes((k.bit_length() + 7) // 8, "big")
        return hashlib.sha256(b"JUVIAN_BD_GENESIS::" + k_bytes).digest()


# ==============================================================================
# 2. PAIRWISE ELLIPTIC-CURVE DH  (X25519) -- two-party confidential link
# ==============================================================================

class PairwiseECDH:
    """Genuine elliptic-curve Diffie-Hellman over Curve25519 for a direct
    two-party channel. Public keys cross the wire; the shared secret never
    does. Result is run through HKDF-SHA256 to a 32-byte key."""

    def __init__(self):
        self._priv = X25519PrivateKey.generate()

    def public_bytes(self) -> bytes:
        return self._priv.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def public_hex(self) -> str:
        return self.public_bytes().hex()

    def derive(self, peer_public: bytes, info: bytes = b"juvian-ecdh") -> bytes:
        peer = X25519PublicKey.from_public_bytes(peer_public)
        shared = self._priv.exchange(peer)
        return HKDF(
            algorithm=hashes.SHA256(), length=32,
            salt=None, info=info,
        ).derive(shared)

    def derive_from_hex(self, peer_public_hex: str,
                        info: bytes = b"juvian-ecdh") -> bytes:
        return self.derive(bytes.fromhex(peer_public_hex), info)
