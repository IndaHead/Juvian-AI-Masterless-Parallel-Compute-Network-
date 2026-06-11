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
JUVIAN GRID :: RELIABLE CHUNKED DATAGRAM LAYER
==============================================================================
Closes audit findings 2.4 (UDP datagram size limit) and 2.5 (no reliability).

A single UDP datagram is only reliably ~1 MTU (~1472 B) and hard-capped at
65507 B; above an MTU it is IP-fragmented and frequently dropped. The tensor
messages base64-encode whole matrices, so on a real LAN they silently vanished.
This layer sits between the node's logical messages and the wire and provides:

  * FRAGMENTATION  -- a message is split into <=chunk_size datagrams
  * REASSEMBLY     -- the receiver rebuilds the message from its chunks
  * RELIABILITY    -- each chunk is ACKed; unacked chunks are retransmitted
                      (selective-repeat ARQ) until acknowledged or max_retries
  * DEDUP / REORDER tolerance -- chunks carry (msg_id, seq, total); duplicates
                      are dropped and a fully-received message is delivered once

It is transport-agnostic: you give it a synchronous `send_raw(addr, datagram)`
primitive and feed it inbound datagrams via `feed(addr, datagram)`; it calls
`deliver(addr, whole_message_bytes)` when a message is complete. `UDPTransport`
wires it over real sockets; `LossyDatagramBus` below lets the test-suite drive
it over a simulated lossy/reordering/duplicating network with no sockets.

Wire framing (binary, 21-byte header):
    MAGIC(4) | type(1) | msg_id(8) | seq(u32) | total(u32) | payload
    type: 0=DATA (reliable, ACKed)  1=ACK  2=DATA_NOACK (broadcast, best-effort)
==============================================================================
"""

import os
import time
import struct
import random
import asyncio
import collections
from typing import Callable, Dict, List, Optional, Tuple

MAGIC = b"JVR1"
T_DATA, T_ACK, T_NOACK = 0, 1, 2
_HDR = struct.Struct(">II")          # seq, total
_HEADER_LEN = len(MAGIC) + 1 + 8 + _HDR.size


def _frame(t: int, msg_id: bytes, seq: int, total: int, payload: bytes) -> bytes:
    return MAGIC + bytes([t]) + msg_id + _HDR.pack(seq, total) + payload


def _parse(dg: bytes) -> Optional[Tuple[int, bytes, int, int, bytes]]:
    if len(dg) < _HEADER_LEN or dg[:4] != MAGIC:
        return None
    t = dg[4]
    msg_id = dg[5:13]
    seq, total = _HDR.unpack(dg[13:_HEADER_LEN])
    return t, msg_id, seq, total, dg[_HEADER_LEN:]


class ReliableDatagram:
    """Selective-repeat ARQ + chunking over an unreliable datagram primitive."""

    def __init__(self, send_raw: Callable[[str, bytes], None],
                 deliver: Callable[[str, bytes], None],
                 *, chunk_size: int = 1100, rto: float = 0.25,
                 max_retries: int = 12, done_cache: int = 4096,
                 inbound_ttl: float = 10.0):
        self._send_raw = send_raw
        self._deliver = deliver
        self.chunk_size = chunk_size
        self.rto = rto
        self.max_retries = max_retries
        self.inbound_ttl = inbound_ttl
        # outbound (awaiting ACKs): msg_id -> state
        self._out: Dict[bytes, dict] = {}
        # inbound reassembly: (addr, msg_id) -> {total, chunks, t}
        self._in: Dict[Tuple[str, bytes], dict] = {}
        # recently completed messages, to drop duplicate late chunks
        self._done: "collections.deque" = collections.deque(maxlen=done_cache)
        self._done_set: set = set()
        self._loop_task: Optional[asyncio.Future] = None
        self._stopped = False

    # ---- lifecycle ----
    def start(self):
        if self._loop_task is None:
            self._loop_task = asyncio.ensure_future(self._timer_loop())

    async def stop(self):
        self._stopped = True
        if self._loop_task:
            self._loop_task.cancel()
            self._loop_task = None

    # ---- send ----
    def _chunk(self, data: bytes) -> Dict[int, bytes]:
        cs = self.chunk_size
        if not data:
            return {0: b""}
        n = (len(data) + cs - 1) // cs
        return {i: data[i * cs:(i + 1) * cs] for i in range(n)}

    async def send(self, addr: str, data: bytes, reliable: bool = True):
        """Send a whole message to `addr`, fragmenting as needed. Reliable sends
        are retransmitted until ACKed; best-effort sends (broadcast) are not."""
        msg_id = os.urandom(8)
        chunks = self._chunk(data)
        total = len(chunks)
        t = T_DATA if reliable else T_NOACK
        for seq, payload in chunks.items():
            self._send_raw(addr, _frame(t, msg_id, seq, total, payload))
        if reliable:
            self._out[msg_id] = {"addr": addr, "chunks": chunks, "total": total,
                                 "acked": set(), "retries": 0,
                                 "t": time.monotonic()}

    def frame_broadcast(self, data: bytes) -> List[bytes]:
        """Return best-effort (NOACK) datagrams for a broadcast message, so the
        caller can blast them over a broadcast socket. Large broadcasts are
        chunked; the receiver reassembles them the same way."""
        msg_id = os.urandom(8)
        chunks = self._chunk(data)
        total = len(chunks)
        return [_frame(T_NOACK, msg_id, seq, total, payload)
                for seq, payload in chunks.items()]

    # ---- receive ----
    def feed(self, addr: str, datagram: bytes):
        parsed = _parse(datagram)
        if parsed is None:
            return
        t, msg_id, seq, total, payload = parsed

        if t == T_ACK:
            o = self._out.get(msg_id)
            if o is not None:
                o["acked"].add(seq)
                if len(o["acked"]) >= o["total"]:
                    self._out.pop(msg_id, None)
            return

        # DATA or DATA_NOACK
        key = (addr, msg_id)
        if key in self._done_set:
            # already delivered; re-ACK so the sender stops retransmitting
            if t == T_DATA:
                self._send_raw(addr, _frame(T_ACK, msg_id, seq, 0, b""))
            return

        buf = self._in.get(key)
        if buf is None:
            buf = {"total": total, "chunks": {}, "t": time.monotonic()}
            self._in[key] = buf
        buf["chunks"][seq] = payload

        if t == T_DATA:                       # acknowledge every received chunk
            self._send_raw(addr, _frame(T_ACK, msg_id, seq, 0, b""))

        if len(buf["chunks"]) >= buf["total"]:
            try:
                data = b"".join(buf["chunks"][i] for i in range(buf["total"]))
            except KeyError:
                return                        # missing a chunk still; wait
            self._in.pop(key, None)
            self._mark_done(key)
            res = self._deliver(addr, data)
            if asyncio.iscoroutine(res):
                asyncio.ensure_future(res)

    def _mark_done(self, key):
        if len(self._done) == self._done.maxlen and self._done:
            self._done_set.discard(self._done[0])   # about to be evicted
        self._done.append(key)
        self._done_set.add(key)

    # ---- background timer: retransmit + inbound expiry ----
    async def _timer_loop(self):
        try:
            while not self._stopped:
                await asyncio.sleep(self.rto / 2)
                now = time.monotonic()
                # retransmit unacked outbound chunks
                for msg_id, o in list(self._out.items()):
                    if now - o["t"] < self.rto:
                        continue
                    if o["retries"] >= self.max_retries:
                        self._out.pop(msg_id, None)      # give up on this message
                        continue
                    o["retries"] += 1
                    o["t"] = now
                    for seq, payload in o["chunks"].items():
                        if seq in o["acked"]:
                            continue
                        self._send_raw(o["addr"],
                                       _frame(T_DATA, msg_id, seq, o["total"], payload))
                # expire stale partial inbound messages (lost broadcast chunks)
                for key, buf in list(self._in.items()):
                    if now - buf["t"] > self.inbound_ttl:
                        self._in.pop(key, None)
        except asyncio.CancelledError:
            pass


# ==============================================================================
# TEST HARNESS: a simulated lossy/reordering/duplicating datagram network
# ==============================================================================

class LossyDatagramBus:
    """In-process stand-in for a UDP network, with tunable loss / duplication /
    reordering, so the reliability layer can be tested without sockets."""

    def __init__(self, loss: float = 0.0, dup: float = 0.0,
                 reorder: float = 0.0, seed: int = 1234):
        self.endpoints: Dict[str, Callable[[str, bytes], None]] = {}
        self.rng = random.Random(seed)
        self.loss, self.dup, self.reorder = loss, dup, reorder
        self.sent = 0
        self.dropped = 0

    def register(self, addr: str, feed: Callable[[str, bytes], None]):
        self.endpoints[addr] = feed

    def send_raw_for(self, src: str) -> Callable[[str, bytes], None]:
        def _send(dst: str, data: bytes):
            self.sent += 1
            if self.rng.random() < self.loss:
                self.dropped += 1
                return
            copies = 2 if self.rng.random() < self.dup else 1
            delay = (self.rng.random() * 0.005) if self.rng.random() < self.reorder else 0.0
            for _ in range(copies):
                asyncio.ensure_future(self._deliver(src, dst, data, delay))
        return _send

    async def _deliver(self, src: str, dst: str, data: bytes, delay: float):
        if delay:
            await asyncio.sleep(delay)
        fb = self.endpoints.get(dst)
        if fb:
            fb(src, data)


class ReliableInMemoryTransport:
    """A BaseTransport-compatible transport that runs the real node stack over a
    LossyDatagramBus through ReliableDatagram -- lets tests exercise the whole
    sign -> chunk -> lossy-network -> reassemble -> verify path."""

    def __init__(self, addr: str, bus: LossyDatagramBus, **rdt_kwargs):
        self.addr = addr
        self.bus = bus
        self._handler = None
        self.rdt = ReliableDatagram(bus.send_raw_for(addr), self._deliver,
                                    **rdt_kwargs)

    def set_handler(self, handler):
        self._handler = handler

    async def start(self):
        self.bus.register(self.addr, self.rdt.feed)
        self.rdt.start()

    def _deliver(self, src_addr: str, data: bytes):
        import json
        if self._handler is None:
            return
        try:
            msg = json.loads(data.decode())
        except Exception:
            return
        return self._handler(msg, src_addr)

    async def send(self, addr: str, message: dict):
        import json
        await self.rdt.send(addr, json.dumps(message).encode(), reliable=True)

    async def broadcast(self, message: dict):
        import json
        data = json.dumps(message).encode()
        for dst in list(self.bus.endpoints.keys()):
            if dst != self.addr:
                # best-effort, like a real UDP broadcast, but chunked
                await self.rdt.send(dst, data, reliable=False)

    async def stop(self):
        await self.rdt.stop()
