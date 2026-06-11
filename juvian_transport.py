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
JUVIAN GRID :: TRANSPORT LAYER
Pluggable message transport so the same node code runs over real UDP in
deployment and over an in-memory bus in tests / single-machine demos.

Message framing: every message is a JSON object with at least:
    {"type": <str>, "from": <node_id>, "from_addr": <str>, ...}
==============================================================================
"""

import json
import asyncio
import socket
from typing import Callable, Dict, Optional, Awaitable

from juvian_reliable import ReliableDatagram


Handler = Callable[[dict, str], Awaitable[None]]   # (message, sender_addr)


class BaseTransport:
    def __init__(self):
        self._handler: Optional[Handler] = None

    def set_handler(self, handler: Handler):
        self._handler = handler

    async def start(self):
        raise NotImplementedError

    async def send(self, addr: str, message: dict):
        raise NotImplementedError

    async def broadcast(self, message: dict):
        raise NotImplementedError

    async def stop(self):
        pass


# ------------------------------------------------------------------ UDP -------

class _UDPProtocol(asyncio.DatagramProtocol):
    def __init__(self, transport_ref, is_unicast: bool = False):
        self.ref = transport_ref
        self.is_unicast = is_unicast

    def connection_made(self, transport):
        # only the unicast endpoint is used for sending; the broadcast listener
        # must not clobber it (this was a latent bug before the rewrite)
        if self.is_unicast:
            self.ref._udp = transport

    def datagram_received(self, data, addr):
        sender = f"{addr[0]}:{addr[1]}"
        rdt = self.ref.rdt
        if rdt is not None:
            rdt.feed(sender, data)        # all inbound goes through the ARQ layer


class UDPTransport(BaseTransport):
    """Real UDP transport with a reliable, chunked layer on top (audit 2.4/2.5).

    Unicast sends are fragmented to MTU-sized datagrams and retransmitted until
    acknowledged, so large tensor messages survive a real network instead of
    being dropped as oversized/fragmented datagrams. Broadcasts (discovery, KEX,
    verify-open/result) are chunked but best-effort, matching UDP broadcast
    semantics."""

    MAX_INFLIGHT = 1024     # cap concurrent handler tasks (packet-flood DoS)

    def __init__(self, host: str, port: int, broadcast_port: int = 8001):
        super().__init__()
        self.host = host
        self.port = port
        self.broadcast_port = broadcast_port
        self._udp = None
        self._bcast_sock = None
        self.rdt: Optional[ReliableDatagram] = None
        self._inflight = 0

    async def start(self):
        loop = asyncio.get_running_loop()
        self.rdt = ReliableDatagram(self._send_raw, self._deliver)
        self.rdt.start()

        await loop.create_datagram_endpoint(
            lambda: _UDPProtocol(self, is_unicast=True),
            local_addr=(self.host, self.port),
            allow_broadcast=True,
        )
        # separate raw socket for sending broadcasts
        self._bcast_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._bcast_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._bcast_sock.setblocking(False)
        # listener for inbound broadcasts -- reuse_port lets several nodes on
        # one host share the discovery port (audit 2.7)
        try:
            await loop.create_datagram_endpoint(
                lambda: _UDPProtocol(self, is_unicast=False),
                local_addr=("", self.broadcast_port),
                allow_broadcast=True,
                reuse_port=True,
            )
        except (TypeError, OSError):
            await loop.create_datagram_endpoint(
                lambda: _UDPProtocol(self, is_unicast=False),
                local_addr=("", self.broadcast_port),
                allow_broadcast=True,
            )

    # -- raw datagram primitive used by the reliability layer --
    def _send_raw(self, addr: str, datagram: bytes):
        if not self._udp:
            return
        host, _, port = addr.rpartition(":")     # IPv6-safe split (audit 2.8)
        if not host or not port.isdigit():
            return
        try:
            self._udp.sendto(datagram, (host, int(port)))
        except OSError:
            pass

    # -- a fully-reassembled message is ready: dispatch to the handler --
    def _deliver(self, sender: str, data: bytes):
        if self._inflight >= self.MAX_INFLIGHT or not self._handler:
            return                                # shed load under flood
        try:
            msg = json.loads(data.decode())
        except Exception:
            return
        self._inflight += 1
        task = asyncio.create_task(self._handler(msg, sender))
        task.add_done_callback(lambda _t: setattr(self, "_inflight",
                                                  self._inflight - 1))

    async def send(self, addr: str, message: dict):
        if self.rdt is None:
            return
        await self.rdt.send(addr, json.dumps(message).encode(), reliable=True)

    async def broadcast(self, message: dict):
        if not self._bcast_sock or self.rdt is None:
            return
        for dg in self.rdt.frame_broadcast(json.dumps(message).encode()):
            try:
                self._bcast_sock.sendto(
                    dg, ("255.255.255.255", self.broadcast_port))
            except OSError:
                pass

    async def stop(self):
        if self.rdt is not None:
            await self.rdt.stop()


# ------------------------------------------------------------- in-memory ------

class InMemoryBus:
    """Shared switchboard for in-process nodes. Lets the full protocol run with
    zero sockets -- used by the local demo and the test-suite."""

    def __init__(self):
        self.endpoints: Dict[str, "InMemoryTransport"] = {}

    def register(self, addr: str, transport: "InMemoryTransport"):
        self.endpoints[addr] = transport

    async def deliver(self, to_addr: str, from_addr: str, message: dict):
        ep = self.endpoints.get(to_addr)
        if ep and ep._handler:
            await asyncio.sleep(0)  # yield, simulate async hop
            await ep._handler(message, from_addr)

    async def deliver_broadcast(self, from_addr: str, message: dict):
        for addr, ep in list(self.endpoints.items()):
            if addr != from_addr and ep._handler:
                await asyncio.sleep(0)
                await ep._handler(message, from_addr)


class InMemoryTransport(BaseTransport):
    def __init__(self, addr: str, bus: InMemoryBus):
        super().__init__()
        self.addr = addr
        self.bus = bus

    async def start(self):
        self.bus.register(self.addr, self)

    async def send(self, addr: str, message: dict):
        await self.bus.deliver(addr, self.addr, message)

    async def broadcast(self, message: dict):
        await self.bus.deliver_broadcast(self.addr, message)
