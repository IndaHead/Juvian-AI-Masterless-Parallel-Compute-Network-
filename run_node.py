#!/usr/bin/env python3
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
JUVIAN GRID :: NODE ENTRY POINT  (real UDP, LAN deployment)
==============================================================================

Run one Juvian node on a real machine. Nodes on the same LAN discover each
other automatically via UDP broadcast beacons, then form a strict 3-of-3
pi-Mandelbrot key-verification mesh and can run distributed tensor jobs.

USAGE
-----
  # terminal 1
  python3 run_node.py --name anchor-1 --port 9101 --type ANCHOR

  # terminal 2
  python3 run_node.py --name mobile-1 --port 9102 --type MOBILE

  # terminal 3
  python3 run_node.py --name mobile-2 --port 9103 --type MOBILE

All three must share the SAME --session-secret (default shown below) or their
genesis salts won't match and verification will reject. Once at least 3 nodes
are up, type commands at the prompt:

  kex            run group key agreement -> install a confidential genesis salt
                 (do this first for confidentiality against eavesdroppers)
  req <text>     submit a compute-request; runs 3-of-3 key verification
  tensor [N]     run a distributed tensor reduce over an NxN matrix (default 8)
  peers          list discovered peers
  chain          show this node's key-chain depth + latest fingerprint
  snap           print full JSON snapshot
  web            print the dashboard URL (if --web was passed)
  quit           persist chain and exit

NOTE ON SCALE
-------------
This is a real LAN mesh of real processes. It is NOT the million-node planetary
substrate -- that figure comes from the in-browser simulation. Here, every node
you see is an actual OS process exchanging real UDP datagrams.
"""

import sys
import asyncio
import argparse

import numpy as np

from juvian_transport import UDPTransport
from juvian_node import JuvianNode
from juvian_crypto import DEFAULT_ITER


def parse_args():
    p = argparse.ArgumentParser(
        description="Run a Juvian Grid node (real UDP LAN mesh).")
    p.add_argument("--name", required=True,
                   help="human-readable node name, e.g. anchor-1")
    p.add_argument("--host", default="0.0.0.0",
                   help="bind host for the unicast socket (default 0.0.0.0)")
    p.add_argument("--port", type=int, required=True,
                   help="unicast UDP port for this node, e.g. 9101")
    p.add_argument("--broadcast-port", type=int, default=8001,
                   help="shared LAN discovery broadcast port (default 8001)")
    p.add_argument("--advertise-host", default=None,
                   help="address other nodes should reply to "
                        "(defaults to 127.0.0.1 for same-machine testing; "
                        "set to this machine's LAN IP for multi-machine)")
    p.add_argument("--type", default="ANCHOR",
                   choices=["ANCHOR", "MOBILE", "DESKTOP"],
                   help="device class; affects hardware weight")
    p.add_argument("--session-secret", default="juvian-default-session",
                   help="shared session secret -- MUST match across all nodes")
    p.add_argument("--iter", type=int, default=DEFAULT_ITER,
                   help=f"Mandelbrot iteration depth (default {DEFAULT_ITER})")
    p.add_argument("--web", type=int, default=0, metavar="PORT",
                   help="if set, serve a live web dashboard on this port")
    return p.parse_args()


def weight_for_type(device_type: str) -> float:
    return {"ANCHOR": 5.0, "DESKTOP": 4.0, "MOBILE": 2.0}.get(device_type, 2.0)


async def repl(node: JuvianNode, web_port: int):
    """Minimal async command prompt driving the node."""
    loop = asyncio.get_running_loop()
    print()
    print(f"  node {node.name} ready  |  id={node.node_id[:16]}  "
          f"|  type={node.device_type}")
    print(f"  listening udp {node.address}  |  discovery broadcast on "
          f"{node.transport.broadcast_port}")
    if web_port:
        print(f"  dashboard:  http://127.0.0.1:{web_port}")
    print("  commands: kex | req <text> | tensor [N] | peers | chain | snap | quit")
    print()

    while True:
        try:
            line = await loop.run_in_executor(None, sys.stdin.readline)
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        line = line.strip()
        if not line:
            continue

        cmd, _, arg = line.partition(" ")
        cmd = cmd.lower()

        if cmd in ("quit", "exit", "q"):
            break

        elif cmd == "kex":
            print("  -> running group key agreement (Burmester-Desmedt)...")
            res = await node.establish_session(timeout=10.0)
            if res.get("status") == "ESTABLISHED":
                print(f"  <- ESTABLISHED  members={res['members']}  "
                      f"genesis_fp={res['genesis_fingerprint']}  "
                      f"(source now GROUP_DH)")
            else:
                print(f"  <- {res.get('status')}  ({res.get('reason', '')})")

        elif cmd == "req":
            payload = (arg or "ping").encode()
            print(f"  -> submitting request ({len(payload)} bytes)...")
            res = await node.submit_request(payload, timeout=6.0)
            status = res.get("status")
            fp = (res.get("fingerprint") or "")[:16]
            depth = res.get("chain_depth")
            vs = [v[:8] for v in res.get("verifiers", [])]
            if status == "VERIFIED":
                print(f"  <- VERIFIED  fp={fp}  chain_depth={depth}  "
                      f"verifiers={vs}")
            else:
                print(f"  <- {status}  ({res.get('reason', '')})")

        elif cmd == "tensor":
            n = int(arg) if arg.strip().isdigit() else 8
            tensor = np.random.default_rng().standard_normal((n, n))
            print(f"  -> running distributed tensor reduce over {n}x{n}...")
            res = await node.run_tensor_job(tensor, rank=min(4, n), timeout=6.0)
            print(f"  <- {res.get('status')}  energy="
                  f"{res.get('frobenius_energy', 0):.2f}  "
                  f"valid={len(res.get('valid_nodes', []))}  "
                  f"purged={len(res.get('purged_nodes', []))}")

        elif cmd == "peers":
            peers = node.routing.all_peers()
            if not peers:
                print("  (no peers discovered yet -- start more nodes)")
            for pid, info in peers.items():
                print(f"  {pid[:16]}  {info.get('name','?'):12s} "
                      f"{info.get('device_type','?'):8s} {info.get('addr','?')}")

        elif cmd == "chain":
            print(f"  chain depth: {node.chain.depth()}  |  latest fp: "
                  f"{node.snapshot()['latest_fingerprint']}")

        elif cmd == "snap":
            import json
            print(json.dumps(node.snapshot(), indent=2))

        else:
            print(f"  unknown command: {cmd}")

    print("\n  shutting down, persisting chain...")
    await node.stop()


async def main():
    args = parse_args()
    advertise = args.advertise_host or "127.0.0.1"
    address = f"{advertise}:{args.port}"

    transport = UDPTransport(args.host, args.port,
                             broadcast_port=args.broadcast_port)
    node = JuvianNode(
        name=args.name,
        address=address,
        transport=transport,
        session_secret=args.session_secret,
        device_type=args.type,
        hw_weight=weight_for_type(args.type),
        mandelbrot_iter=args.iter,
        history_path=f"history_{args.name}.npy",
        chain_path=f"chain_{args.name}.frac",
    )

    await node.start()

    web_runner = None
    if args.web:
        # optional dashboard -- only import if requested so aiohttp stays optional
        try:
            from juvian_web import start_dashboard
            web_runner = await start_dashboard(node, args.web)
        except ImportError:
            print("  [web] aiohttp not installed; run: pip install aiohttp")
            print("  [web] continuing without dashboard")

    try:
        await repl(node, args.web)
    finally:
        if web_runner is not None:
            await web_runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
