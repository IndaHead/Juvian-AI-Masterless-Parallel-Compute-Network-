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
Juvian Grid :: honest scale study.

We CANNOT run 1,000,000 real JuvianNode objects in one process (RAM + O(N^2)
discovery + per-node PoW make it physically infeasible). So instead we:
  1. measure the REAL protocol stack at a ladder of feasible sizes,
  2. measure the REAL one-time PoW cost of minting an identity,
  3. run a GENUINE 1,000,000-entry keyspace computation (routing + Sybil math),
  4. extrapolate (1)-(3) to 10^6 with the arithmetic shown.
Everything labelled REAL is actually executed; everything labelled MODEL is an
extrapolation from a measured constant.
"""
import asyncio, time, random, heapq, os, gc, math
from juvian_identity import mint_pow
from juvian_transport import InMemoryBus, InMemoryTransport
from juvian_node import JuvianNode, POW_DIFFICULTY

PAGE = os.sysconf("SC_PAGE_SIZE")
def rss_bytes():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * PAGE

def human(n):
    for u in ["", "K", "M", "G", "T", "P"]:
        if abs(n) < 1000: return f"{n:.1f}{u}"
        n /= 1000
    return f"{n:.1f}E"

def human_secs(s):
    if s < 1:    return f"{s*1e3:.1f} ms"
    if s < 3600: return f"{s:.1f} s"
    if s < 86400:return f"{s/3600:.2f} h"
    return f"{s/86400:.2f} days"

# ---------------------------------------------------------------- PART 1: PoW
def measure_pow(diff=POW_DIFFICULTY, samples=8):
    ts, atts = [], []
    for _ in range(samples):
        pub = os.urandom(32)
        t = time.perf_counter(); _, a = mint_pow(pub, diff); ts.append(time.perf_counter()-t); atts.append(a)
    return sum(ts)/len(ts), sum(atts)/len(atts)

# ---------------------------------------------- PART 2: real-stack scale ladder
async def run_size(N, iters=80):
    bus = InMemoryBus()
    ctr = {"n": 0}
    nodes = []
    for i in range(N):
        addr = f"10.{(i>>16)&255}.{(i>>8)&255}.{i&255}:9000"
        nd = JuvianNode(f"n{i}", addr, InMemoryTransport(addr, bus),
                        mandelbrot_iter=iters, pow_difficulty=0,   # PoW measured separately
                        history_path=f"/tmp/sc{i}.npy", chain_path=f"/tmp/sc{i}.frac")
        orig = nd._on_message
        def wrap(o):
            async def w(env, sa):
                ctr["n"] += 1
                return await o(env, sa)
            return w
        nd._on_message = wrap(orig)
        nodes.append(nd)
    for nd in nodes:
        await nd.start()

    # discovery: every node announces once (all-to-all flood, as the demo does)
    ctr["n"] = 0
    t = time.perf_counter()
    for nd in nodes:
        await nd.announce()
    await asyncio.sleep(0.2)
    disc_t = time.perf_counter() - t
    disc_msgs = ctr["n"]
    avg_peers = sum(nd.routing.count() for nd in nodes) / N

    # one verify round driven by the canonical sequencer
    seq = min(nodes, key=lambda n: n.node_id)
    ctr["n"] = 0
    t = time.perf_counter()
    res = await seq.submit_request(b"scale", timeout=20.0)
    await asyncio.sleep(0.2)
    ver_t = time.perf_counter() - t
    ver_msgs = ctr["n"]
    depths = {nd.chain.depth() for nd in nodes}
    heads = {nd.chain.current_salt().hex() for nd in nodes}
    lockstep = len(depths) == 1 and len(heads) == 1

    for nd in nodes:
        await nd.stop()
    nodes.clear(); gc.collect()
    return dict(N=N, disc_t=disc_t, disc_msgs=disc_msgs, avg_peers=avg_peers,
                ver_t=ver_t, ver_msgs=ver_msgs, status=res.get("status"), lockstep=lockstep)

async def mem_per_node(N=120):
    gc.collect(); base = rss_bytes()
    bus = InMemoryBus(); nodes = []
    for i in range(N):
        addr = f"10.9.{(i>>8)&255}.{i&255}:9000"
        nd = JuvianNode(f"m{i}", addr, InMemoryTransport(addr, bus),
                        mandelbrot_iter=80, pow_difficulty=0,
                        history_path=f"/tmp/mm{i}.npy", chain_path=f"/tmp/mm{i}.frac")
        await nd.start(); nodes.append(nd)
    for nd in nodes:
        await nd.announce()
    await asyncio.sleep(0.2)
    used = rss_bytes() - base
    for nd in nodes:
        await nd.stop()
    return used / N

# ------------------------------------------ PART 3: genuine 1,000,000 keyspace
def keyspace_1M(N=1_000_000):
    t = time.perf_counter()
    ids = [random.getrandbits(160) for _ in range(N)]
    gen_t = time.perf_counter() - t
    target = random.getrandbits(160)                  # a request payload hash
    t = time.perf_counter()
    closest = heapq.nsmallest(5, ids, key=lambda x: x ^ target)  # the verifier set
    sel_t = time.perf_counter() - t
    d5 = (closest[-1] ^ target) + 1                   # XOR dist of the 5th-closest honest id
    SPACE = 1 << 160
    # to land ONE id closer to the target than the 5th honest id, a grinder needs
    # ~ SPACE/d5 random pubkeys (XOR is a bijection, so #ids within d5 == d5)
    grind_one = SPACE / d5
    return dict(gen_t=gen_t, sel_t=sel_t, d5_frac=d5/SPACE, grind_one=grind_one)

# ----------------------------------------------------------------------- main
async def main():
    print("="*70); print("JUVIAN GRID :: SCALE STUDY (real measurement + 10^6 extrapolation)"); print("="*70)

    print("\n[PART 1] REAL proof-of-work cost (the per-identity admission price)")
    pow_t, pow_att = measure_pow()
    rate = pow_att / pow_t
    print(f"  difficulty                : {POW_DIFFICULTY} leading zero bits")
    print(f"  measured per identity     : {human(pow_att)} hashes, {human_secs(pow_t)}  (~{human(rate)} H/s, single core)")

    print("\n[PART 2] REAL protocol stack at a scale ladder (in-memory bus, no PoW)")
    print(f"  {'N':>5} {'discTime':>9} {'discMsgs':>9} {'peers':>6} {'verTime':>8} {'verMsgs':>8} {'status':>9} {'lockstep':>8}")
    ladder = []
    for N in (10, 25, 50, 100):
        r = await run_size(N)
        ladder.append(r)
        print(f"  {r['N']:>5} {human_secs(r['disc_t']):>9} {human(r['disc_msgs']):>9} "
              f"{r['avg_peers']:>6.0f} {human_secs(r['ver_t']):>8} {human(r['ver_msgs']):>8} "
              f"{r['status']:>9} {str(r['lockstep']):>8}")
    bpn = await mem_per_node()
    print(f"  approx RAM per node (RSS delta): {human(bpn)}B")

    print("\n[PART 3] GENUINE 1,000,000-id keyspace computation (routing + Sybil)")
    ks = keyspace_1M()
    print(f"  generated 10^6 node ids   : {human_secs(ks['gen_t'])}")
    print(f"  verifier selection (5 XOR-closest of 10^6): {human_secs(ks['sel_t'])}  -- works fine at 10^6")
    print(f"  5th-closest id sits at    : {ks['d5_frac']:.2e} of the keyspace from the target")
    print(f"  => to grind ONE id nearer the target than the honest verifiers: ~{human(ks['grind_one'])} pubkey tries")
    print(f"     to capture a 3-of-5 quorum for a CHOSEN target: ~{human(3*ks['grind_one'])} grind tries")
    print(f"     ... PLUS a PoW cert per accepted id (~{human(pow_att)} hashes each at difficulty {POW_DIFFICULTY})")

    # -------- extrapolations to 10^6 --------
    print("\n[PART 4] MODEL :: extrapolation to 1,000,000 devices (flat single mesh)")
    M = 1_000_000
    # per-message handler cost, from the largest real rung's discovery
    big = ladder[-1]
    per_msg = big["disc_t"] / max(big["disc_msgs"], 1)
    # 1) PoW to admit 1e6 identities
    pow_total_s = M * pow_att / rate
    # 2) all-to-all flood discovery is O(N^2) messages
    disc_msgs_M = M * (M - 1)
    disc_time_M = disc_msgs_M * per_msg
    # 3) reliable-broadcast verify round fans out to all members => O(N) msgs/round
    ver_msgs_M = 3 * M                      # open + commits + result, ~3 per member
    ver_time_M = ver_msgs_M * per_msg
    # 4) RAM
    ram_M = M * bpn
    print(f"  per-message handler cost (measured)      : {human_secs(per_msg)}")
    print(f"  1) mint 10^6 PoW identities  (1 core)    : {human_secs(pow_total_s)}   "
          f"(embarrassingly parallel: /{os.cpu_count()} cores ~ {human_secs(pow_total_s/max(os.cpu_count(),1))})")
    print(f"  2) all-to-all flood discovery            : {human(disc_msgs_M)} msgs  ->  {human_secs(disc_time_M)}")
    print(f"  3) ONE verify round (O(N) fan-out bcast) : {human(ver_msgs_M)} msgs  ->  {human_secs(ver_time_M)} per request")
    print(f"  4) RAM for 10^6 nodes                    : {human(ram_M)}B")
    hops = math.log2(M)
    print(f"  routing lookup cost (Kademlia, theory)   : ~{hops:.0f} hops worst-case (O(log2 N)); "
          f"this is the ONE thing that scales")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("A FLAT 10^6-node mesh in one process is infeasible on three axes:")
    print(f"  - RAM    : ~{human(ram_M)}B,")
    print(f"  - PoW    : ~{human_secs(pow_total_s/max(os.cpu_count(),1))} of CPU just to mint identities,")
    print(f"  - O(N^2) flood discovery: ~{human(disc_msgs_M)} messages.")
    print("The verify path's reliable-broadcast fan-out is O(N) per request, which")
    print("also does not scale flat. What DOES scale is the DHT lookup (~20 hops).")
    print("10^6 is reached by FEDERATION, not a flat mesh: many geo-zones / subnets")
    print("each running the real protocol at feasible size (tens-thousands), with")
    print("hierarchical cross-zone routing. That is exactly what the geo-zone")
    print("sharding and the browser model's 8 regional subnets represent. The real")
    print("Python stack proven here is the PER-ZONE engine; the 10^6 figure is an")
    print("aggregate visualization, not the protocol executed a million times.")

if __name__ == "__main__":
    asyncio.run(main())
