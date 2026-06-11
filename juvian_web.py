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
JUVIAN GRID :: PER-NODE WEB DASHBOARD
==============================================================================

A tiny aiohttp server that exposes one live node's real state:

  GET /            -> single-page dashboard (auto-refreshing)
  GET /api/snapshot-> JSON snapshot of this node (peers, chain, stats)
  POST /api/request-> submit a compute-request; runs real 3-of-3 verification
  POST /api/tensor -> run a real distributed tensor reduce

This reflects the ACTUAL node process -- the peers, chain depth, fingerprints
and verification outcomes shown are real, not simulated. Start it via:

  python3 run_node.py --name anchor-1 --port 9101 --web 8080

Requires aiohttp:  pip install aiohttp
"""

import json
import numpy as np

try:
    from aiohttp import web
except ImportError:  # pragma: no cover
    web = None


PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>JUVIAN NODE :: {name}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@700;900&display=swap');
:root{{--bg:#04080c;--panel:#081119;--bd:#0d2536;--accent:#00e5ff;--green:#00ff88;
--amber:#ffaa00;--red:#ff2a44;--dim:#244a5e;--txt:#7fc6dc;}}
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:var(--bg);color:var(--txt);font-family:'Share Tech Mono',monospace;
font-size:13px;padding:18px;}}
body::before{{content:'';position:fixed;inset:0;background:repeating-linear-gradient(
0deg,transparent,transparent 2px,rgba(0,229,255,.012) 2px,rgba(0,229,255,.012) 4px);
pointer-events:none;}}
h1{{font-family:'Orbitron',sans-serif;font-weight:900;font-size:22px;color:var(--accent);
letter-spacing:5px;text-shadow:0 0 24px var(--accent);margin-bottom:2px;}}
.sub{{color:var(--dim);font-size:11px;letter-spacing:2px;margin-bottom:18px;}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;
margin-bottom:18px;}}
.stat{{background:var(--panel);border:1px solid var(--bd);padding:12px;}}
.stat .l{{font-size:9px;color:var(--dim);letter-spacing:2px;margin-bottom:4px;}}
.stat .v{{font-family:'Orbitron',sans-serif;font-size:22px;font-weight:700;
color:var(--green);text-shadow:0 0 10px var(--green);}}
.stat .v.accent{{color:var(--accent);text-shadow:0 0 10px var(--accent);}}
.stat .v.amber{{color:var(--amber);text-shadow:0 0 10px var(--amber);}}
.stat .v.small{{font-size:12px;word-break:break-all;}}
.panel{{background:var(--panel);border:1px solid var(--bd);padding:14px;margin-bottom:14px;}}
.panel h2{{font-family:'Orbitron',sans-serif;font-size:11px;color:var(--accent);
letter-spacing:3px;margin-bottom:10px;}}
button{{font-family:'Orbitron',sans-serif;font-size:10px;font-weight:700;letter-spacing:2px;
padding:8px 16px;border:1px solid var(--accent);background:rgba(0,229,255,.05);
color:var(--accent);cursor:pointer;margin-right:8px;text-transform:uppercase;}}
button:hover{{background:var(--accent);color:var(--bg);box-shadow:0 0 16px var(--accent);}}
input{{background:#02070b;border:1px solid var(--bd);color:var(--txt);padding:8px;
font-family:'Share Tech Mono',monospace;font-size:12px;width:240px;margin-right:8px;}}
table{{width:100%;border-collapse:collapse;font-size:11px;}}
th,td{{text-align:left;padding:5px 8px;border-bottom:1px solid var(--bd);}}
th{{color:var(--dim);font-size:9px;letter-spacing:1px;}}
#out{{font-size:11px;color:var(--green);min-height:18px;margin-top:8px;white-space:pre-wrap;}}
.ok{{color:var(--green);}} .bad{{color:var(--red);}}
</style></head><body>
<h1>JUVIAN NODE</h1>
<div class="sub">{name} &nbsp;|&nbsp; id {node_id} &nbsp;|&nbsp; {device_type} &nbsp;|&nbsp; {address}</div>

<div class="grid">
  <div class="stat"><div class="l">PEERS</div><div class="v" id="s-peers">-</div></div>
  <div class="stat"><div class="l">CHAIN DEPTH</div><div class="v accent" id="s-chain">-</div></div>
  <div class="stat"><div class="l">KEYS VERIFIED</div><div class="v" id="s-verified">-</div></div>
  <div class="stat"><div class="l">REJECTED</div><div class="v amber" id="s-rejected">-</div></div>
  <div class="stat"><div class="l">TENSOR JOBS</div><div class="v accent" id="s-tensor">-</div></div>
  <div class="stat"><div class="l">BYZANTINE PURGED</div><div class="v amber" id="s-purged">-</div></div>
</div>

<div class="stat" style="margin-bottom:14px;">
  <div class="l">LATEST KEY FINGERPRINT</div>
  <div class="v small accent" id="s-fp">-</div>
</div>

<div class="panel">
  <h2>SUBMIT COMPUTE REQUEST &mdash; REAL 3-OF-3 VERIFICATION</h2>
  <input id="req-text" placeholder="payload text" value="hello-juvian">
  <button onclick="submitReq()">SUBMIT REQUEST</button>
  <button onclick="runTensor()">RUN TENSOR REDUCE 8&times;8</button>
  <div id="out"></div>
</div>

<div class="panel">
  <h2>DISCOVERED PEERS</h2>
  <table><thead><tr><th>NODE ID</th><th>NAME</th><th>TYPE</th><th>ADDRESS</th></tr></thead>
  <tbody id="peer-rows"><tr><td colspan="4">loading...</td></tr></tbody></table>
</div>

<script>
async function refresh(){{
  try{{
    const r = await fetch('/api/snapshot'); const s = await r.json();
    document.getElementById('s-peers').textContent    = s.peers;
    document.getElementById('s-chain').textContent    = s.chain_depth;
    document.getElementById('s-verified').textContent = s.stats.keys_verified;
    document.getElementById('s-rejected').textContent = s.stats.keys_rejected;
    document.getElementById('s-tensor').textContent   = s.stats.tensor_jobs;
    document.getElementById('s-purged').textContent   = s.stats.byzantine_purged;
    document.getElementById('s-fp').textContent       = s.latest_fingerprint;
    const rows = (s.peer_list||[]).map(p =>
      `<tr><td>${{p.id}}</td><td>${{p.name}}</td><td>${{p.device_type}}</td><td>${{p.addr}}</td></tr>`
    ).join('') || '<tr><td colspan="4">no peers yet</td></tr>';
    document.getElementById('peer-rows').innerHTML = rows;
  }}catch(e){{}}
}}
async function submitReq(){{
  const text = document.getElementById('req-text').value || 'ping';
  const out = document.getElementById('out');
  out.textContent = 'submitting...';
  const r = await fetch('/api/request',{{method:'POST',headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{payload:text}})}});
  const res = await r.json();
  if(res.status==='VERIFIED'){{
    out.innerHTML = `<span class="ok">VERIFIED</span>  fp=${{(res.fingerprint||'').slice(0,16)}}  `
      + `chain_depth=${{res.chain_depth}}  verifiers=${{(res.verifiers||[]).map(v=>v.slice(0,8)).join(', ')}}`;
  }}else{{
    out.innerHTML = `<span class="bad">${{res.status}}</span>  ${{res.reason||''}}`;
  }}
  refresh();
}}
async function runTensor(){{
  const out = document.getElementById('out');
  out.textContent = 'running distributed tensor reduce...';
  const r = await fetch('/api/tensor',{{method:'POST',headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{n:8}})}});
  const res = await r.json();
  out.innerHTML = `<span class="ok">${{res.status}}</span>  energy=${{(res.frobenius_energy||0).toFixed(2)}}  `
    + `valid=${{(res.valid_nodes||[]).length}}  purged=${{(res.purged_nodes||[]).length}}`;
  refresh();
}}
refresh(); setInterval(refresh, 2000);
</script>
</body></html>
"""


def _snapshot_with_peers(node) -> dict:
    snap = node.snapshot()
    peers = node.routing.all_peers()
    snap["peer_list"] = [
        {"id": pid[:16], "name": info.get("name", "?"),
         "device_type": info.get("device_type", "?"),
         "addr": info.get("addr", "?")}
        for pid, info in peers.items()
    ]
    return snap


async def start_dashboard(node, port: int):
    """Start the aiohttp dashboard bound to 127.0.0.1:<port>.
    Returns the AppRunner so the caller can clean it up on shutdown."""
    if web is None:
        raise ImportError("aiohttp is required for the web dashboard")

    async def index(request):
        snap = node.snapshot()
        html = PAGE.format(
            name=node.name,
            node_id=node.node_id[:16],
            device_type=node.device_type,
            address=node.address,
        )
        return web.Response(text=html, content_type="text/html")

    async def api_snapshot(request):
        return web.json_response(_snapshot_with_peers(node))

    async def api_request(request):
        body = await request.json()
        payload = (body.get("payload") or "ping").encode()
        res = await node.submit_request(payload, timeout=6.0)
        safe = {k: v for k, v in res.items()
                if k not in ("fernet_key", "derived")}
        return web.json_response(safe)

    async def api_tensor(request):
        body = await request.json()
        n = int(body.get("n", 8))
        tensor = np.random.default_rng().standard_normal((n, n))
        res = await node.run_tensor_job(tensor, rank=min(4, n), timeout=6.0)
        safe = {k: (v if _jsonable(v) else str(v)) for k, v in res.items()}
        return web.json_response(safe)

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/api/snapshot", api_snapshot)
    app.router.add_post("/api/request", api_request)
    app.router.add_post("/api/tensor", api_tensor)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    return runner


def _jsonable(v) -> bool:
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False
