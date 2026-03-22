"""
monitor_v25.py — Live Training Dashboard for Mamba-3 V25
=========================================================
Parses finetune_v25_run.log and serves a real-time web dashboard.
Run:  python monitor_v25.py
Then open:  http://localhost:8080
"""
import re
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

LOG_FILE = "finetune_v25_run.log"
PORT = 8082

STEP_RE = re.compile(
    r"Step\s+(\d+)\s*\|\s*Loss:\s*([\d.]+)\s*\|\s*Acc:\s*([\d.]+)%"
    r"\s*\|\s*LR(?:\(\w+\))?:\s*([\deE.+-]+)"   # matches 'LR:' or 'LR(emb):'
    r".*?TPS:\s*([\d.]+)"                         # skip any text between LR and TPS
    r"\s*\|\s*VRAM:\s*([\d.]+)GB\s*\|\s*MaxN:\s*(\d+)"
)


def parse_log():
    """Parse the training log and return structured data."""
    path = Path(LOG_FILE)
    if not path.exists():
        return {"steps": [], "error": f"{LOG_FILE} not found"}

    steps, losses, accs, tps_list, vrams, maxns, lrs = [], [], [], [], [], [], []
    milestone_steps = []
    curr_maxn = 1

    for line in path.read_text().splitlines():
        m = STEP_RE.search(line)
        if m:
            step, loss, acc, lr, tps, vram, maxn = m.groups()
            steps.append(int(step))
            losses.append(float(loss))
            accs.append(float(acc))
            lrs.append(float(lr))
            tps_list.append(float(tps))
            vrams.append(float(vram))
            maxns.append(int(maxn))
            if int(maxn) != curr_maxn:
                milestone_steps.append({"step": int(step), "from": curr_maxn, "to": int(maxn)})
                curr_maxn = int(maxn)

    latest = {}
    if steps:
        latest = {
            "step": steps[-1], "loss": losses[-1], "acc": accs[-1],
            "tps": tps_list[-1], "vram": vrams[-1], "maxn": maxns[-1], "lr": lrs[-1]
        }

    return {
        "steps": steps, "losses": losses, "accs": accs,
        "tps": tps_list, "vrams": vrams, "maxns": maxns,
        "milestones": milestone_steps, "latest": latest
    }


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mamba-3 V25 Training Monitor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #0a0e1a;
    --surface:   #111827;
    --border:    #1f2937;
    --accent:    #6366f1;
    --accent2:   #22d3ee;
    --green:     #10b981;
    --orange:    #f59e0b;
    --red:       #ef4444;
    --text:      #f1f5f9;
    --muted:     #64748b;
    --glow:      rgba(99,102,241,0.15);
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
    padding: 24px;
  }

  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 28px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
  }

  .title-block h1 {
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .title-block p {
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
  }

  .live-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.3);
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 0.75rem;
    color: var(--green);
    font-weight: 600;
  }

  .live-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse 1.4s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.8); }
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
  }

  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
  }

  .stat-card:hover {
    border-color: var(--accent);
    transform: translateY(-2px);
  }

  .stat-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: var(--glow);
    opacity: 0;
    transition: opacity 0.2s;
  }

  .stat-card:hover::before { opacity: 1; }

  .stat-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
  }

  .stat-value {
    font-size: 1.9rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
  }

  .stat-sub {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 6px;
    font-family: 'JetBrains Mono', monospace;
  }

  .color-acc    { color: var(--green); }
  .color-loss   { color: var(--accent); }
  .color-tps    { color: var(--accent2); }
  .color-vram   { color: var(--orange); }
  .color-maxn   { color: #a78bfa; }
  .color-step   { color: var(--text); }

  .charts-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 20px;
    margin-bottom: 20px;
  }

  .chart-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
  }

  .chart-card h3 {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 14px;
  }

  .chart-wrapper {
    position: relative;
    height: 200px;
  }

  .chart-wide {
    grid-column: 1 / -1;
  }

  .milestones {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
  }

  .milestones h3 {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
  }

  .milestone-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 8px;
    margin-bottom: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
  }

  .milestone-item .rocket { font-size: 1.1rem; }
  .no-milestones { color: var(--muted); font-size: 0.85rem; font-style: italic; }

  footer {
    text-align: center;
    margin-top: 20px;
    color: var(--muted);
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
  }
</style>
</head>
<body>

<header>
  <div class="title-block">
    <h1>⬡ Mamba-3 V25 Training Monitor</h1>
    <p>mamba-130m · JIT Fused CUDA MIMO Phase Kernel · N-Scale ACT</p>
  </div>
  <div class="live-badge"><div class="live-dot"></div>LIVE — refreshing every 5s</div>
</header>

<div class="stats-grid">
  <div class="stat-card">
    <div class="stat-label">Step</div>
    <div class="stat-value color-step" id="val-step">—</div>
    <div class="stat-sub" id="val-lr">LR: —</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Accuracy (Loop N)</div>
    <div class="stat-value color-acc" id="val-acc">—</div>
    <div class="stat-sub">Target: 85.0% → N+1</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Loss</div>
    <div class="stat-value color-loss" id="val-loss">—</div>
    <div class="stat-sub">Cross-Entropy (all loops)</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">TPS</div>
    <div class="stat-value color-tps" id="val-tps">—</div>
    <div class="stat-sub">Tokens / sec</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">VRAM</div>
    <div class="stat-value color-vram" id="val-vram">—</div>
    <div class="stat-sub">GPU allocated</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Max N (Loops)</div>
    <div class="stat-value color-maxn" id="val-maxn">—</div>
    <div class="stat-sub">Current curriculum depth</div>
  </div>
</div>

<div class="charts-grid">
  <div class="chart-card chart-wide">
    <h3>Accuracy — Final Loop Only (Curriculum Gate)</h3>
    <div class="chart-wrapper"><canvas id="chartAcc"></canvas></div>
  </div>
  <div class="chart-card">
    <h3>Loss (all loops)</h3>
    <div class="chart-wrapper"><canvas id="chartLoss"></canvas></div>
  </div>
  <div class="chart-card">
    <h3>TPS (1/N Scale)</h3>
    <div class="chart-wrapper"><canvas id="chartTPS"></canvas></div>
  </div>
  <div class="chart-card">
    <h3>VRAM Usage</h3>
    <div class="chart-wrapper"><canvas id="chartVRAM"></canvas></div>
  </div>
  <div class="chart-card">
    <h3>MaxN (Curriculum Depth)</h3>
    <div class="chart-wrapper"><canvas id="chartMaxN"></canvas></div>
  </div>
</div>

<div class="milestones">
  <h3>🚀 Curriculum Upgrades</h3>
  <div id="milestone-list"><span class="no-milestones">Waiting for first curriculum graduation…</span></div>
</div>

<footer id="footer">Last updated: —</footer>

<script>
const WINDOW = 300; // max data points shown

function makeChart(id, label, color, fill=true, yMin=null, yMax=null) {
  const ctx = document.getElementById(id).getContext('2d');
  const grad = ctx.createLinearGradient(0, 0, 0, 200);
  grad.addColorStop(0, color + '33');
  grad.addColorStop(1, color + '00');
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label, data: [],
        borderColor: color,
        borderWidth: 2,
        backgroundColor: fill ? grad : 'transparent',
        fill: fill,
        tension: 0.35,
        pointRadius: 0,
        pointHoverRadius: 4,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: '#1f2937' }, ticks: { color: '#64748b', maxTicksLimit: 8, font: { family: 'JetBrains Mono', size: 10 } } },
        y: {
          grid: { color: '#1f2937' },
          ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } },
          ...(yMin !== null ? { min: yMin } : {}),
          ...(yMax !== null ? { max: yMax } : {})
        }
      }
    }
  });
}

const charts = {
  acc:   makeChart('chartAcc',   'Accuracy (%)', '#10b981', true, 0, 100),
  loss:  makeChart('chartLoss',  'Loss',         '#6366f1', true),
  tps:   makeChart('chartTPS',   'TPS',          '#22d3ee', true),
  vram:  makeChart('chartVRAM',  'VRAM (GB)',    '#f59e0b', true, 0, 12),
  maxn:  makeChart('chartMaxN',  'MaxN',         '#a78bfa', false, 0),
};

function updateChart(chart, labels, data) {
  const sl = labels.slice(-WINDOW);
  const sd = data.slice(-WINDOW);
  chart.data.labels = sl;
  chart.data.datasets[0].data = sd;
  chart.update('none');
}

async function refresh() {
  try {
    const res = await fetch('/data');
    const d = await res.json();

    if (d.error) { document.getElementById('footer').textContent = d.error; return; }
    if (!d.steps.length) return;

    const L = d.latest;
    document.getElementById('val-step').textContent  = L.step.toLocaleString();
    document.getElementById('val-lr').textContent    = 'LR: ' + L.lr.toExponential(2);
    document.getElementById('val-acc').textContent   = L.acc.toFixed(1) + '%';
    document.getElementById('val-loss').textContent  = L.loss.toFixed(4);
    document.getElementById('val-tps').textContent   = Math.round(L.tps).toLocaleString();
    document.getElementById('val-vram').textContent  = L.vram.toFixed(2) + 'GB';
    document.getElementById('val-maxn').textContent  = 'N=' + L.maxn;

    updateChart(charts.acc,  d.steps, d.accs);
    updateChart(charts.loss, d.steps, d.losses);
    updateChart(charts.tps,  d.steps, d.tps);
    updateChart(charts.vram, d.steps, d.vrams);
    updateChart(charts.maxn, d.steps, d.maxns);

    const ml = document.getElementById('milestone-list');
    if (d.milestones.length) {
      ml.innerHTML = d.milestones.map(m =>
        `<div class="milestone-item"><span class="rocket">🚀</span>
         Step ${m.step.toLocaleString()} — Curriculum upgraded N=${m.from} → N=${m.to}</div>`
      ).join('');
    }

    document.getElementById('footer').textContent =
      'Last updated: ' + new Date().toLocaleTimeString() + ' · Step ' + L.step.toLocaleString() + ' of 50,000';

  } catch(e) {
    document.getElementById('footer').textContent = 'Waiting for log data… (' + e.message + ')';
  }
}

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    """HTTP handler serving the dashboard and JSON data."""

    def log_message(self, fmt, *args):
        """Suppress default access logs."""
        pass

    def do_GET(self):
        """Handle GET requests for the dashboard and data API."""
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML.encode())
        elif self.path == '/data':
            data = json.dumps(parse_log()).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', PORT), Handler)
    print(f"\n  Mamba-3 V25 Monitor running at  http://localhost:{PORT}")
    print(f"  Reading: {LOG_FILE}")
    print(f"  Ctrl+C to stop\n")
    server.serve_forever()
