#!/usr/bin/env python3
"""
rlf_dashboard.py — Phone-accessible training monitor + command server.

Access from phone browser: http://<machine-ip>:7860

Commands available from phone:
  /status   — current step, loss, mem_norm
  /kill     — stop training gracefully
  /restart  — restart from last checkpoint
  /log      — last 30 log lines
"""

import re
import os
import json
import signal
import subprocess
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

LOG      = Path("/home/phil/.gemini/antigravity/scratch/tiny-refinement/rlf_trainer.log")
DIAG_DIR = Path("/home/phil/.gemini/antigravity/scratch/tiny-refinement/diagnostics")
PORT     = 7860

PHASE_RE = re.compile(
    r"\[Phase(\w+)\]\[S(\d+)\] Loss=([\d.inf]+).*?Acc=([\d.]+).*?mem_norm=([\d.]+)"
)


def get_status() -> dict:
    """Parse the last status from the training log."""
    try:
        result = subprocess.run(["strings", str(LOG)],
                                capture_output=True, text=True, timeout=5)
        lines = result.stdout.splitlines()
    except Exception:
        return {}

    for line in reversed(lines):
        m = PHASE_RE.search(line)
        if m:
            phase, step, loss, acc, mem_norm = m.groups()
            total = {"3a": 6000, "3b": 8000, "3c": 1000}.get(phase, 1)
            pct   = int(step) / total * 100
            return {
                "phase":    phase,
                "step":     int(step),
                "total":    total,
                "pct":      pct,
                "loss":     loss,
                "acc":      acc,
                "mem_norm": mem_norm,
                "updated":  datetime.now().strftime("%H:%M:%S"),
            }
    return {"phase": "starting", "step": 0, "total": 1, "pct": 0,
            "loss": "—", "acc": "—", "mem_norm": "—", "updated": "—"}


def get_log_tail(n: int = 30) -> str:
    """Return last n printable log lines."""
    try:
        result = subprocess.run(["strings", str(LOG)],
                                capture_output=True, text=True, timeout=5)
        lines = [l for l in result.stdout.splitlines() if "[Phase" in l or "gate" in l.lower() or "complete" in l.lower()]
        return "\n".join(lines[-n:])
    except Exception:
        return "Log unavailable"


def get_trainer_pid() -> int | None:
    """Find the running trainer PID."""
    try:
        r = subprocess.run(["pgrep", "-f", "rlf_trainer_1_4b.py"],
                           capture_output=True, text=True)
        pids = r.stdout.strip().splitlines()
        return int(pids[0]) if pids else None
    except Exception:
        return None


def build_html(s: dict) -> str:
    """Build the mobile-friendly dashboard HTML."""
    phase_color = {"3a": "#4ade80", "3b": "#facc15", "3c": "#60a5fa"}.get(
        s.get("phase", ""), "#94a3b8"
    )
    norm = float(s.get("mem_norm", 0) or 0)
    norm_color = "#4ade80" if 0.3 <= norm <= 5.0 else "#f87171"
    pid = get_trainer_pid()
    status_dot = "🟢" if pid else "🔴"

    alert = ""
    alert_file = DIAG_DIR / "ALERT.txt"
    if alert_file.exists():
        alert_text = alert_file.read_text()[:300]
        alert = f"""<div style="background:#7f1d1d;border-radius:12px;padding:16px;margin:12px 0">
            <b>⚠️ ALERT</b><pre style="font-size:12px;white-space:pre-wrap">{alert_text}</pre>
            </div>"""

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta http-equiv="refresh" content="30">
  <title>RLF V3 Monitor</title>
  <style>
    * {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{ background:#0f172a; color:#e2e8f0; font-family:system-ui,sans-serif; padding:16px; }}
    h1 {{ font-size:20px; margin-bottom:16px; }}
    .card {{ background:#1e293b; border-radius:12px; padding:16px; margin-bottom:12px; }}
    .label {{ font-size:11px; color:#94a3b8; text-transform:uppercase; margin-bottom:4px; }}
    .value {{ font-size:28px; font-weight:bold; }}
    .row {{ display:flex; gap:12px; }}
    .row .card {{ flex:1; }}
    .bar-bg {{ background:#334155; border-radius:99px; height:12px; margin-top:8px; }}
    .bar-fill {{ height:12px; border-radius:99px; background:#4ade80; transition:width 0.5s; }}
    .cmd {{ display:block; background:#334155; border:none; color:#e2e8f0;
            font-size:16px; padding:14px; border-radius:12px;
            margin-bottom:8px; width:100%; cursor:pointer; text-align:left; }}
    .cmd:active {{ background:#475569; }}
    pre {{ font-size:11px; color:#94a3b8; white-space:pre-wrap; overflow-x:auto; }}
    .badge {{ display:inline-block; padding:2px 10px; border-radius:99px;
              font-size:12px; font-weight:bold; }}
  </style>
</head>
<body>
  <h1>🤖 RLF V3 Monitor</h1>
  {alert}

  <div class="card">
    <div class="label">Status</div>
    <div class="value">{status_dot} Phase {s.get('phase','?').upper()}</div>
    <div style="margin-top:8px">
      <span class="badge" style="background:{phase_color};color:#000">
        S{s.get('step',0):05d} / {s.get('total',1)}
      </span>
      &nbsp;{s.get('pct',0):.1f}% complete
    </div>
    <div class="bar-bg"><div class="bar-fill" style="width:{min(s.get('pct',0),100):.1f}%"></div></div>
    <div style="font-size:12px;color:#64748b;margin-top:6px">Updated: {s.get('updated','—')}</div>
  </div>

  <div class="row">
    <div class="card">
      <div class="label">Loss</div>
      <div class="value" style="font-size:22px">{s.get('loss','—')}</div>
    </div>
    <div class="card">
      <div class="label">Acc</div>
      <div class="value" style="font-size:22px">{s.get('acc','—')}</div>
    </div>
    <div class="card">
      <div class="label">mem_norm</div>
      <div class="value" style="font-size:22px;color:{norm_color}">{s.get('mem_norm','—')}</div>
    </div>
  </div>

  <div class="card">
    <div class="label">Commands</div>
    <button class="cmd" onclick="cmd('/api/restart')">🔄 Restart from checkpoint</button>
    <button class="cmd" onclick="cmd('/api/kill')">🛑 Kill trainer</button>
    <button class="cmd" onclick="location.reload()">🔃 Refresh now</button>
    <div id="resp" style="font-size:13px;color:#4ade80;margin-top:8px"></div>
  </div>

  <div class="card">
    <div class="label">Recent Log</div>
    <pre id="log">{get_log_tail(20)}</pre>
  </div>

  <script>
    function cmd(url) {{
      fetch(url).then(r=>r.json()).then(d=>{{
        document.getElementById('resp').textContent = d.msg;
      }});
    }}
    // Auto-refresh log every 30s
    setInterval(()=>{{
      fetch('/api/log').then(r=>r.json()).then(d=>{{
        document.getElementById('log').textContent=d.log;
      }});
    }}, 30000);
  </script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    """Simple HTTP handler for the dashboard."""

    def log_message(self, fmt, *args):
        """Suppress default access log."""
        pass

    def send_json(self, data: dict) -> None:
        """Send a JSON response."""
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/" or self.path == "/dashboard":
            body = build_html(get_status()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/api/status":
            self.send_json(get_status())

        elif self.path == "/api/log":
            self.send_json({"log": get_log_tail(30)})

        elif self.path == "/api/kill":
            pid = get_trainer_pid()
            if pid:
                os.kill(pid, signal.SIGTERM)
                self.send_json({"msg": f"Sent SIGTERM to PID {pid}"})
            else:
                self.send_json({"msg": "No trainer process found"})

        elif self.path == "/api/restart":
            pid = get_trainer_pid()
            if pid:
                os.kill(pid, signal.SIGTERM)
            nccl = "/home/phil/.gemini/antigravity/scratch/quill/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2"
            site = "/home/phil/.local/share/mise/installs/python/3.14.3/lib/python3.14/site-packages"
            python = "/home/phil/.local/share/mise/installs/python/3.14.3/bin/python"
            env = os.environ.copy()
            env["LD_PRELOAD"] = f"{nccl}:{site}/nvidia/cu13/lib/libnvJitLink.so.13"
            env["TOKENIZERS_PARALLELISM"] = "false"
            trainer = Path("/home/phil/.gemini/antigravity/scratch/mamba2backbonerecursion/mamba14b")
            log_f = open(str(LOG), "a")
            subprocess.Popen(
                [python, "-B", "auto_recovery.py"],
                cwd=str(trainer), env=env,
                stdout=log_f, stderr=log_f
            )
            self.send_json({"msg": "Trainer restarted via auto_recovery.py"})

        else:
            self.send_response(404)
            self.end_headers()


def main() -> None:
    """Start the dashboard server."""
    # Print local IP for phone access
    try:
        r = subprocess.run(["hostname", "-I"], capture_output=True, text=True)
        ip = r.stdout.strip().split()[0]
    except Exception:
        ip = "localhost"

    print(f"\n{'='*50}")
    print(f"  RLF Dashboard running")
    print(f"  Open on phone: http://{ip}:{PORT}")
    print(f"  Auto-refreshes every 30s")
    print(f"{'='*50}\n")

    server = HTTPServer(("0.0.0.0", PORT), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
