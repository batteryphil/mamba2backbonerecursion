import http.server
import socketserver
import json
import os
import re

PORT = 8888

class MonitorHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Mamba-3 Latent Forge Telemetry</title>
                <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700&family=Fira+Code:wght@400;600&display=swap" rel="stylesheet">
                <style>
                    :root {
                        --bg-color: #090a0f;
                        --panel-bg: rgba(20, 22, 30, 0.6);
                        --glow: #00e5ff;
                        --text-main: #e2e8f0;
                        --text-highlight: #00e5ff;
                    }
                    body {
                        background-color: var(--bg-color);
                        background-image: radial-gradient(circle at 50% 0%, rgba(0, 229, 255, 0.15), transparent 50%);
                        color: var(--text-main);
                        font-family: 'Outfit', sans-serif;
                        margin: 0; padding: 3rem 2rem;
                        display: flex; flex-direction: column; align-items: center;
                        min-height: 100vh;
                        user-select: none;
                    }
                    h1 {
                        color: #fff;
                        text-shadow: 0 0 20px rgba(0, 229, 255, 0.4);
                        font-weight: 700; letter-spacing: 4px;
                        margin-bottom: 3rem;
                        font-size: 2.5rem;
                        text-transform: uppercase;
                    }
                    .dashboard {
                        display: grid;
                        grid-template-columns: repeat(3, 1fr);
                        gap: 30px; width: 100%; max-width: 1100px;
                        margin-bottom: 3rem;
                    }
                    .card {
                        background: var(--panel-bg);
                        border-radius: 16px;
                        padding: 30px 20px;
                        text-align: center;
                        border: 1px solid rgba(255, 255, 255, 0.05);
                        backdrop-filter: blur(20px);
                        -webkit-backdrop-filter: blur(20px);
                        box-shadow: 0 10px 30px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.1);
                        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                        position: relative;
                        overflow: hidden;
                    }
                    .card::before {
                        content: ''; position: absolute; top: 0; left: -100%; width: 50%; height: 100%;
                        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent);
                        transition: left 0.5s;
                    }
                    .card:hover::before { left: 100%; }
                    .card:hover { 
                        transform: translateY(-10px); 
                        border-color: rgba(0, 229, 255, 0.3);
                        box-shadow: 0 15px 40px rgba(0,229,255,0.15), inset 0 1px 0 rgba(255,255,255,0.1);
                    }
                    .card h2 { 
                        margin: 0; font-size: 0.9rem; color: #8b949e; 
                        text-transform: uppercase; letter-spacing: 2px; font-weight: 400;
                    }
                    .card .value { 
                        font-size: 3.5rem; font-weight: 700; color: #fff; 
                        margin-top: 15px; 
                        text-shadow: 0 0 15px rgba(0, 229, 255, 0.3);
                        transition: color 0.3s;
                    }
                    .terminal {
                        background: rgba(5, 6, 8, 0.8);
                        width: 100%; max-width: 1100px;
                        border-radius: 16px;
                        padding: 25px;
                        font-family: 'Fira Code', 'Courier New', Courier, monospace;
                        font-size: 0.95rem;
                        line-height: 1.6;
                        overflow-y: auto;
                        height: 400px;
                        border: 1px solid rgba(255,255,255,0.08);
                        box-shadow: inset 0 0 30px rgba(0,0,0,0.8), 0 10px 30px rgba(0,0,0,0.4);
                        white-space: pre-wrap;
                        backdrop-filter: blur(10px);
                        user-select: text;
                    }
                    .terminal::-webkit-scrollbar { width: 8px; }
                    .terminal::-webkit-scrollbar-track { background: rgba(0,0,0,0.3); border-radius: 4px; }
                    .terminal::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 4px; }
                    .terminal::-webkit-scrollbar-thumb:hover { background: rgba(0,229,255,0.3); }

                    .log-line { margin: 0; padding: 2px 0; color: #a0aec0; transition: color 0.2s;}
                    .log-highlight { color: #00e5ff; font-weight: 600; text-shadow: 0 0 8px rgba(0,229,255,0.4); }
                    .log-branch { color: #10b981; }
                    .log-system { color: #f59e0b; }
                    .status-indicator {
                        position: absolute; top: 20px; right: 20px;
                        width: 12px; height: 12px; border-radius: 50%;
                        background-color: #f43f5e;
                        box-shadow: 0 0 10px #f43f5e;
                        transition: background-color 0.3s, box-shadow 0.3s;
                    }
                    .status-online {
                        background-color: #10b981;
                        box-shadow: 0 0 15px #10b981;
                        animation: pulse 2s infinite;
                    }
                    @keyframes pulse {
                        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
                        70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
                        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
                    }
                </style>
            </head>
            <body>
                <div class="status-indicator" id="status-led"></div>
                <h1>Mamba-3 Void Telemetry</h1>
                <div class="dashboard">
                    <div class="card">
                        <h2>Step</h2>
                        <div class="value" id="val-step">--</div>
                    </div>
                    <div class="card">
                        <h2>Loss</h2>
                        <div class="value" id="val-loss">--</div>
                    </div>
                    <div class="card">
                        <h2>VRAM Usage</h2>
                        <div class="value" id="val-vram">--</div>
                    </div>
                </div>
                <div class="terminal" id="terminal-out">
                    <div class="log-system">Establishing dimensional link to the Void...</div>
                </div>
                
                <script>
                    let lastLogCount = 0;
                    async function fetchTelemetry() {
                        try {
                            let res = await fetch('/api/telemetry');
                            let data = await res.json();
                            
                            document.getElementById('status-led').classList.add('status-online');
                            
                            document.getElementById('val-step').innerText = data.step;
                            document.getElementById('val-loss').innerText = data.loss;
                            document.getElementById('val-vram').innerHTML = data.vram;
                            
                            let term = document.getElementById('terminal-out');
                            // Only update if logs changed to prevent scroll flickering
                            if(data.logs.length > 0 && data.logs[data.logs.length-1] !== term.lastChild?.innerText) {
                                term.innerHTML = '';
                                data.logs.forEach(line => {
                                    let div = document.createElement('div');
                                    div.className = 'log-line';
                                    if(line.includes("Loss:") || line.includes("Adv:")) div.className += ' log-highlight';
                                    else if(line.includes("Branch")) div.className += ' log-branch';
                                    else if(line.includes("[SYSTEM]") || line.includes("[INIT]")) div.className += ' log-system';
                                    
                                    div.innerText = line;
                                    term.appendChild(div);
                                });
                                term.scrollTop = term.scrollHeight;
                            }
                        } catch(e) {
                            document.getElementById('status-led').classList.remove('status-online');
                        }
                    }
                    setInterval(fetchTelemetry, 1500);
                    fetchTelemetry();
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
            
        elif self.path == '/api/telemetry':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            data = {"step": "--", "loss": "--", "reward": "--", "vram": "--", "logs": []}
            # Priority: always show the most recent active phase
            if os.path.exists("training_p14.log") and os.path.getsize("training_p14.log") > 0:
                log_file = "training_p14.log"
            elif os.path.exists("training_p13.log") and os.path.getsize("training_p13.log") > 0:
                log_file = "training_p13.log"
            elif os.path.exists("training_p12c.log") and os.path.getsize("training_p12c.log") > 0:
                log_file = "training_p12c.log"
            elif os.path.exists("training_p11.log"):
                log_file = "training_p11.log"
            elif os.path.exists("training_phase10.log"):
                log_file = "training_phase10.log"
            elif os.path.exists("training_gsm8k.log"):
                log_file = "training_gsm8k.log"
            else:
                log_file = "training.log"
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                    data["logs"] = lines[-20:] # Load tail
                    
                    # Back-scan for latest stats across all phase log formats
                    for line in reversed(lines):
                        # Phase 14 format: [P14 S00050] LM Loss: X | Halt Loss: X | Avg Loops: X | VRAM: X GB
                        m14 = re.search(r"\[P14 S(\d+)\].*LM Loss:\s*([\d\.]+).*Halt Loss:\s*([\d\.]+).*Avg Loops:\s*([\d\.]+).*VRAM:\s*([\d\.]+\s*GB)", line)
                        # Phase 13 format: [PHASE 13 S0050] Universal Target Masked Loss: X
                        m13 = re.search(r"\[PHASE 13 S(\d+)\].*Loss:\s*([\d\.]+)", line)
                        # Legacy GRPO format: [E1 S0190 G00191] Loss: X | R: X | VRAM: X GB
                        mgrpo = re.search(r"(?:\[E\d+ S\d+ G(\d+)\]|\[(\d+)\]) Loss:\s*([-\d\.]+).*?R:\s*([-\d\.]+).*?VRAM:\s*([\d\.]+\s*GB)", line)
                        if m14:
                            data["step"] = m14.group(1)
                            data["loss"] = m14.group(2)
                            data["reward"] = f"loops:{m14.group(4)}"
                            data["vram"] = m14.group(5)
                            break
                        elif m13:
                            data["step"] = m13.group(1)
                            data["loss"] = m13.group(2)
                            data["reward"] = "SFT"
                            data["vram"] = "--"
                            break
                        elif mgrpo:
                            data["step"] = mgrpo.group(1) or mgrpo.group(2)
                            data["loss"] = mgrpo.group(3)
                            data["reward"] = mgrpo.group(4)
                            data["vram"] = mgrpo.group(5)
                            break
                            
            self.wfile.write(json.dumps(data).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

# Silent output to avoid flooding the user's terminal
import logging
class QuietHandler(MonitorHandler):
    def log_message(self, format, *args):
        pass

socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("", PORT), QuietHandler) as httpd:
    print(f"======================================================")
    print(f"[SYSTEM] Telemetry Monitor Online. Port {PORT} Bound.")
    print(f"======================================================")
    httpd.serve_forever()
