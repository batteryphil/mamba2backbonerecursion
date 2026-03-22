"""
JARVIS Web UI - v3.1
A premium Flask-based interface for DiM-LLM v3.
Runs independently of the training script. Safe to launch while training is active.
"""
import torch
import os
import sys

# Windows DLL path fix for custom CUDA extensions
if os.name == 'nt':
    cuda_path = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1")
    cuda_bin = os.path.join(cuda_path, "bin")
    if os.path.exists(cuda_bin):
        os.add_dll_directory(cuda_bin)
    
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.exists(torch_lib):
        os.add_dll_directory(torch_lib)
import time
import os
import json
import copy
import threading
from flask import Flask, request, jsonify, Response, stream_with_context
from transformers import GPT2Tokenizer
from mamba_llm_diffusion import DiM_LLM, MaskedDiffusionEngine, Config

app = Flask(__name__)

# ─── Model Initialization (Runs once at server startup) ───────────────────────
print("Initializing JARVIS Web UI...")
# Force CPU for inference so training gets the full GPU uncontested
DEVICE = "cpu"
print("  -> Inference locked to CPU (training owns GPU)")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"mask_token": "[MASK]"})

config = Config(vocab_size=len(tokenizer), d_model=1024, n_layers=11, seq_len=256)
model = DiM_LLM(config).to(DEVICE)

# 🚀 Hardware Acceleration: Fuse CUDA Kernels
try:
    if DEVICE == "cuda":
        print("  -> Compiling model (RTX 3060 optimization)...")
        model = torch.compile(model, backend="inductor")
    else:
        # On CPU, compile still helps with graph optimization
        model = torch.compile(model)
except Exception as e:
    print(f"  -> torch.compile skipped or failed: {e}")

EMA_PATH = "dim_llm_ema_checkpoint.pt"
MAIN_PATH = "dim_llm_checkpoint.pt"

if os.path.exists(EMA_PATH):
    print(f"  -> Loaded EMA weights from {EMA_PATH} (Premium Mode)")
    model.load_state_dict(torch.load(EMA_PATH, map_location=DEVICE))
elif os.path.exists(MAIN_PATH):
    print(f"  -> Loaded main weights from {MAIN_PATH}")
    model.load_state_dict(torch.load(MAIN_PATH, map_location=DEVICE))
else:
    print("  -> WARNING: No checkpoint found. Running on random initialization.")

ema_model = copy.deepcopy(model)
engine = MaskedDiffusionEngine(model, config, device=DEVICE, ema_decay=0.999)
engine.ema_model = ema_model
model.eval()
print(f"  -> JARVIS online on {DEVICE}. Ready.")

# ─── Hot-Reload State ─────────────────────────────────────────────────────────
_reload_lock = threading.Lock()
_reload_state = {
    "last_mtime": os.path.getmtime(EMA_PATH) if os.path.exists(EMA_PATH) else 0,
    "reload_count": 0,
    "last_reload_msg": "Weights loaded at startup"
}

def _weight_watcher():
    """Background thread: reloads EMA weights whenever the checkpoint file changes."""
    global model, engine
    while True:
        time.sleep(15)  # poll every 15 seconds
        try:
            if not os.path.exists(EMA_PATH):
                continue
            mtime = os.path.getmtime(EMA_PATH)
            if mtime > _reload_state["last_mtime"]:
                print(f"  [Hot-Reload] New checkpoint detected — reloading EMA weights...")
                with _reload_lock:
                    state = torch.load(EMA_PATH, map_location=DEVICE)
                    model.load_state_dict(state)
                    engine.ema_model.load_state_dict(state)
                    model.eval()
                    _reload_state["last_mtime"] = mtime
                    _reload_state["reload_count"] += 1
                    _reload_state["last_reload_msg"] = (
                        f"Weights reloaded #{_reload_state['reload_count']} "
                        f"— {time.strftime('%H:%M:%S')}"
                    )
                print(f"  [Hot-Reload] Done. ({_reload_state['last_reload_msg']})")
        except Exception as e:
            print(f"  [Hot-Reload] Error: {e}")

# Start watcher as daemon so it exits when the server exits
_watcher_thread = threading.Thread(target=_weight_watcher, daemon=True)
_watcher_thread.start()
print("  -> Weight hot-reload watcher running (15s interval)")



# ─── HTML PAGE ────────────────────────────────────────────────────────────────
UI_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>JARVIS — DiM-LLM v3.1</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg:       #08090f;
      --surface:  #0e101a;
      --border:   #1d2036;
      --cyan:     #00e5ff;
      --pink:     #ff2d78;
      --purple:   #9b5de5;
      --text:     #d4d8f0;
      --muted:    #5a5f7a;
      --green:    #05ffa1;
      --radius:   14px;
    }

    * { margin:0; padding:0; box-sizing:border-box; }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Outfit', sans-serif;
      height: 100vh;
      display: grid;
      grid-template-columns: 280px 1fr;
      grid-template-rows: 60px 1fr;
      overflow: hidden;
    }

    /* ── Header ── */
    header {
      grid-column: 1 / -1;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      padding: 0 24px;
      gap: 16px;
    }
    .logo-dot {
      width: 10px; height: 10px;
      border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 10px var(--green);
      animation: pulse 1.8s infinite;
    }
    @keyframes pulse {
      0%,100% { opacity:1; } 50% { opacity:0.3; }
    }
    header h1 { font-size: 1rem; font-weight: 700; letter-spacing: 3px; color: var(--cyan); }
    header .subtitle { font-size: 0.75rem; color: var(--muted); margin-left: auto; font-family: 'JetBrains Mono', monospace; }

    /* ── Sidebar ── */
    aside {
      background: var(--surface);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      gap: 16px;
      padding: 20px 16px;
      overflow-y: auto;
    }
    .stat-card {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 14px 16px;
    }
    .stat-card label {
      font-size: 0.65rem;
      letter-spacing: 2px;
      color: var(--muted);
      text-transform: uppercase;
      display: block;
      margin-bottom: 6px;
    }
    .stat-card .value {
      font-family: 'JetBrains Mono', monospace;
      font-size: 1.6rem;
      font-weight: 700;
      color: var(--cyan);
    }
    .stat-card .value.green { color: var(--green); }
    .stat-card .value.pink  { color: var(--pink);  }
    .stat-card .value.small { font-size: 1rem; }

    .progress-outer {
      width: 100%;
      background: var(--border);
      border-radius: 99px;
      height: 6px;
      margin-top: 10px;
      overflow: hidden;
    }
    .progress-inner {
      height: 100%;
      border-radius: 99px;
      background: linear-gradient(90deg, var(--cyan), var(--purple));
      transition: width 0.4s ease;
    }

    .section-title {
      font-size: 0.65rem;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: var(--muted);
      padding: 0 4px;
    }

    .settings-row {
      display: flex; flex-direction: column; gap: 6px;
    }
    .settings-row label {
      font-size: 0.72rem;
      color: var(--muted);
    }
    .settings-row input[type=range] {
      width: 100%; accent-color: var(--cyan);
    }
    .settings-row .range-display {
      font-family: 'JetBrains Mono', monospace;
      font-size:0.8rem; color:var(--cyan);
      text-align: right;
    }

    /* ── Main chat area ── */
    main {
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    #chat-window {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      scroll-behavior: smooth;
    }

    .msg {
      max-width: 72%;
      padding: 14px 18px;
      border-radius: var(--radius);
      line-height: 1.6;
      font-size: 0.92rem;
      animation: msgIn 0.25s ease;
    }
    @keyframes msgIn {
      from { opacity:0; transform: translateY(10px); }
      to   { opacity:1; transform: translateY(0); }
    }
    .msg.user {
      align-self: flex-end;
      background: linear-gradient(135deg, #1a1f40, #0f1229);
      border: 1px solid var(--purple);
      color: #c8d0f0;
    }
    .msg.jarvis {
      align-self: flex-start;
      background: linear-gradient(135deg, #0a1a1f, #091520);
      border: 1px solid var(--cyan);
      color: var(--text);
    }
    .msg .sender {
      font-size: 0.65rem;
      letter-spacing: 2px;
      text-transform: uppercase;
      margin-bottom: 6px;
      font-weight: 700;
    }
    .msg.user .sender { color: var(--purple); }
    .msg.jarvis .sender { color: var(--cyan); }
    .msg pre {
      background: #020408;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.78rem;
      overflow-x: auto;
      margin-top: 8px;
      white-space: pre-wrap;
    }

    .thinking {
      align-self: flex-start;
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 0.82rem;
    }
    .dot-row span {
      display: inline-block;
      width: 6px; height: 6px;
      background: var(--cyan);
      border-radius: 50%;
      animation: dotBounce 1.2s infinite;
    }
    .dot-row span:nth-child(2) { animation-delay: 0.2s; }
    .dot-row span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes dotBounce {
      0%,80%,100% { transform: translateY(0); }
      40% { transform: translateY(-6px); }
    }

    /* ── Input bar ── */
    #input-bar {
      border-top: 1px solid var(--border);
      padding: 16px 24px;
      display: flex;
      gap: 12px;
      background: var(--surface);
    }
    #user-input {
      flex: 1;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      color: var(--text);
      font-family: 'Outfit', sans-serif;
      font-size: 0.92rem;
      padding: 12px 16px;
      outline: none;
      transition: border-color 0.2s;
      resize: none;
      height: 48px;
    }
    #user-input:focus { border-color: var(--cyan); }

    #send-btn {
      background: linear-gradient(135deg, var(--cyan), var(--purple));
      border: none;
      color: #000;
      font-weight: 700;
      font-family: 'Outfit', sans-serif;
      font-size: 0.9rem;
      padding: 12px 22px;
      border-radius: var(--radius);
      cursor: pointer;
      transition: opacity 0.2s, transform 0.1s;
    }
    #send-btn:hover  { opacity: 0.9; }
    #send-btn:active { transform: scale(0.97); }
    #send-btn:disabled { opacity: 0.4; cursor: not-allowed; }

    #chat-window::-webkit-scrollbar { width: 5px; }
    #chat-window::-webkit-scrollbar-track { background: transparent; }
    #chat-window::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
  </style>
</head>
<body>

<header>
  <div class="logo-dot"></div>
  <h1>JARVIS</h1>
  <span class="subtitle">DiM-LLM v3.1 &nbsp;|&nbsp; Mamba-Diffusion &nbsp;|&nbsp; 200M Params</span>
</header>

<!-- Sidebar -->
<aside>
  <div class="section-title">Generation Status</div>

  <div class="stat-card">
    <label>Time to Completion</label>
    <div class="value" id="ttc">—</div>
    <div class="progress-outer">
      <div class="progress-inner" id="progress-bar" style="width:0%"></div>
    </div>
  </div>

  <div class="stat-card">
    <label>Tokens / Second</label>
    <div class="value green" id="tps-display">—</div>
  </div>

  <div class="stat-card">
    <label>Diffusion Steps</label>
    <div class="value pink" id="steps-display">—</div>
  </div>

  <div class="stat-card">
    <label>Last Response Latency</label>
    <div class="value small" id="latency-display">—</div>
  </div>

  <div class="section-title" style="margin-top:8px">Training Monitor</div>

  <div class="stat-card">
    <label>Training TPS</label>
    <div class="value small" id="train-tps">Loading…</div>
  </div>

  <div class="stat-card">
    <label>Global Step</label>
    <div class="value small" id="train-step">—</div>
  </div>

  <hr style="border-color:var(--border)">
  <div class="section-title">Settings</div>
  <div class="settings-row">
    <label>Temperature</label>
    <input type="range" id="temp-slider" min="1" max="20" value="7">
    <div class="range-display" id="temp-val">0.7</div>
  </div>
  <div class="settings-row">
    <label>Diffusion Steps</label>
    <input type="range" id="steps-slider" min="8" max="64" value="32" step="8">
    <div class="range-display" id="steps-val">32</div>
  </div>
</aside>

<!-- Chat -->
<main>
  <div id="chat-window">
    <div class="msg jarvis">
      <div class="sender">Jarvis</div>
      Welcome back, Philip. DiM-LLM v3.1 Masked Diffusion engine is online. EMA weights loaded.
      How can I assist?
    </div>
  </div>

  <div id="input-bar">
    <textarea id="user-input" placeholder="Ask Jarvis something…" rows="1"></textarea>
    <button id="send-btn" onclick="sendMessage()">SEND</button>
  </div>
</main>

<script>
  // ── Slider handlers ──────────────────────
  const tempSlider  = document.getElementById('temp-slider');
  const stepsSlider = document.getElementById('steps-slider');
  tempSlider.addEventListener('input',  () => document.getElementById('temp-val').textContent  = (tempSlider.value  / 10).toFixed(1));
  stepsSlider.addEventListener('input', () => document.getElementById('steps-val').textContent = stepsSlider.value);

  // ── Training monitor + hot-reload notification ────
  let lastReloadCount = 0;

  async function refreshTrainStats() {
    try {
      const r = await fetch('/train_status').then(res => res.json());
      document.getElementById('train-tps').textContent  = r.tps  ? r.tps.toFixed(1) + ' TPS' : 'No data yet';
      document.getElementById('train-step').textContent = r.step ? 'Step ' + r.step  : '—';

      // Hot-reload notification
      if (r.reload_count > lastReloadCount) {
        lastReloadCount = r.reload_count;
        showReloadBanner(r.reload_msg);
      }
    } catch {}
  }

  function showReloadBanner(msg) {
    const banner = document.createElement('div');
    banner.style.cssText = `
      align-self: center;
      background: linear-gradient(90deg, #001a14, #002820);
      border: 1px solid #05ffa1;
      border-radius: 8px;
      padding: 8px 16px;
      font-size: 0.78rem;
      font-family: 'JetBrains Mono', monospace;
      color: #05ffa1;
      letter-spacing: 1px;
      animation: msgIn 0.3s ease;
      max-width: 90%;
      text-align: center;
    `;
    banner.textContent = '⚡ ' + msg;
    chatWindow.appendChild(banner);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    // Fade out after 8 seconds
    setTimeout(() => banner.style.opacity = '0.3', 7000);
  }

  refreshTrainStats();
  setInterval(refreshTrainStats, 5000);


  // ── Chat logic ────────────────────────────
  const chatWindow = document.getElementById('chat-window');
  const sendBtn    = document.getElementById('send-btn');

  function appendMsg(role, text) {
    const div = document.createElement('div');
    div.className = `msg ${role}`;
    const sender = document.createElement('div');
    sender.className = 'sender';
    sender.textContent = role === 'user' ? 'Philip' : 'Jarvis';
    div.appendChild(sender);
    // Basic code block detection
    if (text.includes('<tool') || text.includes('import ') || text.includes('<code>')) {
      const pre = document.createElement('pre');
      pre.textContent = text;
      div.appendChild(pre);
    } else {
      div.appendChild(document.createTextNode(text));
    }
    chatWindow.appendChild(div);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return div;
  }

  function showThinking() {
    const div = document.createElement('div');
    div.className = 'thinking';
    div.id = 'thinking-indicator';
    div.innerHTML = `<div class="dot-row"><span></span><span></span><span></span></div><span>Denoising sequence…</span>`;
    chatWindow.appendChild(div);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  function hideThinking() {
    const el = document.getElementById('thinking-indicator');
    if (el) el.remove();
  }

  async function sendMessage() {
    const input = document.getElementById('user-input');
    const text = input.value.trim();
    if (!text) return;

    appendMsg('user', text);
    input.value = '';
    sendBtn.disabled = true;
    showThinking();

    const temp  = tempSlider.value / 10;
    const steps = parseInt(stepsSlider.value);

    // Reset metrics
    document.getElementById('ttc').textContent          = '—';
    document.getElementById('tps-display').textContent  = '—';
    document.getElementById('steps-display').textContent = steps;
    document.getElementById('progress-bar').style.width = '0%';

    const tStart = Date.now();
    let tickInterval = setInterval(() => {
      const elapsed = ((Date.now() - tStart) / 1000).toFixed(1);
      // Estimate: assume ~40s per 32 steps
      const estTotal = (steps / 32) * 40;
      const pct = Math.min(99, ((Date.now() - tStart) / 1000) / estTotal * 100);
      document.getElementById('ttc').textContent = elapsed + 's';
      document.getElementById('progress-bar').style.width = pct.toFixed(0) + '%';
    }, 300);

    try {
      const resp = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: text, temperature: temp, steps: steps })
      });

      clearInterval(tickInterval);
      const data = await resp.json();
      hideThinking();

      const latency = ((Date.now() - tStart) / 1000).toFixed(2);
      const tps = (256 / parseFloat(latency)).toFixed(1);

      document.getElementById('ttc').textContent          = latency + 's';
      document.getElementById('progress-bar').style.width = '100%';
      document.getElementById('tps-display').textContent  = tps + ' T/s';
      document.getElementById('latency-display').textContent = latency + 's total';

      appendMsg('jarvis', data.response);
    } catch (e) {
      clearInterval(tickInterval);
      hideThinking();
      appendMsg('jarvis', '[ ERROR: Could not reach Jarvis engine. ]');
    }

    sendBtn.disabled = false;
  }

  // Enter key to send (Shift+Enter for newline)
  document.getElementById('user-input').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
</script>
</body>
</html>
"""


# ─── Flask Routes ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return UI_HTML


@app.route("/train_status")
def train_status():
    """Reads training_stats.json + hot-reload state. Never touches training process."""
    try:
        with open("training_stats.json", "r") as f:
            stats = json.load(f)
        return jsonify({
            "tps":         stats.get("tps", 0),
            "step":        stats.get("step", 0),
            "reload_count": _reload_state["reload_count"],
            "reload_msg":   _reload_state["last_reload_msg"]
        })
    except Exception:
        return jsonify({"tps": 0, "step": 0, "reload_count": 0, "reload_msg": ""})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    prompt_text = data.get("prompt", "")
    temperature = float(data.get("temperature", 0.7))
    steps = int(data.get("steps", 32))

    # Tokenize prompt
    prompt_ids = torch.tensor(
        tokenizer.encode(f"User: {prompt_text}\nAssistant: ", add_special_tokens=False),
        dtype=torch.long, device=DEVICE
    ).unsqueeze(0)

    if prompt_ids.shape[1] >= config.seq_len - 10:
        prompt_ids = prompt_ids[:, -(config.seq_len // 2):]

    with torch.no_grad():
        output_ids = engine.sample(
            n_samples=1,
            steps=steps,
            prompt_ids=prompt_ids,
            temperature=temperature
        )

    full_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
    prompt_decoded = tokenizer.decode(prompt_ids[0].tolist(), skip_special_tokens=True)
    response = full_text[len(prompt_decoded):].strip()

    if not response:
        response = "[No output generated. Try increasing steps or adjusting temperature.]"

    return jsonify({"response": response})


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  JARVIS Web UI  →  http://localhost:5000")
    print("  Training continues unaffected in background.")
    print("="*55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
