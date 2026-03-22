import os
import json
import psutil
import subprocess
from flask import Flask, render_template, jsonify

app = Flask(__name__)
STATS_FILE = "/home/phil/Desktop/mambadiff/mambadiff llm tts/training_stats.json"

def get_base_stats():
    if not os.path.exists(STATS_FILE):
        return None
    
    try:
        with open(STATS_FILE, "r") as f:
            stats = json.load(f)
            
        # 1. System RAM
        ram = psutil.virtual_memory()
        stats["sys_ram"] = {
            "used": f"{ram.used / (1024**3):.1f} GB",
            "total": f"{ram.total / (1024**3):.1f} GB",
            "percent": ram.percent
        }
        
        # 2. GPU VRAM & Load
        try:
            cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits"
            res = subprocess.check_output(cmd.split()).decode("utf-8").strip()
            gpu_load, vram_used, vram_total, temp = [x.strip() for x in res.split(",")]
            stats["gpu"] = {
                "load": f"{gpu_load}%",
                "vram_used": f"{float(vram_used)/1024:.1f} GB",
                "vram_total": f"{float(vram_total)/1024:.1f} GB",
                "vram_percent": f"{float(vram_used)/float(vram_total)*100:.1f}%",
                "temp": f"{temp}°C"
            }
        except:
            stats["gpu"] = {"load": "N/A", "vram_used": "N/A", "vram_total": "N/A", "vram_percent": "N/A", "temp": "N/A"}

        # 3. CPU Temperature
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                max_cpu = max(t.current for t in temps['coretemp'])
                stats["cpu_temp"] = f"{max_cpu:.1f}°C"
            else:
                stats["cpu_temp"] = "N/A"
        except:
            stats["cpu_temp"] = "N/A"

        return stats
    except Exception as e:
        print(f"Error reading stats: {e}")
        return None

@app.route("/")
def index():
    return render_template("monitor.html")

@app.route("/api/stats")
def api_stats():
    stats = get_base_stats()
    if stats:
        return jsonify(stats)
    return jsonify({"error": "Failed to retrieve stats"}), 500

if __name__ == "__main__":
    print("🚀 DiM-LLM Training Monitor starting on http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=False)
