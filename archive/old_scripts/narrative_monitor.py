import json
import os
import time
from collections import Counter

# Markers for "Narrative Overfitting" / "Story-Mode Trap"
NARRATIVE_MARKERS = [
    "one day", "importance of", "value of the", "there lived", 
    "many years ago", "lessons learned", "empathy", "curiosity",
    "named", "friends named", "adventure", "journey", "once upon"
]

def calculate_bias(text):
    text_lower = text.lower()
    score = 0
    for marker in NARRATIVE_MARKERS:
        score += text_lower.count(marker)
    return score

def run_narrative_monitor():
    stats_file = "training_stats.json"
    print("🕵️ Narrative Bias Monitor Active...")
    print(f"Tracking markers: {', '.join(NARRATIVE_MARKERS[:5])}...")
    print("-" * 50)
    
    last_step = -1
    
    while True:
        if not os.path.exists(stats_file):
            time.sleep(10)
            continue
            
        try:
            with open(stats_file, "r") as f:
                stats = json.load(f)
        except:
            time.sleep(5)
            continue
            
        current_step = stats.get("step", 0)
        salads = stats.get("salads", [])
        
        if current_step != last_step and salads:
            last_step = current_step
            
            # Get latest probe response
            latest_batch = salads[-1]
            if isinstance(latest_batch, list) and len(latest_batch) > 0:
                response = latest_batch[0].get("response", "")
                bias_score = calculate_bias(response)
                
                # Calculate Trend (last 5 probes)
                recent_scores = []
                for b in salads[-5:]:
                    if isinstance(b, list) and len(b) > 0:
                        recent_scores.append(calculate_bias(b[0].get("response", "")))
                
                avg_bias = sum(recent_scores) / len(recent_scores) if recent_scores else 0
                
                print(f"Step: {current_step:6d} | Current Bias: {bias_score:2d} | Avg (5): {avg_bias:.1f} | Loss: {stats.get('loss', 0):.4f}")
                
                if avg_bias > 3:
                    print("⚠️  STATUS: HIGH NARRATIVE BIAS. Hold at N=1.")
                elif avg_bias > 1:
                    print("🔄 STATUS: NARRATIVE DECAYING. Approaching Snap...")
                else:
                    print("✅ STATUS: NEUTRAL MANIFOLD. Reasonably safe for N=2.")
            
        time.sleep(30)

if __name__ == "__main__":
    run_narrative_monitor()
