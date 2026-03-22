import json
import time
import os

# --- Automated Training Monitor ---
# This script monitors the training process from training_stats.json
# and ensures the model is not overfitting or diverging.

def monitor():
    stats_file = "training_stats.json"
    print("Monitoring Mamba-Diffusion Training...")
    
    last_val_loss = float('inf')
    overfit_counter = 0

    while True:
        if not os.path.exists(stats_file):
            print("Waiting for training_stats.json to be created...")
            time.sleep(5)
            continue
            
        try:
            with open(stats_file, "r") as f:
                stats = json.load(f)
        except (json.JSONDecodeError, PermissionError):
            time.sleep(1) # File might be being written
            continue

        if not stats.get("train_loss"):
            time.sleep(5)
            continue

        latest_train = stats["train_loss"][-1]
        latest_val = stats["val_loss"][-1] if stats.get("val_loss") else latest_train
        
        print(f"Monitor -> Step: {stats['step']} | Train: {latest_train:.4f} | Val: {latest_val:.4f}")
        
        # 🧪 Overfitting Detection Logic
        if latest_val > last_val_loss:
            overfit_counter += 1
            print(f"⚠️ Warning: Validation loss increased ({overfit_counter}/5)!")
        else:
            overfit_counter = 0 # reset
            
        if overfit_counter >= 5:
            print("🚨 ALERT: Model is significantly overfitting! Recommendation: Lower LR or stop.")
            
        if latest_train > 10.0 and stats["step"] > 1000:
             print("🚨 ALERT: Training loss is exceptionally high. Divergence suspected.")

        last_val_loss = latest_val
        time.sleep(30) # Monitor every 30 seconds

if __name__ == "__main__":
    monitor()
