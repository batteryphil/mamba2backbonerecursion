import json

with open("training_stats.json", "r") as f:
    stats = json.load(f)

# The array contains: [epoch1, epoch2, epoch3, bad_epoch4, new_epoch1, ...]
# We want to remove index 3 (bad_epoch4)
if len(stats["val_loss"]) > 3:
    stats["train_loss"].pop(3)
    stats["val_loss"].pop(3)
    stats["salads"].pop(3)

with open("training_stats.json", "w") as f:
    json.dump(stats, f)

print("Removed bad Epoch 4 from training_stats.json")
