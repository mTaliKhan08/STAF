import os
import re
import csv
import json

log_path = "checkpoints/staf/training_50epochs.log"
out_dir = "logs_and_metrics/original_staf_training_progression"
os.makedirs(out_dir, exist_ok=True)

# Also copy raw log into out_dir
raw_out_log = os.path.join(out_dir, "staf_original_50epochs.log")
with open(log_path, "r", encoding="utf-8", errors="ignore") as f_in, open(raw_out_log, "w", encoding="utf-8") as f_out:
    f_out.write(f_in.read())

epoch_pattern = re.compile(r"Epoch\s+(\d+)/(\d+)\s+\(([\d\.]+)s\)\s+\|\s+Train:\s+([\d\.]+)\s+\|\s+Val:\s+([\d\.]+)\s+\|\s+LR:\s+([\d\.e\+\-]+)")
gate_pattern = re.compile(r"Epoch\s+(\d+)\s+\[\d+/\d+\]\s+Loss:\s+[\d\.]+\s+w_cam:\s+([\d\.]+)\s+w_lidar:\s+([\d\.]+)")

epochs_data = []
current_gate_cam = []
current_gate_lidar = []

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        # Check gate weights
        m_gate = gate_pattern.search(line)
        if m_gate:
            current_gate_cam.append(float(m_gate.group(2)))
            current_gate_lidar.append(float(m_gate.group(3)))

        # Check epoch summary
        m_ep = epoch_pattern.search(line)
        if m_ep:
            ep = int(m_ep.group(1))
            total_ep = int(m_ep.group(2))
            duration = float(m_ep.group(3))
            train_loss = float(m_ep.group(4))
            val_loss = float(m_ep.group(5))
            lr = float(m_ep.group(6))
            
            mean_cam = sum(current_gate_cam) / len(current_gate_cam) if current_gate_cam else None
            mean_lidar = sum(current_gate_lidar) / len(current_gate_lidar) if current_gate_lidar else None

            epochs_data.append({
                "epoch": ep,
                "total_epochs": total_ep,
                "duration_seconds": duration,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
                "mean_w_cam": round(mean_cam, 4) if mean_cam else None,
                "mean_w_lidar": round(mean_lidar, 4) if mean_lidar else None,
            })
            current_gate_cam = []
            current_gate_lidar = []

print(f"Extracted {len(epochs_data)} epochs from original STAF 50-epoch log.")

# Save CSV and JSON
csv_path = os.path.join(out_dir, "staf_original_50epochs_progression.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(epochs_data[0].keys()))
    writer.writeheader()
    writer.writerows(epochs_data)

json_path = os.path.join(out_dir, "staf_original_50epochs_progression.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(epochs_data, f, indent=2)

print(f"Saved: {csv_path} and {json_path}")

