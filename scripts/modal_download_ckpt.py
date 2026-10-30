"""
Download trained STAF checkpoints from Modal persistent volume to local machine.

Usage:
    modal run scripts/modal_download_ckpt.py
    modal run scripts/modal_download_ckpt.py --filename checkpoint_epoch_17.pth
"""
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import modal

app = modal.App("staf-download-checkpoints")
ckpt_volume = modal.Volume.from_name("staf-checkpoints")
image = modal.Image.debian_slim(python_version="3.11")

@app.function(image=image, volumes={"/checkpoints": ckpt_volume}, timeout=600)
def get_checkpoint_bytes(filename: str = "best_model.pth") -> bytes:
    ckpt_path = os.path.join("/checkpoints/staf", filename)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Reading {ckpt_path} ({os.path.getsize(ckpt_path) / 1024 / 1024:.2f} MB)...")
    with open(ckpt_path, "rb") as f:
        return f.read()

@app.local_entrypoint()
def main(filename: str = "best_model.pth", out_dir: str = "checkpoints/staf"):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    print(f">> Downloading {filename} from Modal volume to {out_path}...")
    data = get_checkpoint_bytes.remote(filename=filename)
    with open(out_path, "wb") as f:
        f.write(data)
    print(f">> Successfully downloaded {filename} to {out_path} ({len(data) / 1024 / 1024:.2f} MB)!")

if __name__ == "__main__":
    main()
