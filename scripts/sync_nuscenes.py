"""
Dedicated nuScenes Ingestion & Volume Sync for STAF Framework
Creates a dedicated Modal Volume ('staf-nuscenes') with 500,000 fresh inodes
and transfers nuScenes archives directly from Google Drive via rclone.
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

app = modal.App("staf-nuscenes-sync")

# Dedicated persistent volume for nuScenes (isolated 500k inode quota)
nuscenes_volume = modal.Volume.from_name("staf-nuscenes", create_if_missing=True)

# Container equipped with rclone and tar/gzip
sync_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("rclone", "tar", "gzip", "unzip")
)

@app.function(
    image=sync_image,
    volumes={"/data/nuscenes": nuscenes_volume},
    timeout=14400,
    cpu=4.0,
    memory=16384,
)
def sync_nuscenes(rclone_conf_str: str, use_mini: bool = True):
    import subprocess
    import os

    # Install rclone config
    rclone_dir = "/root/.config/rclone"
    os.makedirs(rclone_dir, exist_ok=True)
    with open(os.path.join(rclone_dir, "rclone.conf"), "w") as f:
        f.write(rclone_conf_str)

    os.makedirs("/data/nuscenes", exist_ok=True)
    tmp_archive_dir = "/tmp/nuscenes_archives"
    os.makedirs(tmp_archive_dir, exist_ok=True)

    print("=" * 60)
    print(">> SYNCING NUSCENES TO DEDICATED VOLUME (staf-nuscenes)")
    print("=" * 60)

    if use_mini:
        # Mini set for rapid verification + full training on mini subset
        targets = [
            ("can_bus.zip", "zip"),
            ("nuScenes-map-expansion-v1.3.zip", "zip"),
            ("v1.0-mini.tgz", "tar"),
        ]
    else:
        # Full trainval
        targets = [
            ("can_bus.zip", "zip"),
            ("nuScenes-map-expansion-v1.3.zip", "zip"),
            ("v1.0-trainval_meta.tgz", "tar"),
            ("v1.0-trainval01_blobs.tgz", "tar"),
            ("v1.0-trainval02_blobs.tgz", "tar"),
            ("v1.0-trainval03_blobs.tgz", "tar"),
            ("v1.0-trainval04_blobs.tgz", "tar"),
            ("v1.0-trainval05_blobs.tgz", "tar"),
            ("v1.0-trainval06_blobs.tgz", "tar"),
            ("v1.0-trainval07_blobs.tgz", "tar"),
            ("v1.0-trainval08_blobs.tgz", "tar"),
            ("v1.0-trainval09_blobs.tgz", "tar"),
            ("v1.0-trainval10_blobs.tgz", "tar"),
        ]

    for fname, ftype in targets:
        remote_file = f"gdrive:Thesis_Datasets/nuscenes/{fname}"
        local_archive = os.path.join(tmp_archive_dir, fname)

        print(f"\n>> Downloading {fname}...")
        cmd_copy = ["rclone", "copyto", remote_file, local_archive, "--progress", "--drive-chunk-size", "128M"]
        res = subprocess.run(cmd_copy)
        if res.returncode != 0:
            print(f"[ERROR] Failed to download {fname}")
            continue

        print(f">> Extracting {fname} into /data/nuscenes...")
        if ftype == "tar":
            subprocess.run(f"tar xf {local_archive} -C /data/nuscenes", shell=True)
        elif ftype == "zip":
            subprocess.run(f"unzip -q -o {local_archive} -d /data/nuscenes", shell=True)

        # Immediately remove archive from /tmp to keep scratch space clean
        if os.path.exists(local_archive):
            os.remove(local_archive)

    print("\n>> Committing changes to dedicated staf-nuscenes volume...")
    nuscenes_volume.commit()
    print(">> staf-nuscenes volume successfully committed!")

    subprocess.run(["df", "-h", "/data/nuscenes"])
    subprocess.run(["df", "-i", "/data/nuscenes"])
    subprocess.run(["ls", "-lah", "/data/nuscenes"])

@app.local_entrypoint()
def main(mini: bool = True):
    rclone_conf_path = os.path.expanduser(os.path.join(os.environ.get("APPDATA", ""), "rclone", "rclone.conf"))
    if not os.path.exists(rclone_conf_path):
        rclone_conf_path = os.path.expanduser("~/.config/rclone/rclone.conf")
    with open(rclone_conf_path, "r") as f:
        conf_str = f.read()

    sync_nuscenes.remote(rclone_conf_str=conf_str, use_mini=mini)

