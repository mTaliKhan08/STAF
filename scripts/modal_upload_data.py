"""
Modal Dataset Ingestion & Google Drive Bridge for STAF Framework
Thesis: Multi-Modal Fusion Framework for Autonomous Robot Navigation (MS-DS)

Pulls compressed datasets from Google Drive directly into the persistent
Modal Volume ('staf-datasets') using high-speed cloud-to-cloud bandwidth (60-120 MB/s),
then extracts archives directly onto the persistent volume mount (/data).

Supported Datasets:
  - nuScenes v1.0 Full (~320 GB compressed)  -> /data/nuscenes
  - CARLA TransFuser (~120 GB compressed)    -> /data/carla_transfuser
  - EuRoC MAV (~27 GB compressed)            -> /data/euroc_mav

Usage:
    # 1. Check current volume status & disk usage
    modal run scripts/modal_upload_data.py --action status

    # 2. Sync from Google Drive using your local rclone.conf
    modal run scripts/modal_upload_data.py --action rclone --rclone-conf-path ~/.config/rclone/rclone.conf

    # 3. Alternative: Download a specific dataset using a Google Drive Folder ID / Link via gdown
    modal run scripts/modal_upload_data.py --action gdown --gdrive-id "FOLDER_OR_FILE_ID" --dataset euroc
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import modal

app = modal.App("staf-data-pipeline")

# Persistent volume for all raw and processed training datasets
data_volume = modal.Volume.from_name("staf-datasets", create_if_missing=True)

# Cloud container equipped with rclone, unzip, tar, and gdown
data_engine_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("rclone", "unzip", "tar", "gzip", "curl", "tree")
    .pip_install("gdown")
)

@app.function(
    image=data_engine_image,
    volumes={"/data": data_volume},
    timeout=14400,  # 4 hours max for high-speed cloud data transfer
    cpu=4.0,
    memory=16384,   # 16 GB RAM for fast decompressing
)
def inspect_volume():
    """Inspect the current contents and disk usage of the persistent Modal volume."""
    import subprocess
    print("=" * 60)
    print(">> PERSISTENT MODAL VOLUME STATUS: /data")
    print("=" * 60)
    res = subprocess.run(["df", "-h", "/data"], capture_output=True, text=True)
    print(res.stdout)

    print("\nDirectory listing:")
    res2 = subprocess.run(["ls", "-lh", "/data"], capture_output=True, text=True)
    print(res2.stdout)

    for sub in ["nuscenes", "carla_transfuser", "euroc_mav"]:
        sub_path = f"/data/{sub}"
        if os.path.exists(sub_path):
            file_count = sum([len(files) for _, _, files in os.walk(sub_path)])
            print(f"  - {sub}: Exists ({file_count} files)")
        else:
            print(f"  - {sub}: [NOT FOUND]")

@app.function(
    image=data_engine_image,
    volumes={"/data": data_volume},
    timeout=14400,
    cpu=4.0,
    memory=16384,
)
def sync_with_rclone(rclone_conf_str: str, datasets: list):
    """
    Sync datasets from Google Drive using rclone and extract on the volume.
    """
    import subprocess

    # Write rclone config inside the remote container
    rclone_dir = "/root/.config/rclone"
    os.makedirs(rclone_dir, exist_ok=True)
    conf_file = os.path.join(rclone_dir, "rclone.conf")
    with open(conf_file, "w") as f:
        f.write(rclone_conf_str)

    print(">> rclone configuration installed successfully.")

    # Datasets map: (gdrive_subpath, volume_subpath, file_types)
    dataset_targets = {
        "nuscenes": ("gdrive:Thesis_Datasets/nuscenes/", "/data/nuscenes", "tar"),
        "carla": ("gdrive:Thesis_Datasets/carla_transfuser/", "/data/carla_transfuser", "zip"),
        "euroc": ("gdrive:Thesis_Datasets/euroc_mav/", "/data/euroc_mav", "zip"),
    }

    raw_download_dir = "/data/raw_archives"
    os.makedirs(raw_download_dir, exist_ok=True)

    for ds_key in datasets:
        if ds_key not in dataset_targets:
            print(f"[WARN] Unknown dataset: {ds_key}, skipping.")
            continue

        remote_src, target_dir, archive_type = dataset_targets[ds_key]
        dest_archive_dir = os.path.join(raw_download_dir, ds_key)
        os.makedirs(dest_archive_dir, exist_ok=True)
        os.makedirs(target_dir, exist_ok=True)

        print(f"\n========================================================")
        print(f">> Syncing {ds_key.upper()} from {remote_src}...")
        print(f"========================================================")

        cmd = [
            "rclone", "copy", remote_src, dest_archive_dir,
            "--progress",
            "--transfers", "8",
            "--checkers", "16",
            "--drive-chunk-size", "128M"
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"[ERROR] rclone failed for {ds_key}")
            continue

        print(f">> Extracting {ds_key.upper()} archives into {target_dir}...")
        if archive_type == "tar":
            # Extract tar/tgz/tar.gz files
            subprocess.run(
                f"for f in {dest_archive_dir}/*.tar* {dest_archive_dir}/*.tgz; do "
                f"[ -f \"$f\" ] && echo \"Extracting $f...\" && tar xf \"$f\" -C {target_dir}; "
                f"done",
                shell=True
            )
        elif archive_type == "zip":
            subprocess.run(
                f"for f in {dest_archive_dir}/*.zip; do "
                f"[ -f \"$f\" ] && echo \"Extracting $f...\" && unzip -q -o \"$f\" -d {target_dir}; "
                f"done",
                shell=True
            )

        print(f"[PASS] {ds_key.upper()} ready at {target_dir}")

    # Commit persistent changes to the Modal volume
    print(">> Committing changes to Modal persistent volume...")
    data_volume.commit()
    print(">> Data volume successfully committed and permanently saved.")

@app.function(
    image=data_engine_image,
    volumes={"/data": data_volume},
    timeout=14400,
    cpu=4.0,
    memory=16384,
)
def download_with_gdown(gdrive_id: str, dataset_name: str = "Thesis_Datasets", is_folder: bool = True):
    """
    Download dataset archives from Google Drive using gdown and extract them onto the persistent volume.
    """
    import subprocess
    import os

    print(f">> Starting Google Drive ingestion from ID: {gdrive_id}...")

    # Download to /data
    if is_folder:
        cmd = f"gdown --folder https://drive.google.com/drive/folders/{gdrive_id} -O /data --continue"
    else:
        dest_file = f"/data/{dataset_name}.zip"
        cmd = f"gdown https://drive.google.com/uc?id={gdrive_id} -O {dest_file}"

    print(f"Executing: {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"[WARN] gdown exited with code {res.returncode}, continuing to scan and extraction...")

    # Look for any downloaded zip or tar files in /data and subdirectories
    print("\n>> Scanning for downloaded archives to unpack...")
    subprocess.run("find /data -maxdepth 4 -type f \\( -name '*.zip' -o -name '*.tar*' -o -name '*.tgz' \\) -exec ls -lh {} \\;", shell=True)

    # 1. Unpack any .zip archives in-place
    print("\n>> Unpacking .zip files...")
    subprocess.run(
        "find /data -name '*.zip' | while read -r zipfile; do "
        "  dir=$(dirname \"$zipfile\"); "
        "  echo \"Unzipping $zipfile into $dir...\"; "
        "  unzip -q -o \"$zipfile\" -d \"$dir\"; "
        "done",
        shell=True
    )

    # 2. Unpack any .tar / .tgz archives in-place
    print("\n>> Unpacking .tar / .tgz files...")
    subprocess.run(
        "find /data -name '*.tar*' -o -name '*.tgz' | while read -r tarfile; do "
        "  dir=$(dirname \"$tarfile\"); "
        "  echo \"Extracting $tarfile into $dir...\"; "
        "  tar xf \"$tarfile\" -C \"$dir\"; "
        "done",
        shell=True
    )

    # Commit persistent changes to the Modal volume
    print("\n>> Committing changes to Modal persistent volume...")
    data_volume.commit()
    print(">> Data volume committed successfully!")

    # Print final directory structure
    print("\n============================================================")
    print(">> FINAL PERSISTENT VOLUME CONTENTS:")
    print("============================================================")
    subprocess.run(["df", "-h", "/data"])
    subprocess.run("ls -la /data", shell=True)

@app.local_entrypoint()
def main(
    action: str = "status",
    rclone_conf_path: str = "",
    dataset: str = "all",
    gdrive_id: str = "",
):
    """
    CLI local entrypoint.
    Examples:
        modal run scripts/modal_upload_data.py --action status
        modal run scripts/modal_upload_data.py --action rclone --rclone-conf-path ~/.config/rclone/rclone.conf
        modal run scripts/modal_upload_data.py --action gdown --gdrive-id "FOLDER_ID" --dataset euroc
    """
    if action == "status":
        inspect_volume.remote()

    elif action == "rclone":
        if not rclone_conf_path:
            # Check default locations
            candidates = [
                os.path.expanduser("~/.config/rclone/rclone.conf"),
                os.path.expanduser("~/.rclone.conf"),
                os.path.join(os.environ.get("APPDATA", ""), "rclone", "rclone.conf"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    rclone_conf_path = c
                    break

        if not rclone_conf_path or not os.path.exists(rclone_conf_path):
            print(f"[ERROR] rclone.conf not found! Please specify --rclone-conf-path <path_to_rclone.conf>")
            print("Tip: If you already configured rclone on another machine or pod, paste the config file path.")
            return

        with open(rclone_conf_path, "r") as f:
            conf_str = f.read()

        ds_list = ["nuscenes", "carla", "euroc"] if dataset == "all" else [dataset]
        print(f"Launching rclone sync on Modal for datasets: {ds_list}...")
        sync_with_rclone.remote(conf_str, ds_list)

    elif action == "gdown":
        if not gdrive_id:
            print("[ERROR] Please provide --gdrive-id <google_drive_folder_or_file_id>")
            return
        download_with_gdown.remote(gdrive_id, dataset)

    else:
        print(f"Unknown action: {action}. Choose from: status, rclone, gdown")
