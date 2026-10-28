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
def download_with_gdown(gdrive_id: str, dataset_name: str, is_folder: bool = True):
    """
    Download a dataset archive using Google Drive ID via gdown.
    """
    import subprocess
    dest_dir = f"/data/{dataset_name}"
    os.makedirs(dest_dir, exist_ok=True)

    print(f">> Downloading {dataset_name} using gdown from ID: {gdrive_id}...")
    if is_folder:
        cmd = f"gdown --folder https://drive.google.com/drive/folders/{gdrive_id} -O {dest_dir}"
    else:
        cmd = f"gdown https://drive.google.com/uc?id={gdrive_id} -O {dest_dir}/archive.zip"

    subprocess.run(cmd, shell=True, check=True)

    # If it was a zip/tar file, extract it
    if not is_folder:
        print(f">> Extracting archive inside {dest_dir}...")
        subprocess.run(f"unzip -q -o {dest_dir}/archive.zip -d {dest_dir}", shell=True)

    data_volume.commit()
    print(f">> Successfully downloaded & committed {dataset_name} to volume!")

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
