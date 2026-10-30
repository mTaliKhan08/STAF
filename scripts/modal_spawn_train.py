"""
Trigger detached background training on the deployed Modal app.
This launches the training container in Modal's cloud and exits immediately (0s wait).
The cloud job runs independently on NVIDIA A10G and CANNOT be interrupted by local
network blips, sleep mode, or laptop shutdown.
"""
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import modal

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", default="staf", choices=["staf", "transfuser", "diffuser"])
    parser.add_argument("--dataset", default="all", choices=["all", "carla", "nuscenes"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    fn_name = f"train_{args.job}"
    print(f">> Connecting to deployed Modal function: staf-thesis-training :: {fn_name}...")
    fn = modal.Function.from_name("staf-thesis-training", fn_name)

    if args.job == "staf":
        call = fn.spawn(
            dataset_name=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
    else:
        call = fn.spawn(epochs=args.epochs, batch_size=args.batch_size)

    print(f"============================================================")
    print(f">> Successfully spawned cloud job!")
    print(f">> Function Call ID: {call.object_id}")
    print(f">> Job: {args.job.upper()} | Dataset: {args.dataset} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print(f">> View real-time logs & status at:")
    print(f"   https://modal.com/apps/mtalikhan08/main/deployed/staf-thesis-training")
    print(f">> Cloud worker is now running independently on NVIDIA A10G.")
    print(f">> Your laptop can now be closed or shut down safely.")
    print(f"============================================================")

if __name__ == "__main__":
    main()
