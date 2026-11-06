import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import modal

def main():
    fn = modal.Function.from_name("staf-v2-audit-suite", "run_full_audit_and_eval")
    print("Spawning run_full_audit_and_eval()...", flush=True)
    call = fn.spawn()
    print(f"Spawned audit call successfully! Call ID: {call.object_id}", flush=True)

if __name__ == "__main__":
    main()

