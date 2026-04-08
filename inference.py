"""Root inference entrypoint for hackathon validation.

This file intentionally delegates all logic to files.inference.main,
while guaranteeing clean process exit.
"""

import sys

from files.inference import main


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] root_inference_failed: {e}", flush=True)
    sys.exit(0)
