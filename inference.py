"""Root inference entrypoint for hackathon validation.
Delegates execution to files.inference while exposing required env/client interface.
"""

import os
from openai import OpenAI

# Required environment variable interface for hackathon checks
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "HuggingFaceH4/zephyr-7b-beta")
HF_TOKEN = os.getenv("HF_TOKEN", "hf_xxxxxxxxxxxxxxxxx")

client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url=os.getenv("API_BASE_URL"),
)

# Keep files.inference compatibility by ensuring defaults are present
os.environ.setdefault("API_BASE_URL", API_BASE_URL)
os.environ.setdefault("MODEL_NAME", MODEL_NAME)
os.environ.setdefault("HF_TOKEN", HF_TOKEN)
os.environ.setdefault("ENV_URL", os.getenv("ENV_URL", "http://localhost:7860"))

from files.inference import main


if __name__ == "__main__":
    raise SystemExit(main())
