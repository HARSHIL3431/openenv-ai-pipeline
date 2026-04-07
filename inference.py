"""Root inference entrypoint for hackathon validation.
Delegates execution to files.inference while exposing required env/client interface.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Required environment variable interface for hackathon checks
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "files", ".env"))

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("Missing required environment variable: HF_TOKEN")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# Keep files.inference compatibility by ensuring defaults are present
os.environ.setdefault("API_BASE_URL", API_BASE_URL)
os.environ.setdefault("MODEL_NAME", MODEL_NAME)
os.environ.setdefault("HF_TOKEN", HF_TOKEN)
os.environ.setdefault("ENV_URL", os.getenv("ENV_URL", "http://localhost:7860"))

from files.inference import run_task


if __name__ == "__main__":
    run_task("easy_null_fix")
