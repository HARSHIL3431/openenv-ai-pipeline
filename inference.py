"""Root inference entrypoint for hackathon validation.
Delegates execution to files.inference while exposing required env/client interface.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Required environment variable interface for hackathon checks
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "files", ".env"))

API_BASE_URL = os.environ["API_BASE_URL"]
model = os.environ.get("MODEL_NAME", "gpt-4o-mini")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=os.environ["API_KEY"],
)

# Keep files.inference compatibility by ensuring defaults are present
os.environ.setdefault("API_BASE_URL", API_BASE_URL)
os.environ.setdefault("MODEL_NAME", model)
os.environ.setdefault("ENV_URL", os.getenv("ENV_URL", "http://localhost:7860"))

from files.inference import run_task


if __name__ == "__main__":
    run_task("easy_null_fix")
