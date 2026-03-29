---
title: OpenEnv AI Pipeline
emoji: 🤖
colorFrom: red
colorTo: pink
sdk: docker
pinned: false
license: mit
short_description: AI data pipeline repair environment
---
# OpenEnv AI Pipeline

AI-powered environment for **automated data pipeline repair**, built using OpenEnv standards.
The system simulates real-world data engineering issues and allows an AI agent to detect and fix them step-by-step.

---

## 🚀 Overview

OpenEnv AI Pipeline provides a structured environment where:

* Data pipelines contain real-world issues (nulls, duplicates, type errors)
* An AI agent interacts using:

  * `reset()`
  * `step()`
  * `state()`
* The agent identifies and repairs issues
* A deterministic grader evaluates performance (0.0 → 1.0)

---

## 🎯 Key Features

* 🧠 AI Agent-driven pipeline repair
* 🔄 OpenEnv-compliant environment design
* 📊 Deterministic scoring system
* ⚙️ FastAPI-based API interface
* 🐳 Fully Dockerized deployment
* 🌐 Hugging Face Spaces ready

---

## 🧱 Tech Stack

* **Backend:** FastAPI, Pydantic
* **AI Integration:** OpenAI-compatible client (HF Inference)
* **Environment:** OpenEnv framework
* **Deployment:** Docker, Hugging Face Spaces
* **Language:** Python 3.12

---

## 📁 Project Structure

```text
OpenEnv/
├── files/
│   ├── main.py          # FastAPI app
│   ├── environment.py   # Core environment logic
│   ├── models.py        # Pydantic schemas
│   ├── registry.py      # Tasks + graders
│   ├── inference.py     # Agent logic
│   ├── demo.py          # Rule-based agent
│   ├── test_env.py      # Test cases
│   ├── requirements.txt
│   └── __init__.py
├── inference.py         # Root inference (hackathon requirement)
├── Dockerfile
├── docker-compose.yml
├── openenv.yaml
├── .gitignore
└── README.md
```

---

## ⚙️ Setup (Local)

```bash
git clone <your-repo-url>
cd OpenEnv

python -m venv .venv
.venv\Scripts\activate

pip install -r files/requirements.txt
```

---

## 🔐 Environment Setup

Create `.env` from `.env.example`:

```bash
copy files/.env.example files/.env
```

Example:

```env
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=HuggingFaceH4/zephyr-7b-beta
HF_TOKEN=your_token_here
ENV_URL=http://localhost:7860
```

---

## ▶️ Run Locally

```bash
python -m uvicorn files.main:app --reload
```

Open:

* API Docs: http://127.0.0.1:8000/docs

---

## 🐳 Run with Docker

```bash
docker build -t openenv-ai .
docker run --env-file files/.env -p 7860:7860 openenv-ai
```

---

## 🧪 API Endpoints

* `POST /reset` → Initialize environment
* `POST /step` → Perform action
* `GET /state` → Get current state

---

## 🤖 Inference

Run agent:

```bash
python inference.py
```

---

## 📊 Evaluation

* Deterministic grading system
* Score range: **0.0 → 1.0**
* Metrics:

  * Issue identification
  * Correct fixes
  * Pipeline validation

---

## 🧪 Testing

```bash
python files/test_env.py
```

---

## 🌐 Deployment

Deployed on Hugging Face Spaces (Docker):

* Uses environment secrets:

  * `HF_TOKEN`
  * `API_BASE_URL`
  * `MODEL_NAME`

---

## ⚠️ Notes

* `.env` is excluded from repository for security
* Fallback logic ensures system works even if LLM fails
* Designed for robustness and reproducibility

---

## 🏁 Hackathon Compliance

* ✅ OpenEnv-compatible
* ✅ Required endpoints implemented
* ✅ Root `inference.py` included
* ✅ Dockerized deployment
* ✅ Deterministic scoring

---

## 📌 Future Improvements

* Multi-step reasoning agents
* Reinforcement learning integration
* More complex pipeline scenarios
* UI dashboard for visualization

---

## 👨‍💻 Author

Harshil Thakkar

---

## 📄 License

MIT License

---

## 🔗 References

* https://huggingface.co/docs/hub/spaces-config-reference
* OpenEnv Framework
