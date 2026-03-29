.PHONY: install run test docker-build docker-run docker-demo inference clean

# ─── Local dev ───────────────────────────────────────────
install:
	pip install -r requirements.txt

run:
	python -m uvicorn files.main:app --host 0.0.0.0 --port 7860 --reload

test:
	python -m files.test_env

# ─── Docker ──────────────────────────────────────────────
docker-build:
	docker build -t pipeline-repair-env .

docker-run:
	docker run -p 7860:7860 pipeline-repair-env

docker-demo:
	docker compose --profile demo up --build

# ─── Inference (requires HF_TOKEN, MODEL_NAME, API_BASE_URL) ─
inference:
	@echo "Running LLM inference agent..."
	@echo "  ENV_URL:      $(or $(ENV_URL),http://localhost:7860)"
	@echo "  API_BASE_URL: $(or $(API_BASE_URL),not set)"
	@echo "  MODEL_NAME:   $(or $(MODEL_NAME),not set)"
	ENV_URL=$(or $(ENV_URL),http://localhost:7860) python inference.py

# ─── HuggingFace push ───────────────────────────────────
push:
	@echo "Push to HuggingFace Spaces:"
	@echo "  openenv push --repo-id $(HF_REPO_ID)"
	openenv push --repo-id $(HF_REPO_ID)

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -f inference_results.json demo_results.json
