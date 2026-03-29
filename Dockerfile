FROM python:3.12-slim

WORKDIR /app

# Install system build tools required by some Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for better Docker layer caching
COPY files/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application source without creating nested files/files
COPY files/ /app/files/
COPY inference.py /app/inference.py

# Hugging Face Spaces uses port 7860
ENV PORT=7860

EXPOSE 7860

CMD ["uvicorn", "files.main:app", "--host", "0.0.0.0", "--port", "7860"]
