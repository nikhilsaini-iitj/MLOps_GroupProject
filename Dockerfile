# Use a slim Python base image
FROM python:3.11-slim

# Accept HF model name as a build argument
ARG HF_MODEL_NAME=your-username/your-model-repo
ENV HF_MODEL_NAME=${HF_MODEL_NAME}

# Set working directory
WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY id2label.json .

# Run inference
ENTRYPOINT ["python", "src/inference.py"]
