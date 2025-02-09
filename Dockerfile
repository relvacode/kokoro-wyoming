FROM python:3.12-slim
WORKDIR /app

# Install curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install required packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Download required model files
RUN curl -L "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json" -o /app/src/voices.json && \
    curl -L "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx" -o /app/src/kokoro-v0_19.onnx
COPY src/ /app/src/
ENV PYTHONPATH=/app
CMD ["python", "src/main.py"]