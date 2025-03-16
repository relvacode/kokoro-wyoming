FROM python:3.12-slim
ENV PYTHONPATH=/app
WORKDIR /app/src

# Install required packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Download required model files
RUN mkdir -p /app/src
ADD https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx /app/src/kokoro-v1.0.onnx
ADD https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin /app/src/voices-v1.0.bin

COPY src/ /app/src/

ENTRYPOINT ["/usr/local/bin/python3", "main.py"]
