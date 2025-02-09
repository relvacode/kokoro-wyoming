FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ /app/src/
ENV PYTHONPATH=/app
CMD ["python", "src/main.py"]