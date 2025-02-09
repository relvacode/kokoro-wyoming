FROM python:3.12-slim
ADD main.py .
ADD requirements.txt .
RUN pip install -r requirements.txt
CMD ["python", "./main.py"]