FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY app.py backend.py ./
COPY .env ./.env

EXPOSE 8025

RUN pip install -r requirements.txt
CMD ["python", "app.py"]