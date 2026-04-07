FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg gcc

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]