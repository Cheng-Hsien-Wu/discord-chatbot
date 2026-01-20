FROM python:3.13-slim

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Install tzdata for ZoneInfo
RUN apt-get update && apt-get install -y tzdata && rm -rf /var/lib/apt/lists/*

COPY . .

CMD ["python", "main.py"]
