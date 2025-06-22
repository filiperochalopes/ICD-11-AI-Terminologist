FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./

RUN apt-get update && apt-get install -y build-essential cmake python3-dev && rm -rf /var/lib/apt/lists/*

RUN pip install --prefer-binary -r requirements.txt

COPY . .

CMD ["python", "main.py"]