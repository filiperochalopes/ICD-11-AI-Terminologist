FROM python:3.11-slim

ARG VERSION=V3
ENV VERSION=${VERSION}

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python ${VERSION}/download_model.py

CMD ["langgraph", "run", "${VERSION}:app"]
