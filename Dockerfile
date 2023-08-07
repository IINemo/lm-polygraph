FROM python:3.10.12-slim-bullseye
COPY scripts/requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app

RUN pip install openai einops
RUN pip install --upgrade "protobuf<=3.20.1"

RUN pip install .
CMD polygraph_server
