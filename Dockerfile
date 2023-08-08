FROM python:3.10.12-slim-bullseye
COPY scripts/requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
RUN pip install .
RUN rm -rf /root/.cache/*
CMD polygraph_server
