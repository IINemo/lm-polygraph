FROM python:3.11.11-slim-bullseye
COPY scripts/requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
RUN pip install .

CMD ["polygraph_server"]
