FROM python:3.10.12-slim-bullseye
RUN apt update && apt install npm -y
COPY scripts/requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
RUN cd /app && pip install .
RUN cd /app/src/lm_polygraph/app && npm install
WORKDIR /app
CMD bash scripts/polygraph_all.sh
#bash /app/scripts/polygraph_all.sh
#CMD ls /app
#CMD python -c "print('hello world')"

RUN pip install openai einos
