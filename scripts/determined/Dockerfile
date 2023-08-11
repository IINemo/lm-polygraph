FROM  mephodybro/det_polygraph:0.0.11
RUN apt update && apt install wget git tmux -y
RUN wget https://raw.githubusercontent.com/IINemo/lm-polygraph/main/scripts/requirements.txt
RUN pip install -r requirements.txt
RUN curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | tee /etc/apt/sources.list.d/ngrok.list && apt update && apt install ngrok -y
RUN git clone https://github.com/IINemo/lm-polygraph /app
WORKDIR /app
# RUN git checkout demo
RUN pip install .
RUN apt update && apt install nvtop -y
RUN rm -rf /root/.cache/*
