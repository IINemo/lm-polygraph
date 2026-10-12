FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
COPY . /app
WORKDIR /app
RUN pip install '.[evaluation,openai,boostedprob]'
RUN pip install jupyter

CMD ["/bin/bash"]
