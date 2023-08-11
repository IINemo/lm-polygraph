#FROM determinedai/environments:cuda-11.3-pytorch-1.10-deepspeed-0.8.3-gpu-0.22.1
FROM  mephodybro/det_polygraph:0.0.11
#RUN conda create -n poly python=3.10 -y
#SHELL ["conda", "run", "-n", "poly", "/bin/bash", "-c"]
COPY scripts/requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
RUN pip install .
RUN rm -rf /root/.cache/*