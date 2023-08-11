##FROM determinedai/environments:py-3.10-pytorch-2.0-cpu-0.24.0
##RUN apt update && apt install nvidia-cuda-toolkit -y
##RUN pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
#FROM mephodybro/det_torch2:0.1
#RUN wget https://raw.githubusercontent.com/IINemo/lm-polygraph/main/scripts/requirements.txt
#RUN pip install -r requirements.txt
#RUN apt install git
#RUN git clone https://github.com/IINemo/lm-polygraph /app
#WORKDIR /app
#RUN pip install .
#RUN rm -rf /root/.cache/*
#RUN apt update && apt install locate -y
FROM determinedai/environments:cuda-11.3-pytorch-1.10-deepspeed-0.8.3-gpu-0.22.1
RUN conda create -n poly python=3.10 -y
SHELL ["conda", "run", "-n", "poly", "/bin/bash", "-c"]
CMD python --version
COPY scripts/requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
RUN pip install .
RUN rm -rf /root/.cache/*


#RUN conda activate poly
#CMD python --version



#RUN conda update python
#RUN apt-get update && apt-get install neovim nvtop git-lfs sudo -y
#
#RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
#RUN adduser roman.vashurin --uid 1015 --gid 100 --force-badname
#RUN echo "roman.vashurin:changeme"|chpasswd
#RUN echo "root:changeme"|chpasswd
#RUN usermod roman.vashurin -aG sudo
#
#RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
#RUN pip install jupyter tqdm numpy scipy matplotlib scikit-learn tensorboardX pandas plotly
#RUN pip install transformers==4.30 sentence-transformers datasets==2.14 tensorflow
#RUN pip install levenshtein hydra-core omegaconf marisa-trie pytreebank wget peft rouge-score
#RUN pip install nlpaug wikidata
#RUN pip uninstall -y apex
#COPY . /app
