[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/IINemo/isanlp_srl_framebank/blob/master/LICENSE)
![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg)

# LM-Polygraph: Uncertainty estimation for LLMs

## Installation

```
git clone https://github.com/IINemo/lm-polygraph.git && cd lm-polygraph && pip install .
```

## Basic usage

1. Initialize the model (encoder-decoder or decoder-only) from HuggingFace or a local file. For example, `bigscience/bloomz-3b`
```python
from lm_polygraph.utils.model import WhiteboxModel

model = WhiteboxModel.from_pretrained(
    "bigscience/bloomz-3b",
    device="cuda:0",
)
```

2. Specify UE method
```python
from lm_polygraph.estimators import *

ue_method = MeanPointwiseMutualInformation()
```

3. Get predictions and their uncertainty scores
```python
from lm_polygraph.utils.manager import estimate_uncertainty

input_text = "Who is George Bush?"
estimate_uncertainty(model, ue_method, input_text=input_text)
```

### Other examples:

* [example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/notebooks/example.ipynb): examples of library usage
* [qa_example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/notebooks/qa_example.ipynb): examples of library usage for the QA task with `bigscience/bloomz-3b` on the `TriviaQA` dataset
* [mt_example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/notebooks/mt_example.ipynb): examples of library usage for the NMT task with `facebook/wmt19-en-de` on the `WMT14 En-De` dataset
* [ats_example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/notebooks/ats_example.ipynb): examples of library usage for the ATS task with `facebook/bart-large-cnn` model on the `XSUM` dataset
* [Colab](https://colab.research.google.com/drive/1JS-NG0oqAVQhnpYY-DsoYWhz35reGRVJ?usp=sharing): example of running interface from notebook (careful: only `bloomz-560m`, `gpt-3.5-turbo` and `gpt-4` fits default memory limit, other models can be run only with Colab-pro subscription)


## Benchmarks

To evaluate the performance of uncertainty estimation methods run: 

```
polygraph_eval --dataset triviaqa.csv --model databricks/dolly-v2-3b --save_path test.man --cache_path . --seed 1 2 3 4 5
```

Parameters:

* `dataset`: path to .csv dataset
* `model`: path to huggingface model
* `batch_size`: batch size for generation (default: 2)
* `seed`: seed for generation (default: 1; can specify several seeds for multiple tests)
* `device`: `cpu` or `cuda:N` (default: `cuda:0` if avaliable, `cpu` otherwise)
* `save_path`: file path to save test results (the directory better be existing)
* `cache_path`: directory path to cache intermediate calculations (the directory better be existing)

Use `visualization_tables.ipynb` to generate the summarizing tables for an experiment.

The XSUM, TriviaQA, WMT16ru-en datasets downsampled to 300 samples can be found [here](https://drive.google.com/drive/folders/1bQlvPRZHdZvdpAyBQ_lQiXLq9t5whTfi?usp=sharing).

## Demo web application

### Starting the model server with web application

Requires python3.10

```
polygraph_server
```
The server should be available on `http://localhost:3001`

### To start from docker
You can run the image from dockerhub:
```sh
docker run -p 3001:3001 -it --gpus all mephodybro/polygraph_demo:0.0.17 polygraph_server
```
The server should be available on `http://localhost:3001`

If you want to use host huggingface checkpoints, mount the volume with `-v`, e.g. like this
```sh
docker run -p 3001:3001 -it -v $HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub --gpus all mephodybro/polygraph_demo:0.0.17 polygraph_server
```

You could rebuild the image from Dockerfile as well.
