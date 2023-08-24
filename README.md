[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/IINemo/isanlp_srl_framebank/blob/master/LICENSE)
![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg)

# LM-Polygraph: Uncertainty estimation for LLMs

[Installation](#installation) | [Basic usage](#basic_usage) | [Documentation](https://lm-polygraph.readthedocs.io/) | [Benchmark](#benchmark) | [Demo application](#demo_web_application)

LM-Polygraph provides a battery of state-of-the-art of uncertainty estimation (UE) methods for LMs in text generation tasks. High uncertainty can indicate the presence of hallucinations and knowing a score that estimates uncertinaty can help to make applications of LLMs safer.

The framework also introduces an extendable benchmark for consistent evaluation of UE techniques by researchers and a demo web application that enriches the standard chat dialog with confidence scores, empowering end-users to discern unreliable responses.

## Installation

```
git clone https://github.com/IINemo/lm-polygraph.git && cd lm-polygraph && pip install .
```

## <a name="basic_usage"></a>Basic usage

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

* [example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/examples/example.ipynb): simple examples of scoring individual queries;
* [qa_example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/examples/qa_example.ipynb): an example of scoring the `bigscience/bloomz-3b` model on the `TriviaQA` dataset;
* [mt_example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/examples/mt_example.ipynb): an of scoring the `facebook/wmt19-en-de` model on the `WMT14 En-De` dataset;
* [ats_example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/examples/ats_example.ipynb): an example of scoring the `facebook/bart-large-cnn` model on the `XSUM` summarization dataset;
* [colab](https://colab.research.google.com/drive/1JS-NG0oqAVQhnpYY-DsoYWhz35reGRVJ?usp=sharing): demo web application in Colab (`bloomz-560m`, `gpt-3.5-turbo`, and `gpt-4` fit the default memory limit; other models require Colab-pro).


## Benchmark

To evaluate the performance of uncertainty estimation methods consider a quick example: 

```
HYDRA_CONFIG=../configs/polygraph_eval/polygraph_eval.yaml python ./scripts/polygraph_eval \
    dataset="./workdir/data/triviaqa.csv" \
    model="databricks/dolly-v2-3b" \
    save_path="./workdir/output" \
    seed=[1,2,3,4,5]
```

Use [`visualization_tables.ipynb`](https://github.com/IINemo/lm-polygraph/blob/main/notebooks/vizualization_tables.ipynb) to generate the summarizing tables for an experiment.

A detailed description of the benchmark is in the [documentation](https://lm-polygraph.readthedocs.io/en/latest/usage.html#benchmarks).

## <a name="demo_web_application"></a>Demo web application

 
<img width="850" alt="gui7" src="https://github.com/IINemo/lm-polygraph/assets/21058413/51aa12f7-f996-4257-b1bc-afbec6db4da7">


### Start with Docker

```sh
docker run -p 3001:3001 -it \
    -v $HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub \
    --gpus all mephodybro/polygraph_demo:0.0.17 polygraph_server
```
The server should be available on `http://localhost:3001`

A more detailed description of the demo is available in the [documentation](https://lm-polygraph.readthedocs.io/en/latest/web_demo.html).

### Original implementation

The chat GUI is based on the following project: https://github.com/ioanmo226/chatgpt-web-application
