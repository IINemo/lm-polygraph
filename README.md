# Uncertainty estimation for LLMs

## Installation

```
git clone https://github.com/IINemo/lm-polygraph.git
cd lm-polygraph
pip install .
```

## Usage examples

1. Initialize the model (encoder-decoder or decoder-only) from HuggingFace or a local file. For example, `bigscience/bloomz-3b`
```python
from lm_polygraph.utils.model import WhiteboxModel

model = WhiteboxModel.from_pretrained(
    "bigscience/bloomz-3b",
    device="cuda:0",
)
```

2. Specify input text and UE method
```python
from lm_polygraph.estimators import *

ue_method = MutualInformationSeq()
input_text = "Who is George Bush?"
```

3. Compute results
```python
from lm_polygraph.utils.manager import estimate_uncertainty

estimate_uncertainty(model, ue_method, input_text=input_text)
```

### Other examples:

* [example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/notebooks/example.ipynb): examples of library usage
* [qa_example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/notebooks/qa_example.ipynb): examples of library usage for QA task with `bigscience/bloomz-3b` on the `TriviaQA` dataset
* [mt_example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/notebooks/mt_example.ipynb): examples of library usage for NMT task with `facebook/wmt19-en-de` on the `WMT14 En-De` dataset
* [ats_example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/notebooks/ats_example.ipynb): examples of library usage for ATS task with `facebook/bart-large-cnn` model on the `XSUM` dataset
* [Colab](https://colab.research.google.com/drive/1JS-NG0oqAVQhnpYY-DsoYWhz35reGRVJ?usp=sharing): example of running interface from notebook (careful: models other from `bloomz-560m` can be run only with Colab-pro subscription)


## Benchmarks

To evaluate the performance of uncertainty methods run: 

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

### Starting the model server

Requires python3.10

```
polygraph_backend
```

### Starting the web application server

```
polygraph_frontend
```

Once both servers are up and running, the chat model will be available at <http://localhost:3001/>.
