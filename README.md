# Uncertainty estimation for LLMs

## Usage examples

* [example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/example.ipynb): examples of library usage
* [Colab](https://colab.research.google.com/drive/1JS-NG0oqAVQhnpYY-DsoYWhz35reGRVJ?usp=sharing): example of running interface from notebook (careful: models other from `bloomz-560m` can be run only with Colab-pro subscription)

## Benchmarks

To evaluate the performance of uncertainty methods run: 

```
python3 -m main --dataset triviaqa.csv --model databricks/dolly-v2-3b --save_path test.man --cache_path . --seed 1 2 3 4 5
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

## Web application

### Installation
```
cd app && npm install && cd ../
```

### Starting the model server

Requires python3.10

```
python3 -m app.service
```

### Starting the web application server

```
node app/index.js
```

Once both servers are up and running, the chat model will be available at <http://localhost:3001/>.
