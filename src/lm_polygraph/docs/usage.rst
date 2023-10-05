Basic usage
===========

.. _installation:

Installation
------------

To use LM-Polygraph, first clone the repo and conduct installation using pip, it is recommended to use virtual environment.
Code example is presented below:

.. code-block:: console
    
    $ git clone https://github.com/IINemo/lm-polygraph.git
    $ python3 -m venv env
    $ source env/bin/activate
    (env) $ cd lm-polygraph
    (env) $ pip install .

   
.. _quick_start:

Quick start
-----------

1. Initialize the model (encoder-decoder or decoder-only) from HuggingFace or a local file. For example, `bigscience/bloomz-3b`::
    
    from lm_polygraph.utils.model import WhiteboxModel

    model = WhiteboxModel.from_pretrained(
        "bigscience/bloomz-3b",
        device="cuda:0",
    )


2. Specify UE method::

    from lm_polygraph.estimators import *

    ue_method = MeanPointwiseMutualInformation()


3. Get predictions and their uncertainty scores::

    from lm_polygraph.utils.manager import estimate_uncertainty

    input_text = "Who is George Bush?"
    estimate_uncertainty(model, ue_method, input_text=input_text)


Other examples:

* examples of library usage: https://github.com/IINemo/lm-polygraph/blob/main/notebooks/example.ipynb
* examples of library usage for the QA task with `bigscience/bloomz-3b` on the `TriviaQA` dataset: https://github.com/IINemo/lm-polygraph/blob/main/notebooks/qa_example.ipynb
* examples of library usage for the NMT task with `facebook/wmt19-en-de` on the `WMT14 En-De` dataset: https://github.com/IINemo/lm-polygraph/blob/main/notebooks/mt_example.ipynb
* examples of library usage for the ATS task with `facebook/bart-large-cnn` model on the `XSUM` dataset: https://github.com/IINemo/lm-polygraph/blob/main/notebooks/ats_example.ipynb 
* example of running interface from notebook (careful: only `bloomz-560m`, `gpt-3.5-turbo` and `gpt-4` fits default memory limit, other models can be run only with Colab-pro subscription): https://colab.research.google.com/drive/1JS-NG0oqAVQhnpYY-DsoYWhz35reGRVJ?usp=sharing



.. _benchmarks:

Benchmarks
----------

To evaluate the performance of uncertainty estimation methods run::

    polygraph_eval --dataset triviaqa.csv --model databricks/dolly-v2-3b --save_path test.man --cache_path . --seed 1 2 3 4 5


Parameters:

* `dataset`: path to .csv dataset
* `model`: path to huggingface model
* `batch_size`: batch size for generation (default: 2)
* `seed`: seed for generation (default: 1; can specify several seeds for multiple tests)
* `device`: `cpu` or `cuda:N` (default: `cuda:0` if avaliable, `cpu` otherwise)
* `save_path`: file path to save test results (the directory better be existing)
* `cache_path`: directory path to cache intermediate calculations (the directory better be existing)

Use `visualization_tables.ipynb` to generate the summarizing tables for an experiment.

The XSUM, TriviaQA, WMT16ru-en datasets downsampled to 300 samples can be found `here <https://drive.google.com/drive/folders/1bQlvPRZHdZvdpAyBQ_lQiXLq9t5whTfi?usp=sharing>`_.