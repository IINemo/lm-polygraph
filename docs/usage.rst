Basic usage
===========

.. _installation:

Installation
------------

From GitHub
^^^^^^^^^^^^^^^^^^

To install latest from main brach, clone the repo and conduct installation using pip, it is recommended to use virtual environment.
Code example is presented below:

.. code-block:: console
    
    $ git clone https://github.com/IINemo/lm-polygraph.git
    $ python3 -m venv env
    $ source env/bin/activate
    (env) $ cd lm-polygraph
    (env) $ pip install .

Installation from GitHub is recommended if you want to explore notebooks with examples or use default benchmarking configurations, as they are included in the repository but not in the PyPI package.
However code from the main branch may be unstable, so it is recommended to checkout to the latest stable release before installation:

.. code-block:: console
    
    $ git clone https://github.com/IINemo/lm-polygraph.git
    $ git checkout tags/v0.3.0
    $ python3 -m venv env
    $ source env/bin/activate
    (env) $ cd lm-polygraph
    (env) $ pip install .

From PyPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install the latest stable version from PyPI, run:

.. code-block:: console

    $ python3 -m venv env
    $ source env/bin/activate
    $ pip install lm-polygraph

To install a specific version, run:

.. code-block:: console

    $ python3 -m venv env
    $ source env/bin/activate
    $ pip install lm-polygraph==0.3.0

.. _quick_start:

Quick start
-----------

1.
    Initialize the base model (encoder-decoder or decoder-only) and tokenizer from HuggingFace or a local file, and use them to initialize the `WhiteboxModel` for evaluation. For example, with `bigscience/bloomz-560m`:

    .. code-block:: python

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from lm_polygraph.utils.model import WhiteboxModel

        model_path = "bigscience/bloomz-560m"
        base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:0")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        model = WhiteboxModel(base_model, tokenizer, model_path=model_path)

    **Alternatively**, you can use `WhiteboxModel#from_pretrained` method to let LM-Polygraph download the model and tokenizer for you. However, **this approach is deprecated** and will be removed in the next major release.

    .. code-block:: python

        from lm_polygraph.utils.model import WhiteboxModel

        model = WhiteboxModel.from_pretrained(
            "bigscience/bloomz-3b",
            device_map="cuda:0",
        )

2.
    Specify UE method:

    .. code-block:: python

        from lm_polygraph.estimators import *

        ue_method = MeanPointwiseMutualInformation()

3.
    Get predictions and their uncertainty scores:

    .. code-block:: python

        from lm_polygraph.utils.manager import estimate_uncertainty

        input_text = "Who is George Bush?"
        ue = estimate_uncertainty(model, ue_method, input_text=input_text)
        print(ue)
        # UncertaintyOutput(uncertainty=-6.504108926902215, input_text='Who is George Bush?', generation_text=' President of the United States', model_path='bigscience/bloomz-560m')

Other examples:

* examples of library usage: https://github.com/IINemo/lm-polygraph/blob/main/notebooks/example.ipynb
* examples of library usage for the QA task with `bigscience/bloomz-3b` on the `TriviaQA` dataset: https://github.com/IINemo/lm-polygraph/blob/main/notebooks/qa_example.ipynb
* examples of library usage for the NMT task with `facebook/wmt19-en-de` on the `WMT14 En-De` dataset: https://github.com/IINemo/lm-polygraph/blob/main/notebooks/mt_example.ipynb
* examples of library usage for the ATS task with `facebook/bart-large-cnn` model on the `XSUM` dataset: https://github.com/IINemo/lm-polygraph/blob/main/notebooks/ats_example.ipynb 
* example of running interface from notebook (careful: only `bloomz-560m`, `gpt-3.5-turbo` and `gpt-4` fits default memory limit, other models can be run only with Colab-pro subscription): https://colab.research.google.com/drive/1JS-NG0oqAVQhnpYY-DsoYWhz35reGRVJ?usp=sharing



.. _benchmarks:

Benchmarks
----------

Hydra
^^^^^^^^^^
We recommend using Hydra YAMLs to configure LM-Polygraph. Detailed description of various parameters can be found in `examples/configs/polygraph_eval_example.yaml`. 

Evaluation is invoked like so::

    HYDRA_CONFIG=/absolute/path/to/config.yaml polygraph_eval

Direct configuration
^^^^^^^^^^
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
