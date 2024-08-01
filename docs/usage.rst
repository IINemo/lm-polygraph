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

Uncertainty for single input
----------------------------

1.
    Initialize the base model (encoder-decoder or decoder-only) and tokenizer from HuggingFace or a local file, and use them to initialize the ``WhiteboxModel`` for evaluation. For example, with ``bigscience/bloomz-560m``:

    .. code-block:: python

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from lm_polygraph.utils.model import WhiteboxModel

        model_path = "bigscience/bloomz-560m"
        base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:0")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        model = WhiteboxModel(base_model, tokenizer, model_path=model_path)

    **Alternatively**, you can use ``WhiteboxModel#from_pretrained`` method to let LM-Polygraph download the model and tokenizer for you. However, **this approach is deprecated** and will be removed in the next major release.

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

More examples for obtaining uncertainty for single generation: ``examples/basic_example.ipynb``

.. _benchmarks:

Benchmarking uncertainty estimation on a dataset
------------------------------------------------

CLI
^^^

Recommended way of running benchmarks is by invoking the ``polygraph_eval`` script. Configuration for experiments is done via Hydra YAML config files. 

Basic evaluation is invoked like so:

.. code-block:: console

    $ HYDRA_CONFIG=/absolute/path/to/config.yaml polygraph_eval

As usual with Hydra, you can override any parameter from the config file by specifying it in the command line. For example, to override the batch size:

.. code-block:: console

    $ HYDRA_CONFIG=/absolute/path/to/config.yaml polygraph_eval --batch_size=4

Examples of configuration files for several widely used datasets can be found in the ``examples/configs`` directory of the repository.

The results of evaluation will be saved as a serialized UEManager object to the directory specified by ``save_path`` in the config file. Refer to :ref:`UE Manager` for more information about the structure of the UEManager object.

To visualize benchmarking results, use and adapt for your case the ``notebooks/visualization_tables.ipynb`` notebook.

Python
^^^^^^

It is also possible to run benchmarks from Python code. Examples of how to do this can be found in the following notebooks:

* examples for the QA task with `bigscience/bloomz-3b` on the TriviaQA dataset: ``examples/qa_example.ipynb``
* examples for the NMT task with `facebook/wmt19-en-de` on the WMT14 En-De dataset: ``examples/mt_example.ipynb``
* examples for the ATS task with `facebook/bart-large-cnn` model on the XSUM dataset: ``examples/ats_example.ipynb``

To run more elaborate benchmarks directly from python, refer to the source code of ``polygraph_eval`` script.
