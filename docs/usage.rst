Basic usage
===========

.. _installation:

Installation
------------

LM-Polygraph requires Python 3.10 or newer. Use a virtual environment to
isolate its dependencies. Local models need enough RAM or GPU memory to hold
the weights and generation statistics; CUDA is optional for the quick start.

From PyPI
^^^^^^^^^

Install the published package:

.. code-block:: console

    $ python -m venv .venv
    $ source .venv/bin/activate
    $ python -m pip install lm-polygraph

On Windows PowerShell, activate with ``.venv\Scripts\Activate.ps1`` instead.
To pin a release, specify its version, for example ``lm-polygraph==0.5.0``.

From GitHub
^^^^^^^^^^^

Clone the repository to use the example notebooks and benchmark configurations,
which are not included in the PyPI package. Run subsequent benchmark commands
from this checkout:

.. code-block:: console

    $ git clone https://github.com/IINemo/lm-polygraph.git
    $ cd lm-polygraph
    $ python -m venv .venv
    $ source .venv/bin/activate
    $ python -m pip install -e .

This installs the development version. To use a specific release, run
``git checkout v0.5.0`` (or another release tag) inside the checkout before
installing. The editable install makes local source changes available without
reinstalling.

Optional dependencies
^^^^^^^^^^^^^^^^^^^^^

The COMET translation metric requires the ``comet`` extra:

.. code-block:: console

    $ python -m pip install "lm-polygraph[comet]"

COMET constrains NumPy to versions below 2.0, which can conflict with other
inference packages. Use a separate environment when their requirements conflict.
vLLM examples require a separate installation of vLLM compatible with your
hardware and PyTorch version.

.. _quick_start:

Uncertainty for a single input
------------------------------

This complete example generates a short response and estimates its mean token
entropy. The first run downloads the model and tokenizer from Hugging Face.
CPU execution is supported but slower than GPU execution.

.. code-block:: python

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lm_polygraph import WhiteboxModel, estimate_uncertainty
    from lm_polygraph.estimators import MeanTokenEntropy
    from lm_polygraph.utils.generation_parameters import GenerationParameters

    model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    base_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    base_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = WhiteboxModel(
        base_model,
        tokenizer,
        model_path=model_path,
        generation_parameters=GenerationParameters(max_new_tokens=32),
        instruct=True,
    )

    result = estimate_uncertainty(
        model, MeanTokenEntropy(), input_text="What is the capital of France?"
    )
    print(result.generation_text)
    print(result.uncertainty)

``instruct=True`` applies the tokenizer's chat template. Use it with instruction
models that provide a template; use ``instruct=False`` for base completion models.
For encoder-decoder models, load with ``AutoModelForSeq2SeqLM`` and pass
``model_type="Seq2SeqLM"`` to ``WhiteboxModel``. Directly loading the Transformers
model and tokenizer is preferred over the deprecated
``WhiteboxModel.from_pretrained`` helper.

The ``GenerationParameters`` object controls generation. In this single-input
helper, ``max_new_tokens`` limits response length. The benchmark CLI instead
uses the top-level ``max_new_tokens`` configuration value.

Understanding the result
^^^^^^^^^^^^^^^^^^^^^^^^

``estimate_uncertainty`` generates a new response and scores it; it does not
accept an existing response to score. It returns an ``UncertaintyOutput`` with:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``uncertainty``
     - A scalar for sequence-level methods, or per-item scores for finer-grained methods.
   * - ``input_text``
     - The input supplied to the helper.
   * - ``generation_text``
     - The generated response.
   * - ``generation_tokens``
     - Generated tokens when available (IDs for local models, text tokens for API models with log probabilities); may be ``None``.
   * - ``model_path``
     - The model identifier supplied to the wrapper.
   * - ``estimator``
     - The estimator's string identifier.

Higher scores indicate greater uncertainty. The scale depends on the method:
mean token entropy is nonnegative apart from numerical error, while other
estimators can return negative scores. Raw scores are not probabilities of
hallucination and should not be compared directly across estimators or models.
Choose thresholds using representative validation data. See
:doc:`normalization/index` for conversion to confidence values.

Choosing a model and estimator
------------------------------

The estimator's required statistics determine which models it supports:

* ``WhiteboxModel`` exposes local model statistics. ``MeanTokenEntropy`` is a
  simple starting point for a sequence-level score.
* ``BlackboxModel`` with generated-token log probabilities supports methods such
  as ``Perplexity`` and ``MaximumSequenceProbability``.
* Text-only ``BlackboxModel`` services can use sampled-response methods such as
  ``LexicalSimilarity`` or ``EigValLaplacian``. These make multiple generations;
  semantic methods may also load auxiliary models.

Methods requiring training data or custom statistics may need the lower-level
``UEManager`` workflow rather than the single-input helper. See :doc:`design`
and the `low-level notebook <https://github.com/IINemo/lm-polygraph/blob/main/examples/low_level_example.ipynb>`_.

Using an API model
^^^^^^^^^^^^^^^^^^

Set ``OPENAI_API_KEY`` in your environment, then use a model that supports
chat completions and generated-token log probabilities:

.. code-block:: python

    import os
    from lm_polygraph import BlackboxModel, estimate_uncertainty
    from lm_polygraph.estimators import Perplexity

    model = BlackboxModel.from_openai(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model_path="gpt-4o",
        supports_logprobs=True,
    )
    result = estimate_uncertainty(
        model, Perplexity(), input_text="What has a head and a tail but no body?"
    )
    print(result.generation_text)
    print(result.uncertainty)

``supports_logprobs=True`` declares an endpoint capability; it cannot enable
log probabilities on an endpoint that does not provide them. Despite its name,
LM-Polygraph's ``Perplexity`` returns mean negative token log probability
(log perplexity), without exponentiating it. The single-input helper does not register entropy statistics for API models,
so ``MeanTokenEntropy`` cannot be substituted into this example.

For a text-only endpoint, leave ``supports_logprobs=False`` and choose, for
example, ``LexicalSimilarity("rougeL")``. Sample-based estimators increase API
usage. For an OpenAI-compatible provider, set the OpenAI SDK's
``OPENAI_BASE_URL`` environment variable before creating the model. The provider
must support the chat-completion parameters used by the chosen estimator.

.. _benchmarks:

Benchmarking uncertainty estimation on a dataset
------------------------------------------------

CLI
^^^

The ``polygraph_eval`` command runs benchmarks configured with Hydra YAML files.
From the repository root, run a small CoQA evaluation with the default model:

.. code-block:: console

    $ polygraph_eval --config-dir=./examples/configs --config-name=polygraph_eval_coqa.yaml subsample_eval_dataset=10 batch_size=1

The default configuration runs many estimators, so even a small dataset can
require additional model downloads and substantial memory. Inspect
``examples/configs/estimators/default_estimators.yaml`` before running a full
benchmark. Model and dataset downloads require network access; gated models
also require Hugging Face authorization.

Hydra overrides use ``key=value`` syntax, without a leading ``--``. For example,
append ``batch_size=4`` or ``max_new_tokens=32`` to the command above. You can
also supply the configuration through ``HYDRA_CONFIG``:

.. code-block:: console

    $ HYDRA_CONFIG=/absolute/path/to/polygraph_eval_coqa.yaml polygraph_eval batch_size=4

Keep the configuration's referenced files and directories available alongside it.
The CLI resolves relative model-loading script paths from the configuration's
directory.

Results are saved under ``save_path`` when configured, otherwise under Hydra's
runtime output directory. The CoQA configuration defaults to a dated directory
under ``./workdir/output``. Each seed produces a serialized manager named
``ue_manager_seed<seed>``. Refer to :ref:`UE Manager` for the manager's structure.
Use the `result tables notebook <https://github.com/IINemo/lm-polygraph/blob/main/notebooks/result_tables.ipynb>`_
or `visualization notebook <https://github.com/IINemo/lm-polygraph/blob/main/notebooks/vizualization_tables.ipynb>`_
to inspect results.

Python and further examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `Basic usage <https://github.com/IINemo/lm-polygraph/blob/main/examples/basic_example.ipynb>`_: single-input scoring.
* `Low-level integration <https://github.com/IINemo/lm-polygraph/blob/main/examples/low_level_example.ipynb>`_: statistics and claim-level uncertainty.
* `vLLM integration <https://github.com/IINemo/lm-polygraph/blob/main/examples/low_level_vllm_example.ipynb>`_: inference with vLLM.
* `Visual models <https://github.com/IINemo/lm-polygraph/blob/main/examples/basic_example_visual.ipynb>`_: image inputs.
* Dataset evaluation examples: `question answering <https://github.com/IINemo/lm-polygraph/blob/main/examples/other/qa_example.ipynb>`_,
  `translation <https://github.com/IINemo/lm-polygraph/blob/main/examples/other/mt_example.ipynb>`_, and
  `summarization <https://github.com/IINemo/lm-polygraph/blob/main/examples/other/ats_example.ipynb>`_.

Troubleshooting
---------------

* **CUDA is unavailable or memory runs out:** use a smaller model, shorten
  generation, or reduce benchmark batch size. The quick start falls back to CPU.
  Sample-based and auxiliary-model methods can need additional memory.
* **A chat template is missing:** use a tokenizer with the instruction model's
  chat template, or set ``instruct=False`` when using a base completion model.
* **An estimator reports missing statistics:** check that the model exposes the
  required statistics. API log probabilities are not a replacement for full
  white-box access.
* **Hydra cannot find a configuration:** run from the repository root or supply
  an absolute ``--config-dir`` path. Installing from PyPI does not install the
  example configuration files.
