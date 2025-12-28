Library design
==============

.. _design:

The uncertainty estimation functionality in the library resides on three main entities:

- Model wrappers: ``lm_polygraph/model_wrappers``
- Stat calculators: ``lm_polygraph/stat_calculators``
- Uncertainty estimators: ``lm_polygraph/estimators``

This design aims to decompose the LLM inference, heavy calculations, and uncertainty estimation process for modularity, flexibility, and evaluation performance.

Model wrappers
--------------

Model wrappers aim to encapsulate the guts of LLM inference process and provide a standardized interface for stat calculators. At the moment, the library supports:

- Wrapper for whitebox LLMs: ``lm_polygraph.model_wrappers.WhiteboxModel`` and ``lm_polygraph.model_wrappers.WhiteboxModelBasic``
- Wrapper for whitebox LLMs inferenced via vLLM: ``lm_polygraph.model_wrappers.WhiteboxModelvLLM``
- Wrapper for visual whitebox LLMs (image to text models): ``lm_polygraph.model_wrappers.VisualWhiteboxModel``
- Wrapper for LLMs deployed as services in the cloud, such as ChatGPT, Claude, etc. These models can be blackbox (when they provide only text) and greybox (when they provide also logits): ``lm_polygraph.model_wrappers.BlackboxModel``

Different model types should be inferenced in different ways, so wrappers help to abstract the inference process. Note also that not all stat calculators and estimators are available for all model types.
For example, blackbox models do not provide logits, so only sampling-based and verbalized estimators are available. vLLMs does not provide access to internal states of the model, so attention-based methods are not supported by them.

Stat calculators
----------------

Stat calculators perform heavy computations on top of the LLM. They control the LLM's inference and postprocess its results. The reason behind that is because UQ methods require special output from the LLM or need to aggregate results of multiple inferences.
Usually, there is not just one stat calculator, but a chain of them. For example, for performing claim-level UQ, you need to infer an LLM with ``GreedyProbsCalculator`` and split the generated text into atomic claims using ``ClaimsExtractor``. During benchmarking the results of the stat calculators could be consumed by many different uncertainty estimators, hence saving time for repetitive calculations.

Due to differences in inference procedures for different model types, stat calculators are not universally compatible with all model wrappers. To determine what LLM types are supported by a stat calculator, you can look at the type of the ``model`` argument in the ``__call__`` method. The most general stat calculator has the type ``Model``. For example, the type of the argument for ``GreedyProbsVisualCalculator`` is ``VisualWhiteboxModel``.

Estimators
----------

Estimators are the final step in the uncertainty estimation process. They take the results of the stat calculators and aggregate them into the uncertainty score.
The majority of estimators are computationally light, because in benchmarking the results of heavy computations should be leveraged by multiple uncertainty estimators for efficiency.


Automatic resolution of stat calculators for estimators
------------------------------------------------------
``UEManager`` is used for automatic resolution of stat calculators for estimators. This is crucial for 

- High-level API represented by the ``estimate_uncertainty`` function. 
- Benchmarking process represented by the ``polygraph_eval`` script.


Default configuration
----------------------
``estimate_uncertainty`` and option ``stat_calculators: auto`` in ``polygraph_eval`` leverage the configuration of stat_calculators and estimators specified in the ``lm_polygraph/defaults`` directory.

