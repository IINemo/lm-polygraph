Benchmarking novel UQ methods
-----------------------------

For benchmarking, the stat calculators should be equipped with a builder function, so ``polygraph_eval`` knows how to create a corresponding object. The path to this function is specified in the .yaml config file for ``polygraph_eval``.


Multi-reference datasets
------------------------

When running a benchmark on a dataset with multiple reference values (like TriviaQA with multiple ``alias`` values for each question), you can evaluate generation metrics against each provided reference. Resulting metric value will be the maximum among all references.

CLI
---

When running benchmark from CLI using ``polygraph_eval`` script, just set ``multiref`` config option to ``true``.
