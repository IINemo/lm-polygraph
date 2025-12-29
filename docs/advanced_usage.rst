Advanced Usage
==============


Using UEManager directly
-------------------------

If you are calling ``UEManager`` directly from Python code, you'll need to wrap each generation metric in ``AggregatedMetric`` before passing them to ``UEManager`` constructor:

.. code-block:: python

    from lm_polygraph.generation_metrics import AggregatedMetric, RougeMetric
    from lm_polygraph.utils.manager import UEManager

    metrics = [
        AggregatedMetric(base_metric=RougeMetric('rouge1'))
        AggregatedMetric(base_metric=RougeMetric('rouge2'))
        AggregatedMetric(base_metric=RougeMetric('rougeL'))
    ]

    man = UEManager(
        dataset,
        model,
        estimators,
        generation_metrics,
        ue_metrics
        **other_args)

    man()


Constrained generation
----------------------

WiP

Uncertainty calibration
-----------------------

WiP

Custom modules
--------------

WiP
