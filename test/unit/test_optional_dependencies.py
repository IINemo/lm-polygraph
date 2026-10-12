"""Offline packaging regressions; no model downloads or API credentials needed."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[2]


def run_isolated(code, blocked=()):
    # A fresh interpreter prevents previously imported optional modules masking bugs.
    bootstrap = f"""
import importlib.abc
import importlib.util
import sys
# Optional availability probes must see the same absence as actual imports.
original_find_spec = importlib.util.find_spec
def find_spec(name, package=None):
    if name.split('.')[0] in {set(blocked)!r}:
        return None
    return original_find_spec(name, package)
importlib.util.find_spec = find_spec
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {set(blocked)!r}:
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, BlockOptional())
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(bootstrap) + textwrap.dedent(code)],
        cwd=ROOT,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_core_without_optional_dependencies():
    run_isolated(
        """
import numpy as np
import lm_polygraph
from lm_polygraph import WhiteboxModel, BlackboxModel, UEManager, estimate_uncertainty
from lm_polygraph import Dataset, CausalLMWithUncertainty, APIWithUncertainty
from lm_polygraph.estimators import MeanTokenEntropy
from lm_polygraph.stat_calculators import EntropyCalculator
from lm_polygraph.utils.factory_estimator import FactoryEstimator
from lm_polygraph.defaults.register_default_stat_calculators import register_default_stat_calculators
for backend in ['Whitebox', 'Blackbox', 'VisualLM']:
    assert register_default_stat_calculators(backend)
stats = EntropyCalculator()({'greedy_log_probs': [np.log([[.5, .5], [.5, .5]])]})
np.testing.assert_allclose(MeanTokenEntropy()(stats), [np.log(2)])
assert isinstance(FactoryEstimator()('MeanTokenEntropy', {}), MeanTokenEntropy)
from lm_polygraph.utils import estimate_uncertainty as utility_estimate
assert utility_estimate is estimate_uncertainty
assert lm_polygraph.WhiteboxModel is WhiteboxModel
assert 'WhiteboxModel' in dir(lm_polygraph)
try:
    lm_polygraph.not_a_public_name
except AttributeError:
    pass
else:
    raise AssertionError('Unknown attributes must fail')
""",
        blocked=(
            "openai",
            "diskcache",
            "bitsandbytes",
            "flask",
            "fastchat",
            "spacy",
            "nltk",
            "rouge_score",
            "sacrebleu",
            "bert_score",
            "evaluate",
            "sentence_transformers",
            "boostedprob",
            "datasets",
            "pandas",
            "comet",
            "vllm",
        ),
    )


def test_missing_backend_has_install_hint():
    run_isolated(
        """
from lm_polygraph import BlackboxModel
try:
    BlackboxModel.from_openai('test-key', 'test-model')
except ImportError as e:
    assert 'lm-polygraph[openai]' in str(e), str(e)
else:
    raise AssertionError('Missing backend should fail on use')
""",
        blocked=("openai",),
    )


def test_missing_metric_has_install_hint():
    run_isolated(
        """
from lm_polygraph import generation_metrics
try:
    generation_metrics.RougeMetric
except ImportError as e:
    assert 'lm-polygraph[evaluation]' in str(e), str(e)
else:
    raise AssertionError('Missing metric dependency should fail on access')
""",
        blocked=("rouge_score",),
    )


def test_missing_semantic_backend_has_install_hint():
    run_isolated(
        """
from lm_polygraph.stat_calculators import CrossEncoderSimilarityMatrixCalculator
calc = CrossEncoderSimilarityMatrixCalculator()
try:
    calc._setup('cpu')
except ImportError as e:
    assert 'lm-polygraph[semantic]' in str(e), str(e)
else:
    raise AssertionError('Missing semantic backend should fail on use')
""",
        blocked=("sentence_transformers",),
    )


def test_cli_can_load_without_optional_metrics():
    run_isolated(
        """
import runpy
module = runpy.run_path('scripts/polygraph_eval', run_name='packaging_test')
from omegaconf import OmegaConf
config = OmegaConf.create({'generation_metrics': [{'name': 'AccuracyMetric'}]})
assert len(module['get_generation_metrics'](config)) == 1
""",
        blocked=("rouge_score", "bert_score", "spacy", "openai", "boostedprob"),
    )


def test_wheel_metadata_keeps_optional_dependencies_out_of_core():
    from importlib.metadata import metadata
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    package = metadata("lm-polygraph")
    requirements = [Requirement(r) for r in package.get_all("Requires-Dist")]
    core = {
        canonicalize_name(r.name)
        for r in requirements
        if r.marker is None or r.marker.evaluate({"extra": ""})
    }
    assert not core.intersection(
        {
            "openai",
            "diskcache",
            "bitsandbytes",
            "flask",
            "fschat",
            "spacy",
            "pytest",
            "datasets",
            "pandas",
            "rouge-score",
            "bert-score",
            "sentence-transformers",
            "boostedprob",
            "evaluate",
        }
    )
    assert {"openai", "vllm", "quantization", "evaluation", "demo", "dev"} <= set(
        package.get_all("Provides-Extra")
    )
