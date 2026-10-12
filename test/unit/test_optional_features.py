"""Small offline checks for the optional experiment profile."""

import numpy as np
import pytest


def test_lexical_similarity():
    pytest.importorskip("rouge_score")
    pytest.importorskip("nltk")
    from lm_polygraph.estimators import LexicalSimilarity
    from lm_polygraph.utils.factory_estimator import FactoryEstimator

    estimator = FactoryEstimator()("LexicalSimilarity", {"metric": "rougeL"})
    assert isinstance(estimator, LexicalSimilarity)
    np.testing.assert_allclose(
        estimator({"sample_texts": [["the same sentence", "the same sentence"]]}),
        [-1.0],
    )


def test_openai_backend_construction():
    pytest.importorskip("openai")
    from lm_polygraph import BlackboxModel

    model = BlackboxModel.from_openai("test-key", "test-model")
    assert model.openai_api.api_key == "test-key"
    model.openai_api.close()


def test_csv_dataset(tmp_path):
    pytest.importorskip("pandas")
    from lm_polygraph import Dataset

    csv = tmp_path / "data.csv"
    csv.write_text("input,target\nquestion,answer\n")
    dataset = Dataset.from_csv(str(csv), "input", "target", batch_size=1)
    assert dataset.x == ["question"]
    assert dataset.y == ["answer"]
