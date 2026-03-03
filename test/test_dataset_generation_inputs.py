import pytest

from lm_polygraph.utils.dataset import Dataset


def test_dataset_iter_yields_generation_inputs_when_provided():
    dataset = Dataset(
        x=["p1", "p2", "p3"],
        y=["", "", ""],
        batch_size=2,
        generation_inputs=["g1", "g2", "g3"],
    )

    batches = list(dataset)
    assert len(batches) == 2

    x1, y1, g1 = batches[0]
    assert x1 == ["p1", "p2"]
    assert y1 == ["", ""]
    assert g1 == ["g1", "g2"]

    x2, y2, g2 = batches[1]
    assert x2 == ["p3"]
    assert y2 == [""]
    assert g2 == ["g3"]


def test_dataset_select_keeps_generation_inputs_aligned():
    dataset = Dataset(
        x=["p1", "p2", "p3"],
        y=["t1", "t2", "t3"],
        batch_size=2,
        generation_inputs=["g1", "g2", "g3"],
    )
    dataset.select([2, 0])

    assert dataset.x == ["p3", "p1"]
    assert dataset.y == ["t3", "t1"]
    assert dataset.generation_inputs == ["g3", "g1"]


def test_dataset_generation_inputs_length_validation():
    with pytest.raises(ValueError, match="generation_inputs length must match x length"):
        Dataset(
            x=["p1", "p2"],
            y=["", ""],
            batch_size=1,
            generation_inputs=["g1"],
        )

