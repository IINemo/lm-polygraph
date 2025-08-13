from functools import partial
from .stripped_formatters import summarization_stripped
import datasets


def prepare_xsum(
    dataset,
    prompt,
):
    x, y, s = [], [], []
    for inst in dataset:
        x.append(prompt.format(text=inst["document"]))
        y.append(inst["summary"])
        s.append(summarization_stripped(inst["document"]))
    return x, y, s


CONFIG = {
    "xsum": {
        "name": "xsum",
        "train_split": "train",
        "test_split": "test",
        "prepare_func": partial(
            prepare_xsum,
            prompt="Here's the text and it's short one-sentence summary.\n\nText:\n{text}\n\nSummary (one sentence):",
        ),
        "dataset": "xsum",
        "subset": "continuation",
        "features": datasets.Features(
            {
                "input": datasets.Value("string"),
                "output": datasets.Value("string"),
                "stripped_input": datasets.Value("string"),
            }
        ),
    },
    "xsum_simple_instruct": {
        "name": "xsum",
        "train_split": "train",
        "test_split": "test",
        "prepare_func": partial(
            prepare_xsum,
            prompt="Provide a short one-sentence summary for the following text.\n\nHere's the text:\n{text} (End of text).\n\nSummary:\n",
        ),
        "dataset": "xsum",
        "subset": "simple_instruct",
        "features": datasets.Features(
            {
                "input": datasets.Value("string"),
                "output": datasets.Value("string"),
                "stripped_input": datasets.Value("string"),
            }
        ),
    },
}
