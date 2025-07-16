from functools import partial


def prepare_xsum(
    dataset,
    prompt,
):
    x, y = [], []
    for inst in dataset:
        x.append(prompt.format(text=inst["document"]))
        y.append(inst["summary"])
    return x, y


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
    },
}
