from functools import partial


def prepare_samsum(
    dataset,
    prompt,
):
    x, y = [], []
    for inst in dataset:
        x.append(prompt.format(text=inst["dialogue"]))
        y.append(inst["summary"])
    return x, y


CONFIG = {
    "samsum": {
        "name": "samsum",
        "train_split": "train",
        "test_split": "test",
        "prepare_func": partial(
            prepare_samsum,
            prompt="Here's the dialogue and it's short one-sentence summary.\n\nDialogue:\n{text}\n\nSummary (one sentence):",
        ),
        "dataset": "samsum",
        "subset": "continuation",
    },
}
