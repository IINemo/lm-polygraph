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
    "samsum_simple_instruct": {
        "name": "samsum",
        "train_split": "train",
        "test_split": "test",
        "prepare_func": partial(
            prepare_samsum,
            prompt="Provide a summary for the following dialogue. The summary should (1) be rather short, (2) extract important pieces of information, (3) include names of interlocutors, (4) be written in the third person.\n\nHere's the dialogue:\n\n{text} (End of dialogue).\n\nSummary:\n",
        ),
        "dataset": "samsum",
        "subset": "simple_instruct",
    },
}
