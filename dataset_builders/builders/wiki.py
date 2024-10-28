from functools import partial


def prepare_wiki(dataset, input_column, prompt):
    x, y = [], []
    for sample in dataset[input_column]:
        x.append(prompt.format(context=sample["context".strip()]))
        y.append("")
    return x, y


CONFIG = {
    "wiki_bio": {
        "name": "wiki_bio",
        "test_split": "test",
        "prepare_func": partial(
            prepare_wiki,
            input_column="input_text",
            prompt="This is a Wikipedia passage about {context}:\n",
        ),
    },
}
