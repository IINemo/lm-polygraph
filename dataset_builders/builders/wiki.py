from functools import partial
from .stripped_formatters import continuation_stripped


def prepare_wiki(dataset, input_column, prompt):
    x, y, s = [], [], []
    for sample in dataset[input_column]:
        context = sample["context"].strip()
        x.append(prompt.format(context=context))
        y.append("")
        s.append(continuation_stripped(context))
    return x, y, s


CONFIG = {
    "wiki_bio": {
        "name": "wiki_bio",
        "test_split": "test",
        "prepare_func": partial(
            prepare_wiki,
            input_column="input_text",
            prompt="This is a Wikipedia passage about {context}:\n",
        ),
        "dataset": "wiki_bio",
        "subset": "continuation",
    },
}
