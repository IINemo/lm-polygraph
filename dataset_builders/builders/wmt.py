from functools import partial
from .stripped_formatters import translation_stripped
import datasets


def prepare_wmt(dataset, input_column, output_column, prompt):
    column_lang = {
        "de": "German",
        "fr": "French",
        "en": "English",
        "ru": "Russian",
    }
    x, y, s = [], [], []
    for inst in dataset["translation"]:
        x.append(
            prompt.format(
                source_lang=column_lang[input_column],
                target_lang=column_lang[output_column],
                text=inst[input_column],
            )
        )
        y.append(inst[output_column])
        s.append(translation_stripped(inst[input_column]))
    return x, y, s


CONFIG = {
    "wmt14_deen": {
        "name": ["wmt14", "de-en"],
        "train_split": "train",
        "test_split": "test",
        "prepare_func": partial(
            prepare_wmt,
            input_column="de",
            output_column="en",
            prompt="Here is a sentence in {source_lang} language and its translation in {target_lang} language.\n\nOriginal:\n{text}\nTranslation:\n",
        ),
        "dataset": "wmt14",
        "subset": "deen",
        "features": datasets.Features(
            {
                "input": datasets.Value("string"),
                "output": datasets.Value("string"),
                "stripped_input": datasets.Value("string"),
            }
        ),
    },
    "wmt14_fren": {
        "name": ["wmt14", "fr-en"],
        "train_split": "train",
        "test_split": "test",
        "prepare_func": partial(
            prepare_wmt,
            input_column="fr",
            output_column="en",
            prompt="Here is a sentence in {source_lang} language and its translation in {target_lang} language.\n\nOriginal:\n{text}\nTranslation:\n",
        ),
        "dataset": "wmt14",
        "subset": "fren",
        "features": datasets.Features(
            {
                "input": datasets.Value("string"),
                "output": datasets.Value("string"),
                "stripped_input": datasets.Value("string"),
            }
        ),
    },
    "wmt14_fren_simple_instruct": {
        "name": ["wmt14", "fr-en"],
        "train_split": "train",
        "test_split": "test",
        "prepare_func": partial(
            prepare_wmt,
            input_column="fr",
            output_column="en",
            prompt="Translate the following {source_lang} text into {target_lang}, give only the translation.\n\nOriginal:\n{text}\nTranslation:\n",
        ),
        "dataset": "wmt14",
        "subset": "fren_simple_instruct",
        "features": datasets.Features(
            {
                "input": datasets.Value("string"),
                "output": datasets.Value("string"),
                "stripped_input": datasets.Value("string"),
            }
        ),
    },
    "wmt19_deen": {
        "name": ["wmt19", "de-en"],
        "train_split": "train",
        "test_split": "validation",
        "prepare_func": partial(
            prepare_wmt,
            input_column="de",
            output_column="en",
            prompt="Here is a sentence in {source_lang} language and its translation in {target_lang} language.\n\nOriginal:\n{text}\nTranslation:\n",
        ),
        "dataset": "wmt19",
        "subset": "deen",
        "features": datasets.Features(
            {
                "input": datasets.Value("string"),
                "output": datasets.Value("string"),
                "stripped_input": datasets.Value("string"),
            }
        ),
    },
    "wmt19_deen_simple_instruct": {
        "name": ["wmt19", "de-en"],
        "train_split": "train",
        "test_split": "validation",
        "prepare_func": partial(
            prepare_wmt,
            input_column="de",
            output_column="en",
            prompt="Translate the following {source_lang} text into {target_lang}, give only the translation.\n\nOriginal:\n{text}\nTranslation:\n",
        ),
        "dataset": "wmt19",
        "subset": "deen_simple_instruct",
        "features": datasets.Features(
            {
                "input": datasets.Value("string"),
                "output": datasets.Value("string"),
                "stripped_input": datasets.Value("string"),
            }
        ),
    },
    "wmt19_ruen": {
        "name": ["wmt19", "ru-en"],
        "train_split": "train",
        "test_split": "validation",
        "prepare_func": partial(
            prepare_wmt,
            input_column="ru",
            output_column="en",
            prompt="Here is a sentence in {source_lang} language and its translation in {target_lang} language.\n\nOriginal:\n{text}\nTranslation:\n",
        ),
        "dataset": "wmt19",
        "subset": "ruen",
        "features": datasets.Features(
            {
                "input": datasets.Value("string"),
                "output": datasets.Value("string"),
                "stripped_input": datasets.Value("string"),
            }
        ),
    },
    "wmt19_ruen_simple_instruct": {
        "name": ["wmt19", "ru-en"],
        "train_split": "train",
        "test_split": "validation",
        "prepare_func": partial(
            prepare_wmt,
            input_column="ru",
            output_column="en",
            prompt="Translate the following {source_lang} text into {target_lang}, give only the translation.\n\nOriginal:\n{text}\nTranslation:\n",
        ),
        "dataset": "wmt19",
        "subset": "ruen_simple_instruct",
        "features": datasets.Features(
            {
                "input": datasets.Value("string"),
                "output": datasets.Value("string"),
                "stripped_input": datasets.Value("string"),
            }
        ),
    },
}
