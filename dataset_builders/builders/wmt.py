from functools import partial


def prepare_wmt(dataset, input_column, output_column, prompt):
    column_lang = {
        "de": "German",
        "fr": "French",
        "en": "English",
    }
    x, y = [], []
    for inst in dataset["translation"]:
        x.append(
            prompt.format(
                source_lang=column_lang[input_column],
                target_lang=column_lang[output_column],
                text=inst[input_column],
            )
        )
        y.append(inst[output_column])
    return x, y


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
        "is_main_dataset": False,
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
        "is_main_dataset": False,
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
        "is_main_dataset": False,
    },
}
