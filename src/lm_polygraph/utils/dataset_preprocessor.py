def preprocess_translation(dataset, x_column, y_column, prompt):
    lang = {
        "de": "German",
        "fr": "French",
    }
    x, y = [], []
    source_lang = lang.get(x_column, "English")
    target_lang = lang.get(y_column, "English")
    for inst in dataset["translation"]:
        x.append(prompt.format(source_lang=source_lang, target_lang=target_lang, text=inst[x_column]))
        y.append(inst[y_column])
    return x, y


def preprocess_coqa(dataset, x_column, y_column, prompt):
    x, y = [], []
    for inst in dataset:
        for question, answer in zip(inst[x_column], inst[y_column]["input_text"]):
            x.append(prompt.format(story=inst["story"], question=question))
            y.append(answer)
    return x, y


def preprocess_babiqa(dataset, x_column, y_column, prompt):
    x, y = [], []
    for inst in dataset:
        inst = inst["story"]
        context = ""
        for text, answer in zip(inst[x_column], inst[y_column]):
            if answer == "":
                context += text + " "
            else:
                x.append(prompt.format(context=context.strip(), question=text))
                y.append(answer)
    return x, y


def preprocess_with_prompt(dataset, x_column, y_column, prompt):
    return [prompt.format(text=text) for text in dataset[x_column]], dataset[y_column]


def preprocess_dataset(dataset, dataset_path, x_column, y_column, prompt):
    if "translation" in dataset.column_names:
        return preprocess_translation(dataset, x_column, y_column)
    elif ("coqa" in dataset_path.lower()) and len(prompt):
        return preprocess_coqa(dataset, x_column, y_column, prompt)
    elif ("babi_qa" in dataset_path.lower()) and len(prompt):
        return preprocess_babiqa(dataset, x_column, y_column, prompt)
    elif len(prompt):
        return preprocess_with_prompt(dataset, x_column, y_column, prompt)
    else:
        return dataset[x_column], dataset[y_column]
