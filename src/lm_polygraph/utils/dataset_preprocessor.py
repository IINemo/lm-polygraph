import numpy as np


def preprocess_translation(dataset, x_column, y_column, prompt):
    lang = {
        "de": "German",
        "fr": "French",
    }
    x, y = [], []
    source_lang = lang.get(x_column, "English")
    target_lang = lang.get(y_column, "English")
    for inst in dataset["translation"]:
        x.append(
            prompt.format(
                source_lang=source_lang, target_lang=target_lang, text=inst[x_column]
            )
        )
        y.append(inst[y_column])
    return x, y


def preprocess_coqa(dataset, x_column, y_column, prompt, description):
    def doc_to_text(doc, prompt, i=0):
        # Given a passage p, the conversation history {q1, a1, . . . qi−1, ai−1}
        # and a question qi, the task is to predict the answer ai
        doc_text = ""
        for q, a in zip(doc["questions"][:i], doc["answers"]["input_text"][:i]):
            doc_text += prompt.format(question=q, answer=a)

        return doc_text

    x, y = [], []
    for inst in dataset:
        formatted_description = description.format(story=inst["story"])
        for j, (question, answer) in enumerate(
            zip(inst[x_column], inst[y_column]["input_text"])
        ):
            formatted_prompt = (
                formatted_description
                + doc_to_text(inst, prompt, j)
                + prompt.format(
                    question=question,
                    answer="",
                )
            )
            x.append(formatted_prompt)
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


def preprocess_gsm8k(
    dataset,
    y_column,
    n_shot,
    description,
    prompt,
    few_shot_dataset,
    mmlu_max_subject_size,
):
    answers = ["A", "B", "C", "D"]
    subjects = np.array(dataset["subject"])
    x, y = [], []
    for subject in np.unique(subjects):
        formatted_description = description.format(subject=subject.replace("_", " "))
        if n_shot > 0:
            few_shot_ids = np.random.choice(
                len(few_shot_dataset), n_shot, replace=False
            )
            few_shot_data = few_shot_dataset.select(few_shot_ids)
            formatted_few_shot_prompt = ""
            for inst in few_shot_data:
                formatted_few_shot_prompt += prompt.format(
                    choices=inst["choices"],
                    question=inst["question"].strip(),
                    answer=answers[inst["answer"]],
                )

        subject_data = dataset.select(
            np.argwhere(subjects == subject).flatten()
        ).select(range(mmlu_max_subject_size))
        for inst in subject_data:
            formatted_prompt = prompt.format(
                choices=inst["choices"],
                question=inst["question"].strip(),
                answer="",
            )
            x.append(
                formatted_description + formatted_few_shot_prompt + formatted_prompt
            )
            y.append(answers[inst[y_column]])

    return x, y


def preprocess_mmlu(
    dataset,
    y_column,
    n_shot,
    description,
    few_shot_dataset,
    mmlu_max_subject_size,
    prompt,
):
    answers = ["A", "B", "C", "D"]
    subjects = np.array(dataset["subject"])
    x, y = [], []
    for subject in np.unique(subjects):
        formatted_description = description.format(subject=subject.replace("_", " "))
        if n_shot > 0:
            few_shot_ids = np.random.choice(
                len(few_shot_dataset), n_shot, replace=False
            )
            few_shot_data = few_shot_dataset.select(few_shot_ids)
            formatted_few_shot_prompt = ""
            for inst in few_shot_data:
                formatted_few_shot_prompt += prompt.format(
                    choices=inst["choices"],
                    question=inst["question"].strip(),
                    answer=answers[inst["answer"]],
                )

        subject_data = dataset.select(
            np.argwhere(subjects == subject).flatten()
        ).select(range(mmlu_max_subject_size))
        for inst in subject_data:
            formatted_prompt = prompt.format(
                choices=inst["choices"],
                question=inst["question"].strip(),
                answer="",
            )
            x.append(
                formatted_description + formatted_few_shot_prompt + formatted_prompt
            )
            y.append(answers[inst[y_column]])

    return x, y


def preprocess_dataset(
    dataset,
    dataset_name,
    x_column,
    y_column,
    prompt,
    description,
    n_shot,
    few_shot_dataset,
    mmlu_max_subject_size,
):
    if "translation" in dataset.column_names:
        return preprocess_translation(dataset, x_column, y_column)
    elif ("coqa" in dataset_name.lower()) and len(prompt):
        return preprocess_coqa(dataset, x_column, y_column, prompt)
    elif ("babi_qa" in dataset_name.lower()) and len(prompt):
        return preprocess_babiqa(dataset, x_column, y_column, prompt)
    elif len(prompt):
        return preprocess_with_prompt(dataset, x_column, y_column, prompt)
    elif ("gsm8k" in dataset_name.lower()) and len(prompt):
        return preprocess_gsm8k(
            dataset,
            y_column,
            n_shot,
            description,
            prompt,
            few_shot_dataset,
            mmlu_max_subject_size,
        )
    elif ("mmlu" in dataset_name.lower()) and len(prompt):
        return preprocess_mmlu(
            dataset,
            y_column,
            n_shot,
            description,
            few_shot_dataset,
            mmlu_max_subject_size,
            prompt,
        )
    else:
        return dataset[x_column], dataset[y_column]
