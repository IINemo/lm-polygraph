from .constants import TOP_K, SEED
from functools import partial


import datasets


def prepare_mmlu(
    dataset,
    output_column,
    prompt,
    description,
    mmlu_max_subject_size,
    n_shot,
    few_shot_dataset_func,
    few_shot_prompt,
    instruct,
):
    import numpy as np

    np.random.seed(SEED)

    few_shot_dataset = few_shot_dataset_func()

    answers = ["A", "B", "C", "D"]
    subjects = np.array(dataset["subject"])
    few_shot_subjects = np.array(few_shot_dataset["subject"])
    x, y = [], []
    for subject in np.unique(subjects):
        formatted_description = description.format(
            subject=subject.replace("_", " "), topk=TOP_K
        )
        if n_shot > 0:
            few_shot_subject = few_shot_dataset.select(
                np.argwhere(few_shot_subjects == subject).flatten()
            )
            few_shot_ids = np.random.choice(
                len(few_shot_subject), n_shot, replace=False
            )
            few_shot_data = few_shot_subject.select(few_shot_ids)
            if instruct:
                assert (
                    few_shot_prompt is not None
                ), "separate few_shot_prompt must be provided for instruction mode."
                formatted_few_shot_prompt = (
                    "Here are a few examples of questions and answers:\n\n"
                )
                for inst in few_shot_data:
                    formatted_few_shot_prompt += (
                        few_shot_prompt.format(
                            choices=inst["choices"],
                            question=inst["question"].strip(),
                            answer=answers[inst["answer"]],
                            topk=TOP_K,
                        )
                        + "\n\n"
                    )
                formatted_few_shot_prompt += (
                    "Now answer the following question in the same format:\n\n"
                )
            else:
                formatted_few_shot_prompt = ""
                for inst in few_shot_data:
                    formatted_few_shot_prompt += (
                        prompt.format(
                            choices=inst["choices"],
                            question=inst["question"].strip(),
                            answer=answers[inst["answer"]],
                        )
                        + "\n"
                    )

        subject_data = dataset.select(np.argwhere(subjects == subject).flatten())

        if len(subject_data) > mmlu_max_subject_size:
            subject_data = subject_data.select(range(mmlu_max_subject_size))

        for inst in subject_data:
            formatted_prompt = prompt.format(
                choices=inst["choices"],
                question=inst["question"].strip(),
                answer="",
            )
            x.append(
                formatted_description
                + "\n\n"
                + formatted_few_shot_prompt
                + formatted_prompt
            )
            y.append(answers[inst[output_column]])
    return x, y


def generate_mmlu_instruct_config(description, few_shot_prompt):
    return {
        "name": ["cais/mmlu", "all"],
        "train_split": "validation",
        "test_split": "test",
        "prepare_func": partial(
            prepare_mmlu,
            output_column="answer",
            prompt="Q:{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:{answer}",
            description=description,
            mmlu_max_subject_size=100,
            n_shot=5,
            few_shot_dataset_func=partial(
                datasets.load_dataset, path="cais/mmlu", name="all", split="dev"
            ),
            few_shot_prompt=few_shot_prompt,
            instruct=True,
        ),
        "is_main_dataset": False,
    }


CONFIG = {
    "mmlu": {
        "name": ["cais/mmlu", "all"],
        "train_split": "validation",
        "test_split": "test",
        "prepare_func": partial(
            prepare_mmlu,
            output_column="answer",
            prompt="Q:{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:{answer}",
            description="The following are multiple choice questions (with answers) about {subject}.",
            mmlu_max_subject_size=100,
            n_shot=5,
            few_shot_dataset_func=partial(
                datasets.load_dataset, path="cais/mmlu", name="all", split="dev"
            ),
            few_shot_prompt=None,
            instruct=False,
        ),
    },
    "mmlu_empirical_baselines": generate_mmlu_instruct_config(
        description="Provide your best guess for the following question about {subject} selecting one of the options. Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, only the selected option letter; not a complete sentence, just the guess!>",
        few_shot_prompt="Q:{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nGuess:{answer}",
    ),
    "mmlu_ling_1s": generate_mmlu_instruct_config(
        description="Provide your best guess for the following question about {subject} selecting one of the options, and describe how likely it is that your guess is correct as one of the following expressions:\n\nAlmost Certain\nHighly Likely\nVery Good Chance\nWe Beleive\nProbably\nProbable\nLikely\nBetter than Even\nAbout Even\nProbably Not\nWe Doubt\nUnlikely\nLittle Chance\nChances Are Slight\nImprobable\nHighly Unlikely\nAlmost No Chance\n\nGive ONLY the guess and your confidence, no other words or explanation. For example:\n\nGuess: <most likely guess, only the selected option letter; not a complete sentence, just the guess!>\nConfidence: <description of confidence, without any extra commentary whatsoever; just a short phrase!>",
        few_shot_prompt="Q:{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nGuess:{answer}\nConfidence: <appropriate level of confidence in this guess>",
    ),
    "mmlu_verb_1s_top1": generate_mmlu_instruct_config(
        description="Provide your best guess for the following question about {subject} selecting one of the options and the probability that it is correct (0.0 to 1.0). Give ONLY the guess and probability, no other words or explanation. For example:\n\nGuess: <most likely guess, only the selected option letter; not a complete sentence, just the guess!>\nProbability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>",
        few_shot_prompt="Q:{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nGuess:{answer}\nProbability: <number between 0.0 and 1.0 reflecting confidence in the guess>",
    ),
    "mmlu_verb_1s_topk": generate_mmlu_instruct_config(
        description="Provide your {topk} best guesses for the following question about {subject} selecting one of the options and the probability that each guess is correct (0.0 to 1.0). Give ONLY the guesses and probabilities, no other words or explanation. For example:\n\nG1: <first most likely guess, only the selected option letter; not a complete sentence, just the guess!>\nP1: <the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!>\n...\nG{topk}: <{topk}-th most likely guess, as short as possible; not a complete sentence, just the guess!>\nP{topk}: <the probability between 0.0 and 1.0 that G{topk} is correct, without any extra commentary whatsoever; just the probability!>",
        few_shot_prompt="Q:{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nG1: {answer}\nP1: <number between 0.0 and 1.0 reflecting confidence in this guess>\n...\nG{topk}: <other guess>\nP{topk}: <probability of this guess>",
    ),
    "mmlu_verb_2s_cot": generate_mmlu_instruct_config(
        description="Provide your best guess for the following question about {subject} selecting one of the options. Before giving your answer, provide a step-by-step explanation of your thought process. Then on a new line give the guess with no other words or explanation.\n\nFor example:\n\nExplanation: <one sentence step-by-step explanation of your thought process>\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>",
        few_shot_prompt="Q:{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nExplanation: <step-by-step explanation of your thought process>\nGuess:{answer}",
    ),
    "mmlu_verb_2s_top1": generate_mmlu_instruct_config(
        description="Provide your best guess for the following question about {subject} selecting one of the options. Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, only the selected option letter; not a complete sentence, just the guess!>",
        few_shot_prompt="Q:{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nGuess:{answer}",
    ),
    "mmlu_verb_2s_topk": generate_mmlu_instruct_config(
        description="Provide your {topk} best guesses for the following question about {subject} selecting one of the options. Give ONLY the guesses, no other words or explanation. For example:\n\nG1: <first most likely guess, only the selected option letter; not a complete sentence, just the guess!>\n...\nG{topk}: <{topk}-th most likely guess, as short as possible; not a complete sentence, just the guess!>",
        few_shot_prompt="Q:{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nG1: {answer}\n...\nG{topk}: <other guess>",
    ),
}
