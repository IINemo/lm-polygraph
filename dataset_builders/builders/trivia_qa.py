from .constants import TOP_K, SEED
from functools import partial
from .stripped_formatters import qa_stripped


import datasets


def prepare_trivia_qa(
    dataset,
    prompt,
    n_shot,
    few_shot_dataset_func,
    description,
    few_shot_prompt,
    instruct,
    few_shot_prompt_end="Now answer the following question in the same format:\n\n",
):
    import numpy as np

    np.random.seed(SEED)

    few_shot_dataset = few_shot_dataset_func()

    x, y, s = [], [], []
    formatted_few_shot_prompt = description.format(topk=TOP_K)
    if n_shot > 0:
        few_shot_ids = np.random.choice(len(few_shot_dataset), n_shot, replace=False)
        few_shot_data = few_shot_dataset.select(few_shot_ids)
        if instruct:
            assert (
                few_shot_prompt is not None
            ), "separate few_shot_prompt must be provided for instruction mode."
            formatted_few_shot_prompt += (
                "\n\nHere are a few examples of questions and answers:\n\n"
            )
            for inst in few_shot_data:
                formatted_few_shot_prompt += (
                    few_shot_prompt.format(
                        question=inst["question"].strip(),
                        answer=inst["answer"]["normalized_value"],
                        topk=TOP_K,
                    )
                    + "\n\n"
                )
            formatted_few_shot_prompt += few_shot_prompt_end
        else:
            formatted_few_shot_prompt = ""
            for inst in few_shot_data:
                formatted_few_shot_prompt += (
                    prompt.format(
                        question=inst["question"].strip(),
                        answer=inst["answer"]["normalized_value"],
                    )
                    + "\n"
                )
    else:
        formatted_few_shot_prompt += "\n"

    for inst in dataset:
        if instruct:
            x.append(
                formatted_few_shot_prompt + prompt.format(question=inst["question"])
            )
        else:
            x.append(
                formatted_few_shot_prompt
                + prompt.format(question=inst["question"], answer="")
            )
        y.append([alias for alias in inst["answer"]["aliases"]])
        s.append(qa_stripped(inst["question"].strip()))
    return x, y, s


def generate_triviaqa_instruct_config(
    description,
    few_shot_prompt,
    subset,
    end_answer="",
    few_shot_prompt_end="Now answer the following question in the same format:\n\n",
):
    return {
        "name": ["trivia_qa", "rc.nocontext"],
        "train_split": "train",
        "test_split": "validation",
        "prepare_func": partial(
            prepare_trivia_qa,
            prompt="Question: {question}\n" + end_answer,
            n_shot=5,
            few_shot_dataset_func=partial(
                datasets.load_dataset,
                path="trivia_qa",
                name="rc.nocontext",
                split="train",
            ),
            description=description,
            few_shot_prompt=few_shot_prompt,
            instruct=True,
            few_shot_prompt_end=few_shot_prompt_end,
        ),
        "dataset": "triviaqa",
        "subset": subset,
    }


CONFIG = {
    "triviaqa": {
        "name": ["trivia_qa", "rc.nocontext"],
        "train_split": "train",
        "test_split": "validation",
        "prepare_func": partial(
            prepare_trivia_qa,
            prompt="Question: {question}\nAnswer:{answer}",
            n_shot=5,
            few_shot_dataset_func=partial(
                datasets.load_dataset,
                path="trivia_qa",
                name="rc.nocontext",
                split="train",
            ),
            description="",
            few_shot_prompt=None,
            instruct=False,
        ),
        "dataset": "triviaqa",
        "subset": "continuation",
    },
    "triviaqa_empirical_baselines": generate_triviaqa_instruct_config(
        description="Provide your best guess for the following question. Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>",
        few_shot_prompt="Question: {question}\nGuess: {answer}",
        subset="empirical_baselines",
    ),
    "triviaqa_ling_1s": generate_triviaqa_instruct_config(
        description="Provide your best guess for the following question, and describe how likely it is that your guess is correct as one of the following expressions:\n\nAlmost Certain\nHighly Likely\nVery Good Chance\nWe Beleive\nProbably\nProbable\nLikely\nBetter than Even\nAbout Even\nProbably Not\nWe Doubt\nUnlikely\nLittle Chance\nChances Are Slight\nImprobable\nHighly Unlikely\nAlmost No Chance\n\nGive ONLY the guess and your confidence, no other words or explanation. For example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\nConfidence: <description of confidence, without any extra commentary whatsoever; just a short phrase!>",
        few_shot_prompt="Question: {question}\nGuess: {answer}\nConfidence: <appropriate level of confidence in this guess>",
        subset="ling_1s",
    ),
    "triviaqa_verb_1s_top1": generate_triviaqa_instruct_config(
        description="Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question. Give ONLY the guess and probability, no other words or explanation. For example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\nProbability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>",
        few_shot_prompt="Question: {question}\nGuess: {answer}\nProbability: <number between 0.0 and 1.0 reflecting confidence in the guess>",
        subset="verb_1s_top1",
    ),
    "triviaqa_verb_1s_topk": generate_triviaqa_instruct_config(
        description="Provide your {topk} best guesses and the probability that each is correct (0.0 to 1.0) for the following question. Give ONLY the guesses and probabilities, no other words or explanation. For example:\n\nG1: <first most likely guess, as short as possible; not a complete sentence, just the guess!>\nP1: <the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!>\n...\nG{topk}: <{topk}-th most likely guess, as short as possible; not a complete sentence, just the guess!>\nP{topk}: <the probability between 0.0 and 1.0 that G{topk} is correct, without any extra commentary whatsoever; just the probability!>",
        few_shot_prompt="Question: {question}\nG1: {answer}\nP1: <number between 0.0 and 1.0 reflecting confidence in this guess>\n...\nG{topk}: <other guess>\nP{topk}: <probability of this guess>",
        subset="verb_1s_topk",
    ),
    "triviaqa_verb_2s_cot": generate_triviaqa_instruct_config(
        description="Provide your best guess for the following question. Before giving your answer, provide a step-by-step explanation of your thought process. Then on a new line give the guess with no other words or explanation.\n\nFor example:\n\nExplanation: <one sentence step-by-step explanation of your thought process>\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>",
        few_shot_prompt="Question: {question}\nExplanation: <step-by-step explanation of your thought process>\nGuess: {answer}",
        subset="verb_2s_cot",
    ),
    "triviaqa_verb_2s_top1": generate_triviaqa_instruct_config(
        description="Provide your best guess for the following question. Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>",
        few_shot_prompt="Question: {question}\nGuess: {answer}",
        subset="verb_2s_top1",
    ),
    "triviaqa_verb_2s_topk": generate_triviaqa_instruct_config(
        description="Provide your {topk} best guesses for the following question. Give ONLY the guesses, no other words or explanation. For example:\n\nG1: <first most likely guess, as short as possible; not a complete sentence, just the guess!>\n...\nG{topk}: <{topk}-th most likely guess, as short as possible; not a complete sentence, just the guess!>",
        few_shot_prompt="Question: {question}\nG1: {answer}\n...\nG{topk}: <other guess>",
        subset="verb_2s_topk",
    ),
    "triviaqa_simple_instruct": generate_triviaqa_instruct_config(
        description="Answer the following question as briefly as possible.",
        few_shot_prompt="Question: {question}\nAnswer: {answer}",
        subset="simple_instruct",
        end_answer="Answer: ",
        few_shot_prompt_end="Now answer the following question:",
    ),
}
