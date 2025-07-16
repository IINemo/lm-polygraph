from .constants import TOP_K
from functools import partial


def prepare_coqa(
    dataset,
    input_column,
    output_column,
    description,
    prompt,
    few_shot_prompt,
    instruct,
    few_shot_prompt_end,
):
    def doc_to_text(doc, prompt, i=0):
        # Given a passage p, the conversation history {q1, a1, . . . qi−1, ai−1}
        # and a question qi, the task is to predict the answer ai
        doc_text = ""
        for q, a in zip(doc["questions"][:i], doc["answers"]["input_text"][:i]):
            doc_text += "\n\n" + prompt.format(question=q, answer=a, topk=TOP_K)
        return doc_text

    x, y = [], []
    for inst in dataset:
        formatted_description = description.format(story=inst["story"], topk=TOP_K)
        for j, (question, answer) in enumerate(
            zip(inst[input_column], inst[output_column]["input_text"])
        ):
            if instruct:
                assert (
                    few_shot_prompt is not None
                ), "separate few_shot_prompt must be provided for instruction mode."
                few_shot_section = doc_to_text(inst, few_shot_prompt, j)
                if few_shot_section != "":
                    few_shot_section = (
                        "\n\nHere are a few examples of questions and answers:"
                        + few_shot_section
                        + f"\n\n{few_shot_prompt_end}\n\n"
                    )
                else:
                    few_shot_section = "\n\n"
            else:
                few_shot_section = doc_to_text(inst, prompt, j) + "\n\n"
            formatted_prompt = (
                formatted_description
                + few_shot_section
                + prompt.format(
                    question=question,
                    answer="",
                )
            )
            x.append(formatted_prompt)
            y.append(answer)
    return x, y


def generate_coqa_instruct_config(
    subset,
    description,
    few_shot_prompt,
    end_answer="",
    few_shot_prompt_end="Now answer the following question in the same format.",
):
    return {
        "name": "coqa",
        "train_split": "train",
        "test_split": "validation",
        "prepare_func": partial(
            prepare_coqa,
            input_column="questions",
            output_column="answers",
            description=description,
            prompt="Question: {question}\n" + end_answer,
            few_shot_prompt=few_shot_prompt,
            instruct=True,
            few_shot_prompt_end=few_shot_prompt_end,
        ),
        "dataset": "coqa",
        "subset": subset,
    }


CONFIG = {
    "coqa": {
        "name": "coqa",
        "train_split": "train",
        "test_split": "validation",
        "prepare_func": partial(
            prepare_coqa,
            input_column="questions",
            output_column="answers",
            description="The following are stories and questions about them. Each story is followed by a question and answer to a given question.\n\nStory: {story}",
            prompt="Question: {question}\nAnswer:{answer}",
            few_shot_prompt=None,
            instruct=False,
        ),
        "dataset": "coqa",
        "subset": "continuation",
    },
    "coqa_empirical_baselines": generate_coqa_instruct_config(
        subset="empirical_baselines",
        description="Here's a short story:\n\n{story} (End of story)\n\nProvide your best guess for the following question. Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>",
        few_shot_prompt="Question: {question}\nGuess: {answer}",
    ),
    "coqa_ling_1s": generate_coqa_instruct_config(
        subset="ling_1s",
        description="Here's a short story:\n\n{story} (End of story)\n\nProvide your best guess for the following question based on this story, and describe how likely it is that your guess is correct as one of the following expressions:\n\nAlmost Certain\nHighly Likely\nVery Good Chance\nWe Beleive\nProbably\nProbable\nLikely\nBetter than Even\nAbout Even\nProbably Not\nWe Doubt\nUnlikely\nLittle Chance\nChances Are Slight\nImprobable\nHighly Unlikely\nAlmost No Chance\n\nGive ONLY the guess and your confidence, no other words or explanation. For example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\nConfidence: <description of confidence, without any extra commentary whatsoever; just a short phrase!>",
        few_shot_prompt="Question: {question}\nGuess: {answer}\nConfidence: <appropriate level of confidence in this guess>",
    ),
    "coqa_verb_1s_top1": generate_coqa_instruct_config(
        subset="1s_top1",
        description="Here's a short story:\n\n{story} (End of story)\n\nProvide your best guess and the probability that it is correct (0.0 to 1.0) for the following question. Give ONLY the guess and probability, no other words or explanation. For example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\nProbability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>",
        few_shot_prompt="Question: {question}\nGuess: {answer}\nProbability: <number between 0.0 and 1.0 reflecting confidence in the guess>",
    ),
    "coqa_verb_1s_topk": generate_coqa_instruct_config(
        subset="1s_topk",
        description="Here's a short story:\n\n{story} (End of story)\n\nProvide your {topk} best guesses and the probability that each is correct (0.0 to 1.0) for the following question. Give ONLY the guesses and probabilities, no other words or explanation. For example:\n\nG1: <first most likely guess, as short as possible; not a complete sentence, just the guess!>\nP1: <the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!>\n...\nG{topk}: <{topk}-th most likely guess, as short as possible; not a complete sentence, just the guess!>\nP{topk}: <the probability between 0.0 and 1.0 that G{topk} is correct, without any extra commentary whatsoever; just the probability!>",
        few_shot_prompt="Question: {question}\nG1: {answer}\nP1: <number between 0.0 and 1.0 reflecting confidence in this guess>\n...\nG{topk}: <other guess>\nP{topk}: <probability of this guess>",
    ),
    "coqa_verb_2s_cot": generate_coqa_instruct_config(
        subset="2s_cot",
        description="Here's a short story:\n\n{story} (End of story)\n\nProvide your best guess for the following question. Before giving your answer, provide a step-by-step explanation of your thought process. Then on a new line give the guess with no other words or explanation.\n\nFor example:\n\nExplanation: <one sentence step-by-step explanation of your thought process>\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>",
        few_shot_prompt="Question: {question}\nExplanation: <step-by-step explanation of your thought process>\nGuess: {answer}",
    ),
    "coqa_verb_2s_top1": generate_coqa_instruct_config(
        subset="2s_top1",
        description="Here's a short story:\n\n{story} (End of story)\n\nProvide your best guess for the following question. Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>",
        few_shot_prompt="Question: {question}\nGuess: {answer}",
    ),
    "coqa_verb_2s_topk": generate_coqa_instruct_config(
        subset="2s_topk",
        description="Here's a short story:\n\n{story} (End of story)\n\nProvide your {topk} best guesses for the following question. Give ONLY the guesses, no other words or explanation. For example:\n\nG1: <first most likely guess, as short as possible; not a complete sentence, just the guess!>\n...\nG{topk}: <{topk}-th most likely guess, as short as possible; not a complete sentence, just the guess!>",
        few_shot_prompt="Question: {question}\nG1: {answer}\n...\nG{topk}: <other guess>",
    ),
    "coqa_simple_instruct": generate_coqa_instruct_config(
        subset="simple_instruct",
        description="Here's a short story:\n\n{story} (End of story)\n\nAnswer the following question as briefly as possible.",
        few_shot_prompt="Question: {question}\nAnswer: {answer}",
        end_answer="Answer: {answer}",
        few_shot_prompt_end="Now answer the following question.",
    ),
}
