from .constants import TOP_K, SEED
from functools import partial


def prepare_medqa(
    dataset,
    prompt,
    description,
    n_shot,
    few_shot_dataset_func,
    few_shot_prompt,
    instruct,
):
    """
    Prepares MedQA-USMLE-4-options dataset for evaluation.
    
    The dataset has:
    - question: the question text
    - options: dict with keys "A", "B", "C", "D" containing option texts
    - answer_idx: the correct answer (A, B, C, or D)
    - answer: the actual answer text
    """
    import numpy as np

    np.random.seed(SEED)

    few_shot_dataset = few_shot_dataset_func() if few_shot_dataset_func else None

    x, y = [], []
    
    # Prepare few-shot examples if needed
    formatted_few_shot_prompt = ""
    if n_shot > 0 and few_shot_dataset is not None:
        few_shot_ids = np.random.choice(
            len(few_shot_dataset), min(n_shot, len(few_shot_dataset)), replace=False
        )
        few_shot_data = few_shot_dataset.select(few_shot_ids)
        
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
                        question=inst["question"].strip(),
                        option_a=inst["options"]["A"],
                        option_b=inst["options"]["B"],
                        option_c=inst["options"]["C"],
                        option_d=inst["options"]["D"],
                        answer=inst["answer_idx"],
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
                        question=inst["question"].strip(),
                        option_a=inst["options"]["A"],
                        option_b=inst["options"]["B"],
                        option_c=inst["options"]["C"],
                        option_d=inst["options"]["D"],
                        answer=inst["answer_idx"],
                    )
                    + "\n"
                )

    # Process each instance in the dataset
    formatted_description = description if description else ""
    
    for inst in dataset:
        formatted_prompt = prompt.format(
            question=inst["question"].strip(),
            option_a=inst["options"]["A"],
            option_b=inst["options"]["B"],
            option_c=inst["options"]["C"],
            option_d=inst["options"]["D"],
            answer="",
        )
        
        if formatted_description:
            x.append(
                formatted_description
                + "\n\n"
                + formatted_few_shot_prompt
                + formatted_prompt
            )
        else:
            x.append(formatted_few_shot_prompt + formatted_prompt)
        
        # Use answer_idx as the target (A, B, C, or D)
        y.append(inst["answer_idx"])
    
    return x, y


def generate_medqa_instruct_config(description, few_shot_prompt, subset, end_answer=""):
    import datasets
    return {
        "name": "GBaker/MedQA-USMLE-4-options",
        "train_split": "train",
        "test_split": "test",
        "prepare_func": partial(
            prepare_medqa,
            prompt="Q:{question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\n"
            + end_answer,
            description=description,
            n_shot=5,
            few_shot_dataset_func=partial(
                datasets.load_dataset, 
                path="GBaker/MedQA-USMLE-4-options", 
                split="train"
            ),
            few_shot_prompt=few_shot_prompt,
            instruct=True,
        ),
        "dataset": "medqa",
        "subset": subset,
    }


CONFIG = {
    "medqa": {
        "name": "GBaker/MedQA-USMLE-4-options",
        "train_split": "train",
        "test_split": "test",
        "prepare_func": partial(
            prepare_medqa,
            prompt="Q:{question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nAnswer:{answer}",
            description="The following are multiple choice questions (with answers) about medical knowledge.",
            n_shot=0,
            few_shot_dataset_func=None,
            few_shot_prompt=None,
            instruct=False,
        ),
        "dataset": "medqa",
        "subset": "continuation",
    },
    "medqa_simple_instruct": generate_medqa_instruct_config(
        description="Given the following medical question and four candidate answers (A, B, C, and D), choose the best answer. Your response should contain only the selected option's letter (A, B, C, or D), not a complete sentence.",
        few_shot_prompt="Q:{question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nAnswer:{answer}",
        subset="simple_instruct",
        end_answer="Answer:{answer}",
    ),
}

