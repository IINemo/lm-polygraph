from functools import partial


def prepare_base(
    dataset, input_column, output_column, prompt=None
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    x, y = dataset[input_column], dataset[output_column]
    if prompt:
        for i in range(len(x)):
            x[i] = prompt.format(text=x[i])
    return x, y


CONFIG = {
    "trivia_qa_tiny": {
        "name": "SpeedOfMagic/trivia_qa_tiny",
        "train_split": "train",
        "test_split": "test",
        "prepare_func": partial(
            prepare_base, input_column="question", output_column="answer"
        ),
    },
    "aeslc": {
        "name": "aeslc",
        "train_split": "train",
        "test_split": "test",
        "prepare_func": partial(
            prepare_base,
            input_column="email_body",
            output_column="subject_line",
            # prompt is set but not used in LM-Polygraph for this dataset (bug)
            # prompt="Write a short subject line for the email. Output only the subject line itself.\n\nEmail:\n{text}\n\nSubject line:\n",
        ),
    },
    "gsm8k": {
        "name": ["gsm8k", "main"],
        "train_split": "train",
        "test_split": "test",
        "prepare_func": partial(
            prepare_base,
            input_column="question",
            output_column="answer",
            prompt="Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.\n\nQ: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.\n\nQ: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.\n\nQ: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nA: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.\n\nQ: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nA: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.\n\nQ: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nA: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.\n\nQ: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.\n\nQ: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.\n\nQ: {text}\nA:",
        ),
    },
    "xsum": {
        "name": "xsum",
        "train_split": "train",
        "test_split": "test",
        "prepare_func": partial(
            prepare_base,
            input_column="document",
            output_column="summary",
            prompt="Here's the text and it's short one-sentence summary.\n\nText:\n{text}\n\nSummary (one sentence):\n",
        ),
    },
}
