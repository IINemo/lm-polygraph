from functools import partial


def prepare_babi_qa(dataset, input_column, output_column, prompt):
    x, y = [], []
    for inst in dataset:
        inst = inst["story"]
        context = ""
        for text, answer in zip(inst[input_column], inst[output_column]):
            if answer == "":
                context += text + " "
            else:
                x.append(prompt.format(context=context.strip(), question=text))
                y.append(answer)
    return x, y


CONFIG = {
    "babi_qa": {
        "name": ["facebook/babi_qa", "en-10k-qa1"],
        "train_split": "train",
        "test_split": "test",
        "prepare_func": partial(
            prepare_babi_qa,
            input_column="text",
            output_column="answer",
            prompt="Imagine that you are only able to say a single word. Answer the question given a context. You must only output the full name of the location the same way it is mentioned in the text. Do not try to be polite of helpful.\n\nExample:\n\nContext:\nMary moved to the bathroom. John went to the hallway. Daniel went back to the hallway. Sandra moved to the garden. John moved to the office. Sandra journeyed to the bathroom. Mary moved to the hallway. Daniel travelled to the office. John went back to the garden. John moved to the bedroom.\nQuestion:\nWhere is Sandra?\nAnswer:\nbathroom\n\nContext:\n{context}\n\nQuestion:\n{question}\nAnswer:\n",
        ),
    },
}
