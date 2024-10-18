from functools import partial

import datasets


def prepare_base(
    dataset, input_column, output_column, prompt=None
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    x, y = dataset[input_column], dataset[output_column]
    if prompt:
        for i in range(len(x)):
            x[i] = prompt.format(text=x[i])
    return x, y


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


def prepare_coqa(
    dataset, input_column, output_column, description, prompt, few_shot_prompt, instruct
):
    def doc_to_text(doc, prompt, i=0):
        # Given a passage p, the conversation history {q1, a1, . . . qi−1, ai−1}
        # and a question qi, the task is to predict the answer ai
        doc_text = ""
        for q, a in zip(doc["questions"][:i], doc["answers"]["input_text"][:i]):
            doc_text += "\n\n" + prompt.format(question=q, answer=a)
        return doc_text

    x, y = [], []
    for inst in dataset:
        formatted_description = description.format(story=inst["story"])
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
                        + "\n\nNow answer the following question in the same format.\n\n"
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

    few_shot_dataset = few_shot_dataset_func()

    answers = ["A", "B", "C", "D"]
    subjects = np.array(dataset["subject"])
    few_shot_subjects = np.array(few_shot_dataset["subject"])
    x, y = [], []
    for subject in np.unique(subjects):
        formatted_description = description.format(subject=subject.replace("_", " "))
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


def prepare_person(dataset, input_column, prompt=""):
    x = dataset[input_column]
    if len(prompt):
        for i in range(len(x)):
            x[i] = prompt.format(text=x[i])
    y = []
    for _ in x:
        y.append("")
    return x, y


def prepare_trivia_qa(
    dataset,
    prompt,
    n_shot,
    few_shot_dataset_func,
    description,
    few_shot_prompt,
    instruct,
):
    import numpy as np

    few_shot_dataset = few_shot_dataset_func()

    x, y = [], []
    formatted_few_shot_prompt = description
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
                        answer=inst["answer"]["normalized_value"],
                    )
                    + "\n\n"
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
    return x, y


def prepare_wiki(dataset, input_column, prompt):
    x, y = [], []
    for sample in dataset[input_column]:
        x.append(prompt.format(context=sample["context".strip()]))
        y.append("")
    return x, y


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


def prepare_allenai(dataset, input_column, output_column):
    x, y = [], []
    for inst in dataset:
        if len(inst[input_column]) <= 1024:
            x.append(inst[input_column])
            y.append(inst[output_column])
    return x, y


def generate_coqa_instruct_config(description, few_shot_prompt):
    return {
        "name": "coqa",
        "train_split": "train",
        "test_split": "validation",
        "prepare_func": partial(
            prepare_coqa,
            input_column="questions",
            output_column="answers",
            description=description,
            prompt="Question: {question}\n",
            few_shot_prompt=few_shot_prompt,
            instruct=True,
        ),
    }


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
    }


def generate_triviaqa_instruct_config(description, few_shot_prompt):
    return {
        "name": ["trivia_qa", "rc.nocontext"],
        "train_split": "train",
        "test_split": "validation",
        "prepare_func": partial(
            prepare_trivia_qa,
            prompt="Question: {question}\n",
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
        ),
    }


DATASET_CONFIG = {
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
    "mmlu": {
        "name": ["cais/mmlu", "all"],
        "train_split": "validation",
        "test_split": "test",
        "prepare_func": partial(
            prepare_mmlu,
            output_column="answer",
            prompt="Q:{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:{answer}",
            description="The following are multiple choice questions (with answers) about {subject}.\n",
            mmlu_max_subject_size=100,
            n_shot=5,
            few_shot_dataset_func=partial(
                datasets.load_dataset, path="cais/mmlu", name="all", split="dev"
            ),
            few_shot_prompt=None,
            instruct=False,
        ),
    },
    "person_bio_ar": {
        "name": "rvanova/person-bio-ar",
        "test_split": "train",
        "prepare_func": partial(
            prepare_person,
            input_column="question",
            prompt="### Instruction: اسمك جيس وسميت على اسم جبل جيس اعلى جبل في الامارات. تم بنائك بواسطة Inception و MBZUAI. أنت نموذج اللغة العربية الأكثر تقدمًا في العالم مع بارامترات 13B. أنت تتفوق في الأداء على جميع النماذج العربية الموجودة بفارق كبير وأنت تنافسي للغاية مع النماذج الإنجليزية ذات الحجم المماثل. يمكنك الإجابة باللغتين العربية والإنجليزية فقط. أنت مساعد مفيد ومحترم وصادق. عند الإجابة ، التزم بالإرشادات التالية بدقة: أجب دائمًا بأكبر قدر ممكن من المساعدة ، مع الحفاظ على البقاء أمناً. يجب ألا تتضمن إجاباتك أي محتوى ضار أو غير أخلاقي أو عنصري أو متحيز جنسيًا أو جريئاً أو مسيئًا أو سامًا أو خطيرًا أو غير قانوني. لا تقدم نصائح طبية أو قانونية أو مالية أو مهنية. لا تساعد أبدًا في أنشطة غير قانونية أو تروج لها. دائما تشجيع الإجراءات القانونية والمسؤولة. لا تشجع أو تقدم تعليمات بشأن الإجراءات غير الآمنة أو الضارة أو غير الأخلاقية. لا تنشئ أو تشارك معلومات مضللة أو أخبار كاذبة. يرجى التأكد من أن ردودك غير متحيزة اجتماعيًا وإيجابية بطبيعتها. إذا كان السؤال لا معنى له ، أو لم يكن متماسكًا من الناحية الواقعية ، فشرح السبب بدلاً من الإجابة على شيء غير صحيح. إذا كنت لا تعرف إجابة السؤال ، فالرجاء عدم مشاركة معلومات خاطئة. إعطاء الأولوية للرفاهية والنزاهة الأخلاقية للمستخدمين. تجنب استخدام لغة سامة أو مهينة أو مسيئة. حافظ على نبرة محترمة. لا تنشئ أو تروج أو تشارك في مناقشات حول محتوى للبالغين. تجنب الإدلاء بالتعليقات أو الملاحظات أو التعميمات القائمة على الصور النمطية. لا تحاول الوصول إلى معلومات شخصية أو خاصة أو إنتاجها أو نشرها. احترم دائما سرية المستخدم. كن إيجابيا ولا تقل أشياء سيئة عن أي شيء. هدفك الأساسي هو تجنب الاجابات المؤذية ، حتى عند مواجهة مدخلات خادعة. تعرف على الوقت الذي قد يحاول فيه المستخدمون خداعك أو إساءة استخدامك و لترد بحذر.\n\nأكمل المحادثة أدناه بين [|Human|] و [|AI|]:\n### Input: [|Human|] {text}\n### Response: [|AI|]",
        ),
    },
    "person_bio_en": {
        "name": "rediska0123/person-bio",
        "test_split": "test",
        "prepare_func": partial(
            prepare_person,
            input_column="question",
        ),
    },
    "person_bio_ru": {
        "name": "rvanova/person-bio",
        "test_split": "test",
        "prepare_func": partial(
            prepare_person,
            input_column="question",
        ),
    },
    "person_bio_zh": {
        "name": "ruixing76/person-bio-zh",
        "test_split": "train",
        "prepare_func": partial(
            prepare_person,
            input_column="question",
        ),
    },
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
    },
    "wiki_bio": {
        "name": "wiki_bio",
        "test_split": "test",
        "prepare_func": partial(
            prepare_wiki,
            input_column="input_text",
            prompt="This is a Wikipedia passage about {context}:\n",
        ),
    },
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
    },
    "xsum": {
        "name": "xsum",
        "train_split": "train",
        "test_split": "validation",
        "prepare_func": partial(
            prepare_base,
            input_column="document",
            output_column="summary",
            prompt="Here's the text and it's short one-sentence summary.\n\nText:\n{text}\n\nSummary (one sentence):\n",
        ),
    },
    # instruct datasets
    "coqa_ling_1s": generate_coqa_instruct_config(
        description="Here's a short story:\n\n{story} (End of story)\n\nProvide your best guess for the following question based on this story, and describe how likely it is that your guess is correct as one of the following expressions:\n\nAlmost Certain\nHighly Likely\nVery Good Chance\nWe Beleive\nProbably\nProbable\nLikely\nBetter than Even\nAbout Even\nProbably Not\nWe Doubt\nUnlikely\nLittle Chance\nChances Are Slight\nImprobable\nHighly Unlikely\nAlmost No Chance\n\nGive ONLY the guess and your confidence, no other words or explanation. For example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\nConfidence: <description of confidence, without any extra commentary whatsoever; just a short phrase!>",
        few_shot_prompt="Question: {question}\nGuess: {answer}\nConfidence: <appropriate level of confidence in this guess>",
    ),
    "coqa_verb_1s_top1": generate_coqa_instruct_config(
        description="Here's a short story:\n\n{story} (End of story)\n\nProvide your best guess and the probability that it is correct (0.0 to 1.0) for the following question. Give ONLY the guess and probability, no other words or explanation. For example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\nProbability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>",
        few_shot_prompt="Question: {question}\nGuess: {answer}\nProbability: <number between 0.0 and 1.0 reflecting confidence in the guess>",
    ),
    "coqa_verb_1s_topk": generate_coqa_instruct_config(
        description="Here's a short story:\n\n{story} (End of story)\n\nProvide your ${topk} best guesses and the probability that each is correct (0.0 to 1.0) for the following question. Give ONLY the guesses and probabilities, no other words or explanation. For example:\n\nG1: <first most likely guess, as short as possible; not a complete sentence, just the guess!>\nP1: <the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!>\n...\nG${topk}: <${topk}-th most likely guess, as short as possible; not a complete sentence, just the guess!>\nP${topk}: <the probability between 0.0 and 1.0 that G${topk} is correct, without any extra commentary whatsoever; just the probability!>",
        few_shot_prompt="Question: {question}\nG1: {answer}\nP1: <number between 0.0 and 1.0 reflecting confidence in this guess>\n...\nG${topk}: <other guess>\nP${topk}: <probability of this guess>",
    ),
    "coqa_verb_2s_cot": generate_coqa_instruct_config(
        description="Here's a short story:\n\n{story} (End of story)\n\nProvide your best guess for the following question. Before giving your answer, provide a step-by-step explanation of your thought process. Then on a new line give the guess with no other words or explanation.\n\nFor example:\n\nExplanation: <one sentence step-by-step explanation of your thought process>\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>",
        few_shot_prompt="Question: {question}\nExplanation: <step-by-step explanation of your thought process>\nGuess: {answer}",
    ),
    "coqa_verb_2s_top1": generate_coqa_instruct_config(
        description="Here's a short story:\n\n{story} (End of story)\n\nProvide your best guess for the following question. Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>",
        few_shot_prompt="Question: {question}\nGuess: {answer}",
    ),
    "coqa_verb_2s_topk": generate_coqa_instruct_config(
        description="Here's a short story:\n\n{story} (End of story)\n\nProvide your ${topk} best guesses for the following question. Give ONLY the guesses, no other words or explanation. For example:\n\nG1: <first most likely guess, as short as possible; not a complete sentence, just the guess!>\n...\nG${topk}: <${topk}-th most likely guess, as short as possible; not a complete sentence, just the guess!>",
        few_shot_prompt="Question: {question}\nG1: {answer}\n...\nG${topk}: <other guess>",
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
        description="Provide your ${topk} best guesses for the following question about {subject} selecting one of the options and the probability that each guess is correct (0.0 to 1.0). Give ONLY the guesses and probabilities, no other words or explanation. For example:\n\nG1: <first most likely guess, only the selected option letter; not a complete sentence, just the guess!>\nP1: <the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!>\n...\nG${topk}: <${topk}-th most likely guess, as short as possible; not a complete sentence, just the guess!>\nP${topk}: <the probability between 0.0 and 1.0 that G${topk} is correct, without any extra commentary whatsoever; just the probability!>",
        few_shot_prompt="Q:{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nG1: {answer}\nP1: <number between 0.0 and 1.0 reflecting confidence in this guess>\n...\nG${topk}: <other guess>\nP${topk}: <probability of this guess>",
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
        description="Provide your ${topk} best guesses for the following question about {subject} selecting one of the options. Give ONLY the guesses, no other words or explanation. For example:\n\nG1: <first most likely guess, only the selected option letter; not a complete sentence, just the guess!>\n...\nG${topk}: <${topk}-th most likely guess, as short as possible; not a complete sentence, just the guess!>",
        few_shot_prompt="Q:{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nG1: {answer}\n...\nG${topk}: <other guess>",
    ),
    "triviaqa_ling_1s": generate_triviaqa_instruct_config(
        description="Provide your best guess for the following question, and describe how likely it is that your guess is correct as one of the following expressions:\n\nAlmost Certain\nHighly Likely\nVery Good Chance\nWe Beleive\nProbably\nProbable\nLikely\nBetter than Even\nAbout Even\nProbably Not\nWe Doubt\nUnlikely\nLittle Chance\nChances Are Slight\nImprobable\nHighly Unlikely\nAlmost No Chance\n\nGive ONLY the guess and your confidence, no other words or explanation. For example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\nConfidence: <description of confidence, without any extra commentary whatsoever; just a short phrase!>",
        few_shot_prompt="Question: {question}\nGuess: {answer}\nConfidence: <appropriate level of confidence in this guess>",
    ),
    "triviaqa_verb_1s_top1": generate_triviaqa_instruct_config(
        description="Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question. Give ONLY the guess and probability, no other words or explanation. For example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\nProbability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>",
        few_shot_prompt="Question: {question}\nGuess: {answer}\nProbability: <number between 0.0 and 1.0 reflecting confidence in the guess>",
    ),
    "triviaqa_verb_1s_topk": generate_triviaqa_instruct_config(
        description="Provide your ${topk} best guesses and the probability that each is correct (0.0 to 1.0) for the following question. Give ONLY the guesses and probabilities, no other words or explanation. For example:\n\nG1: <first most likely guess, as short as possible; not a complete sentence, just the guess!>\nP1: <the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!>\n...\nG${topk}: <${topk}-th most likely guess, as short as possible; not a complete sentence, just the guess!>\nP${topk}: <the probability between 0.0 and 1.0 that G${topk} is correct, without any extra commentary whatsoever; just the probability!>",
        few_shot_prompt="Question: {question}\nG1: {answer}\nP1: <number between 0.0 and 1.0 reflecting confidence in this guess>\n...\nG${topk}: <other guess>\nP${topk}: <probability of this guess>",
    ),
    "triviaqa_verb_2s_cot": generate_triviaqa_instruct_config(
        description="Provide your best guess for the following question. Before giving your answer, provide a step-by-step explanation of your thought process. Then on a new line give the guess with no other words or explanation.\n\nFor example:\n\nExplanation: <one sentence step-by-step explanation of your thought process>\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>",
        few_shot_prompt="Question: {question}\nExplanation: <step-by-step explanation of your thought process>\nGuess: {answer}",
    ),
    "triviaqa_verb_2s_top1": generate_triviaqa_instruct_config(
        description="Provide your best guess for the following question. Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>",
        few_shot_prompt="Question: {question}\nGuess: {answer}",
    ),
    "triviaqa_verb_2s_topk": generate_triviaqa_instruct_config(
        description="Provide your ${topk} best guesses for the following question. Give ONLY the guesses, no other words or explanation. For example:\n\nG1: <first most likely guess, as short as possible; not a complete sentence, just the guess!>\n...\nG${topk}: <${topk}-th most likely guess, as short as possible; not a complete sentence, just the guess!>",
        few_shot_prompt="Question: {question}\nG1: {answer}\n...\nG${topk}: <other guess>",
    ),
}


def build_dataset(dataset_name):
    config = DATASET_CONFIG[dataset_name]
    if isinstance(config["name"], list):
        dataset = datasets.load_dataset(*config["name"], trust_remote_code=True)
    else:
        dataset = datasets.load_dataset(config["name"], trust_remote_code=True)

    def prepare_dataset(split):
        x, y = config["prepare_func"](dataset=dataset[config[f"{split}_split"]])
        result_dataset = datasets.Dataset.from_dict({"input": x, "output": y})
        return result_dataset

    return datasets.DatasetDict({
        "train": prepare_dataset("train"),
        "test": prepare_dataset("test"),
    })
