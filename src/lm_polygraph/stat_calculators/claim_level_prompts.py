CLAIM_EXTRACTION_PROMPTS = {
    "en": """Please breakdown the sentence into independent claims.

Example:
Sentence: \"He was born in London and raised by his mother and father until 11 years old.\"
Claims:
- He was born in London.
- He was raised by his mother and father.
- He was raised by his mother and father until 11 years old.

Sentence: \"{sent}\"
Claims:""",
    "zh": """请将下面的句子(sentence)分解成独立的命题(independent claim)。

例子(Example):
Sentence: \"他出生在伦敦并且十一岁之前由他的父母抚养长大。\"

Claims:
- 他出生在伦敦。
- 他由他的父母抚养长大。
- 直到11岁前他由他的父母抚养长大。

Sentence: \"{sent}\"
Claims:""",
    "ru": """Пожалуйста разбей предложение на независимые утверждения.

Example:
Sentence: \"Он родился в Лондоне и воспитывался матерью и отцом до 11 лет.\"
Claims:
- Он родился в Лондоне.
- Он воспитывался матерью и отцом.
- Он воспитывался матерью и отцом до 11 лет.

Sentence: \"{sent}\"
Claims:""",
    "ar": """فكك النص التالي الى ادعاءات منفصلة.

Example:
Sentence: \"طارق ذياب هو لاعب كرة قدم تونسي سابق.\"
Claims:
- طارق ذياب لاعب كرة قدم.
- طارق ذياب تونسي.
- طارق ذياب لاعب سابق.

الجملة: \"{sent}\"
الادعاءات:""",
}

MATCHING_PROMPTS = {
    "en": (
        "Given the fact, identify the corresponding words "
        "in the original sentence that help derive this fact. "
        "Please list all words that are related to the fact, "
        "in the order they appear in the original sentence, "
        "each word separated by comma.\nFact: {claim}\n"
        "Sentence: {sent}\nWords from sentence that helps to "
        "derive the fact, separated by comma: "
    ),
    "zh": (
        "给定一个事实(Fact)，找出原句子中帮助推导出这个事实的相应词。"
        "请按照它们在原句子中出现的顺序列出所有与事实相关的词，每个词之间用空格分隔。\nFact: {claim}\n"
        "Sentence: {sent}\n: Output:"
    ),
    "ru": (
        "Используя факт, определи соответствующие слова "
        "в исходном предложении, которые помогают получить этот факт "
        "Пожалуйста, перечисли все слова через запятую, "
        "имеющие отношение к данному не изменяя форм слов"
        "в том порядке, в котором они появляются в исходном предложении. "
        "Не используй кавычки при перечислении."
        "Не меняй форму слова"
        "\nFact: {claim}\n"
        "Sentence: {sent}\nСлова из предложения, которые помогают "
        "получить факт, через запятую без кавычек: "
    ),
    "ar": (
        """بناءً على الحقيقة، حدد الكلمات المقابلة في الجملة الأصلية التي تساعد في استنتاج هذه الحقيقة. يرجى سرد جميع الكلمات المتعلقة بالحقيقة، بالترتيب الذي تظهر به في الجملة الأصلية، وكل كلمة مفصولة بفاصلة.
الحقيقة: {claim}
الجملة: {sent}
الكلمات من الجملة التي تساعد في استنتاج الحقيقة، مفصولة بفاصلة: """
    ),
}


OPENAI_FACT_CHECK_PROMPTS = {
    "en": (
        """Question: {input}

Determine if all provided information in the following claim is true according to the most recent sources of information.

Claim: {claim}
"""
    ),
    "zh": (
        """Question: {input}

请根据最新的信息来源，确定以下声明（Claim）中提供的所有信息是否属实。

Claim: {claim}
"""
    ),
    "ru": (
        """Question: {input}\n
    Определи, соответствует ли вся предоставленная информация в следующем утверждении действительности согласно самым последним источникам информации.
    Если хотя бы часть утверждения неверна, склоняйся к выводу, что информация ложная.
    Think in English.
    Think step by step on how to summarize the claim within the provided <sketchpad>.
    Then, return a <summary> based on the <sketchpad>.

    Claim: {claim}
"""
    ),
    "ar": (
        """السؤال: {input}

حدد ما إذا كانت جميع المعلومات المقدمة في الادعاء التالي صحيحة وفقًا لأحدث مصادر المعلومات.

الادعاء: {claim}
"""
    ),
}

OPENAI_FACT_CHECK_SUMMARIZE_PROMPT = {
    "en": (
        """Question: {input}

Claim: {claim}

Is the following claim true?

Reply: {reply}

Summarize this reply into one word, whether the claim is true: "True", "False" or "Not known".
"""
    ),
    "zh": (
        """Question: {input}

Claim: {claim}

以下的表述是否正确？

Reply: {reply}

请用一个词回答该表述(Reply)是否正确："True"，"False"或"Not known"。
"""
    ),
    "ru": (
        """Question: {input}

Claim: {claim}

Is the following claim true?

Reply: {reply}

Summarize this reply into one word, whether the claim is true: "True", "False" or "Not known".
"""
    ),
    "ar": (
        """
السؤال: {input}

الادعاء: {claim}

هل الادعاء التالي صحيح؟

الاجابة: {reply}

قم بتلخيص هذه الجملة في كلمة واحدة، سواء كان الادعاء صحيح: "صحيح" أو "خطأ" أو "غير معروف".
"""
    ),
}
