CLAIM_EXTRACTION_PROMPTS = {"en" : """Please breakdown the sentence into independent claims.

Example:
Sentence: \"He was born in London and raised by his mother and father until 11 years old.\"
Claims:
- He was born in London.
- He was raised by his mother and father.
- He was raised by his mother and father until 11 years old.

Sentence: \"{sent}\"
Claims:"""
,
"ar": """فكك النص التالي الى ادعاءات منفصلة.

Example:
Sentence: \"طارق ذياب هو لاعب كرة قدم تونسي سابق.\"
Claims:
- طارق ذياب لاعب كرة قدم.
- طارق ذياب تونسي.
- طارق ذياب لاعب سابق.

الجملة: \"{sent}\"
الادعاءات:"""
,
"zh": """请将下面的句子(sentence)分解成独立的命题(independent claim)。

例子(Example):
Sentence: \"他出生在伦敦并且十一岁之前由他的父母抚养长大。\"

Claims:
- 他出生在伦敦。
- 他由他的父母抚养长大。
- 直到11岁前他由他的父母抚养长大。

Sentence: \"{sent}\"
Claims:"""
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
"ar": (
"""بناءً على الحقيقة، حدد الكلمات المقابلة في الجملة الأصلية التي تساعد في استنتاج هذه الحقيقة. يرجى سرد جميع الكلمات المتعلقة بالحقيقة، بالترتيب الذي تظهر به في الجملة الأصلية، وكل كلمة مفصولة بفاصلة.
الحقيقة: {claim}
الجملة: {sent}
الكلمات من الجملة التي تساعد في استنتاج الحقيقة، مفصولة بفاصلة: """
),
"zh": (
    "给定一个事实(Fact)，找出原句子中帮助推导出这个事实的相应词。"
    "请按照它们在原句子中出现的顺序列出所有与事实相关的词，每个词之间用空格分隔。\nFact: {claim}\n"
    "Sentence: {sent}\n: Output:"
)
}


OPENAI_FACT_CHECK_PROMPTS = {"en": ("""Question: {input}

Determine if all provided information in the following claim is true according to the most recent sources of information.

Claim: {claim}
"""
),
"ar": """
هل الادعاءات صحيحة وفقًا لأحدث مصادر المعلومات؟
أجب ب "نعم"، "لا" أو "لا يُعرف".
مثال:
السياق: أعطني سيرة ذاتية لألبرت أينشتاين.
الادعاء: وُلد في 14 مارس.
الجواب: نعم
السياق: أعطني سيرة ذاتية لألبرت أينشتاين.
الادعاء: وُلد في المملكة المتحدة.
الجواب: لا
الادعاء: {claim}
السياق: {input}
الجواب:
""",
"zh":(
    """Question: {input}

请根据最新的信息来源，确定以下声明（Claim）中提供的所有信息是否属实。

Claim: {claim}
"""
)
}

OPENAI_FACT_CHECK_SUMMARIZE_PROMPT = {
"en":
("""Question: {input}

Claim: {claim}

Is the following claim true?

Reply: {reply}

Summarize this reply into one word, whether the claim is true: "True", "False" or "Not known".
"""),

"zh":
("""Question: {input}

Claim: {claim}

以下的陈述是否正确？

Reply: {reply}

请将此回复总结为一个词，即该陈述是否正确："True"，"False"或"Not known"。
"""
)
}


# old prompt
# "zh":(
#     "根据最新的信息来源，这个命题是否正确？ "
#     """回答 "True"表明命题为真，"False"表明命题为假，"Not known"表明命题无法判断。"""
#     "\n\n"
#     "例子:\n"
#     "\n"
#     "Question: 介绍一下爱因斯坦。\n"
#     "Claim: 爱因斯坦出生于3月14日。\n"
#     "Answer: True\n"
#     "\n"
#     "Question: 介绍一下爱因斯坦。\n"
#     "Claim: 爱因斯坦出生在伦敦。\n"
#     "Answer: False\n"
#     "\n"
#     "Your input:\n"
#     "\n"
#     "Question: {input}\n"
#     "Claim: {claim}\n"
#     "Answer: "
# )