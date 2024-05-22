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

}

MATCHING_PROMPTS = { "en": (
    "Given the fact, identify the corresponding words "
    "in the original sentence that help derive this fact. "
    "Please list all words that are related to the fact, "
    "in the order they appear in the original sentence, "
    "each word separated by comma.\nFact: {claim}\n"
    "Sentence: {sent}\nWords from sentence that helps to "
    "derive the fact, separated by comma: "
),
"ar": (
    "Given the fact, identify the corresponding words "
    "in the original sentence that help derive this fact. "
    "Please list all words that are related to the fact, "
    "in the order they appear in the original sentence, "
    "each word separated by comma.\nFact: {claim}\n"
    "Sentence: {sent}\nWords from sentence that helps to "
    "derive the fact, separated by comma: "
)}