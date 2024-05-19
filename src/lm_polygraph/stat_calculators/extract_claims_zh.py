import numpy as np

from typing import Dict, List, Optional
from dataclasses import dataclass
from .extract_claims import ClaimsExtractor, Claim
from lm_polygraph.utils.openai_chat import OpenAIChat
from lm_polygraph.utils.model import WhiteboxModel


@dataclass
class Claim:
    claim_text: str
    # The sentence of the generation, from which the claim was extracted
    sentence: str
    # Indices in the original generation of the tokens, which are related to the current claim
    aligned_tokens: List[int]


CLAIM_EXTRACTION_PROMPT = """请将下面的句子(sentence)分解成独立的命题(independent claim)。

例子(Example):
Sentence: \"他出生在伦敦并且十一岁之前由他的父母抚养长大。\"

Claims:
- 他出生在伦敦。
- 他由他的父母抚养长大。
- 直到11岁前他由他的父母抚养长大。

Sentence: \"{sent}\"
Claims:"""


MATCHING_PROMPT = (
    "给定一个事实(Fact)，找出原句子中帮助推导出这个事实的相应词。"
    "请按照它们在原句子中出现的顺序列出所有与事实相关的词，每个词之间用空格分隔。\nFact: {claim}\n"
    "Sentence: {sent}\n: Output:"
)


class ClaimsExtractorZH(ClaimsExtractor):
    """
    Extracts claims from the text of the model generation.
    """

    def __init__(self, openai_chat: OpenAIChat, sent_separators: str = ".?!。？！\n"):
        super().__init__(
            [
                "claims",
                "claim_texts_concatenated",
                "claim_input_texts_concatenated",
            ],
            [
                "greedy_texts",
                "greedy_tokens",
            ],
        )
        self.openai_chat = openai_chat
        self.sent_separators = sent_separators


    def _match_string(self, sent: str, match_words: List[str]) -> Optional[str]:
        # Greedily matching characters from `match_words` to `sent`.
        # Returns None if matching failed, e.g. due to characters in match_words, which are not present
        # in sent, or if the characters are not in the same order they appear in the sentence.
        #
        # Example:
        # sent = 'Lanny Flaherty is an American actor born on December 18, 1949, in Pensacola, Florida.'
        # match_words = ['Lanny', 'Flaherty', 'born', 'on', 'December', '18', '1949']
        # return '^^^^^ ^^^^^^^^                      ^^^^ ^^ ^^^^^^^^ ^^  ^^^^                        '
    
        last = 0  # pointer to the sentence
        last_match = 0  # pointer to the match_words list
        match_str = ""
    
        # Iterate through each character in the input Chinese text
        for char in sent:
            # Check if the current character matches the next character in match_words[last_match]
            if last_match < len(match_words) and char == match_words[last_match][last]:
                # Match found, update pointers and match_str
                match_str += "^"
                last += 1
                if last == len(match_words[last_match]):
                    last = 0
                    last_match += 1
            else:
                # No match, append a space to match_str
                match_str += " "
    
        # Check if all characters in match_words have been matched
        if last_match < len(match_words):
            return None  # Didn't match all characters to the sentence
    
        return match_str
