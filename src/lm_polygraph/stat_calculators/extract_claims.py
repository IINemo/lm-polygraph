import numpy as np
import re

from typing import Dict, List, Optional
from dataclasses import dataclass

from .stat_calculator import StatCalculator
from lm_polygraph.utils.openai_chat import OpenAIChat
from lm_polygraph.utils.model import WhiteboxModel


@dataclass
class Claim:
    claim_text: str
    # The sentence of the generation, from which the claim was extracted
    sentence: str
    # Indices in the original generation of the tokens, which are related to the current claim
    aligned_tokens: List[int]


CLAIM_EXTRACTION_PROMPT = '''Please breakdown the sentence into independent claims.

Example:
Sentence: "He was born in London and raised by his mother and father until 11 years old."
Claims:
- He was born in London.
- He was raised by his mother and father.
- He was raised by his mother and father until 11 years old.

Sentence: "{sent}"
Claims:'''

MATCHING_PROMPT = '''Given the fact, identify the corresponding words in the original sentence ''' + \
    '''that help derive this fact. Please list all words that are related to the fact, ''' + \
    '''in the order they appear in the original sentence, each word separated by comma.\n''' + \
    '''Fact: {claim}\n''' + \
    '''Sentence: {sent}\n''' + \
    '''Words from sentence that helps to derive the fact, separated by comma: '''


class ClaimsExtractor(StatCalculator):
    """
    Extracts claims from the text of the model generation.
    """

    def __init__(self, openai_chat: OpenAIChat, sent_separators: str = ".?!\n"):
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

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        *args,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Extracts the claims out of each generation text.
        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, which includes:
                * 'greedy_log_probs' (List[List[float]]): log-probabilities of the generation tokens.
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
        Returns:
            Dict[str, np.ndarray]: dictionary with list of claims of type
            lm_polygraph.stat_calculators.extract_claims.Claim for each input text, calculated at
            'claims' key.
        """
        greedy_texts = dependencies["greedy_texts"]
        greedy_tokens = dependencies["greedy_tokens"]
        claims: List[List[Claim]] = []
        claim_texts_concatenated: List[str] = []
        claim_input_texts_concatenated: List[str] = []

        for greedy_text, greedy_tok, inp_text in zip(
            greedy_texts,
            greedy_tokens,
            texts,
        ):
            claims.append(self.claims_from_text(
                greedy_text,
                greedy_tok,
                model.tokenizer,
            ))
            for c in claims[-1]:
                claim_texts_concatenated.append(c.claim_text)
                claim_input_texts_concatenated.append(inp_text)

        return {
            "claims": claims,
            "claim_texts_concatenated": claim_texts_concatenated,
            "claim_input_texts_concatenated": claim_input_texts_concatenated,
        }

    def claims_from_text(
        self,
        text: str,
        tokens: List[int],
        tokenizer,
    ) -> List[Claim]:
        sentences = [
            s
            for s in re.split(f'[{self.sent_separators}]', text)
            if len(s) > 0
        ]
        if not any(text.endswith(x) for x in self.sent_separators):
            # remove last unfinished sentence
            sentences = sentences[:-1]

        sent_start_token_idx, sent_end_token_idx = 0, 0
        sent_start_idx, sent_end_idx = 0, 0
        claims = []
        for s in sentences:
            # Find sentence location in text: text[sent_start_idx:sent_end_idx]
            while not text[sent_start_idx:].startswith(s):
                sent_start_idx += 1
            while not text[:sent_end_idx].endswith(s):
                sent_end_idx += 1

            # Find sentence location in tokens: tokens[sent_start_token_idx:sent_end_token_idx]
            while len(tokenizer.decode(tokens[:sent_start_token_idx])) < sent_start_idx:
                sent_start_token_idx += 1
            while len(tokenizer.decode(tokens[:sent_end_token_idx])) < sent_end_idx:
                sent_end_token_idx += 1

            for c in self._claims_from_sentence(
                s,
                tokens[sent_start_token_idx:sent_end_token_idx],
                tokenizer,
            ):
                for i in range(len(c.aligned_tokens)):
                    c.aligned_tokens[i] += sent_start_token_idx
                claims.append(c)
        return claims

    def _claims_from_sentence(
        self,
        sent: str,
        sent_tokens: List[int],
        tokenizer,
    ) -> List[Claim]:
        extracted_claims = self.openai_chat.ask(
            CLAIM_EXTRACTION_PROMPT.format(sent=sent)
        )
        claims = []
        for claim_text in extracted_claims.split('\n'):
            if not claim_text.startswith("- "):
                continue
            if "there aren't any claims" in claim_text.lower():
                continue
            claim_text = claim_text[2:].strip()
            match_words = self.openai_chat.ask(
                MATCHING_PROMPT.format(sent=sent, claim=claim_text)
            ).strip().split(', ')
            match_string = self._match_string(sent, match_words)
            if match_string is None:
                continue
            aligned_tokens = self._align(sent, match_string, sent_tokens, tokenizer)
            if len(aligned_tokens) == 0:
                continue
            claims.append(Claim(
                claim_text=claim_text,
                sentence=sent,
                aligned_tokens=aligned_tokens,
            ))
        return claims

    def _match_string(self, sent: str, match_words: List[str]) -> Optional[str]:
        # Greedily matching words from `match_words` to `sent`.
        # Returns None if mathcing failed, e.g. due to words in match_words, which are not present
        # in sent, or of the words are specified not in the same order they appear in the sentence.
        #
        # Example:
        # sent = 'Lanny Flaherty is an American actor born on December 18, 1949, in Pensacola, Florida.'
        # match_words = ['Lanny', 'Flaherty', 'born', 'on', 'December', '18', '1949']
        # return '^^^^^ ^^^^^^^^                      ^^^^ ^^ ^^^^^^^^ ^^  ^^^^                        '

        last = 0  # pointer to the sentence
        last_match = 0  # pointer to the match_words list
        match_str = ''
        while last < len(sent):
            check_boundaries = (last == 0 or not sent[last - 1].isalpha())
            if check_boundaries and last_match < len(match_words):
                right_idx = last + len(match_words[last_match])
                if right_idx < len(sent):
                    check_boundaries = not sent[right_idx].isalpha()

            if last_match < len(match_words) and check_boundaries and sent[last:].startswith(
                    match_words[last_match]):
                # match at sent[last] and match_words[last_match]
                len_w = len(match_words[last_match])
                last += len_w
                match_str += '^' * len_w
                last_match += 1
            else:
                # no match at sent[last]
                last += 1
                match_str += ' '

        if last_match < len(match_words):
            # didn't match all words to the sentence
            return None

        return match_str

    def _align(self, sent: str, match_str: str, sent_tokens: List[int], tokenizer) -> List[int]:
        last = 0
        last_token = 0
        aligned_tokens = []
        while last < len(sent):
            if last_token >= len(sent_tokens):
                return aligned_tokens
            cur_token = tokenizer.decode(sent_tokens[last_token])
            if len(cur_token) > 0 and sent[last:].startswith(cur_token):
                # if the match string corresponding to the token contains matches, add to answer
                if any(t == '^' for t in match_str[last:last + len(cur_token)]):
                    aligned_tokens.append(last_token)
                last_token += 1
                last += len(cur_token)
            else:
                last += 1
        return aligned_tokens
