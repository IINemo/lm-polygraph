import numpy as np
import re

from typing import Dict, List, Optional
from dataclasses import dataclass

from .stat_calculator import StatCalculator
from lm_polygraph.utils.openai_chat import OpenAIChat
from lm_polygraph.utils.model import WhiteboxModel
from .prompts import CLAIM_EXTRACTION_PROMPTS, MATCHING_PROMPTS

@dataclass
class Claim:
    claim_text: str
    # The sentence of the generation, from which the claim was extracted
    sentence: str
    # Indices in the original generation of the tokens, which are related to the current claim
    aligned_tokens: List[int]



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
        language: str = "en",
        *args,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Extracts the claims out of each generation text.
        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, which includes:
                * 'greedy_log_probs' (List[List[float]]): log-probabilities of the generation tokens.
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
        Returns:
            Dict[str, np.ndarray]: dictionary with :
                * 'claims' (List[List[lm_polygraph.stat_calculators.extract_claims.Claim]]):
                  list of claims for each input text;
                * 'claim_texts_concatenated' (List[str]): list of all textual claims extracted;
                * 'claim_input_texts_concatenated' (List[str]): for each claim in
                  claim_texts_concatenated, corresponding input text.
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
            claims.append(
                self.claims_from_text(
                    greedy_text,
                    greedy_tok,
                    model.tokenizer,
                    language
                )
            )
<<<<<<< HEAD
=======
            # Iterate over newly added claims to concatenate into list
>>>>>>> origin/claim_level
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
        language
    ) -> List[Claim]:
        sentences = []
        for s in re.split(f"[{self.sent_separators}]", text):
            if len(s) > 0:
                sentences.append(s)
<<<<<<< HEAD
        if not any(text.endswith(x) for x in self.sent_separators):
            # remove last unfinished sentence
=======
        if len(text) > 0 and text[-1] not in self.sent_separators:
            # Remove last unfinished sentence, because extracting claims
            # from unfinished sentence may lead to hallucinated claims.
>>>>>>> origin/claim_level
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

<<<<<<< HEAD
=======
            # Extract claims from current sentence
>>>>>>> origin/claim_level
            for c in self._claims_from_sentence(
                s,
                tokens[sent_start_token_idx:sent_end_token_idx],
                tokenizer,
                language
            ):
<<<<<<< HEAD
=======
                # Correct aligned tokens positions from sentence-level to generation-level
>>>>>>> origin/claim_level
                for i in range(len(c.aligned_tokens)):
                    c.aligned_tokens[i] += sent_start_token_idx
                claims.append(c)
        return claims

    def _claims_from_sentence(
        self,
        sent: str,
        sent_tokens: List[int],
        tokenizer,
        language
    ) -> List[Claim]:
<<<<<<< HEAD
=======
        # Extract claims with specific prompt
>>>>>>> origin/claim_level
        extracted_claims = self.openai_chat.ask(
            CLAIM_EXTRACTION_PROMPTS[language].format(sent=sent)
        )
        claims = []
        for claim_text in extracted_claims.split("\n"):
<<<<<<< HEAD
=======
            # Bad claim_text example:
            # - There aren't any claims in this sentence.
>>>>>>> origin/claim_level
            if not claim_text.startswith("- "):
                continue
            if "there aren't any claims" in claim_text.lower():
                continue
<<<<<<< HEAD
            claim_text = claim_text[2:].strip()
            chat_ask = MATCHING_PROMPTS[language].format(sent=sent, claim=claim_text)
            match_words = self.openai_chat.ask(chat_ask)
            match_words = match_words.strip().split(",")
            for i in range(len(match_words)):
                match_words[i] = match_words[i].strip()
            match_string = self._match_string(sent, match_words)
            if match_string is None:
                continue
=======
            # remove '- ' in the beginning
            claim_text = claim_text[2:].strip()
            # Get words which matches the claim using specific prompt
            chat_ask = MATCHING_PROMPT.format(sent=sent, claim=claim_text)
            match_words = self.openai_chat.ask(chat_ask)
            match_words = match_words.strip().split(",")
            match_words = list(map(lambda x: x.strip(), match_words))
            # Try to highlight matched symbols in sent
            match_string = self._match_string(sent, match_words)
            if match_string is None:
                continue
            # Get token positions which intersect with highlighted regions, that is, correspond to the claim
>>>>>>> origin/claim_level
            aligned_tokens = self._align(sent, match_string, sent_tokens, tokenizer)
            if len(aligned_tokens) == 0:
                continue
            claims.append(
                Claim(
                    claim_text=claim_text,
                    sentence=sent,
                    aligned_tokens=aligned_tokens,
                )
            )
        return claims

    def _match_string(self, sent: str, match_words: List[str]) -> Optional[str]:
<<<<<<< HEAD
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
        match_str = ""
        while last < len(sent):
=======
        """
        Greedily matching words from `match_words` to `sent`.
        Parameters:
            sent (str): sentence string
            match_words (List[str]): list of words from sent, in the same order they appear in it.
        Returns:
            Optional[str]: string of length len(sent), for each symbol in sent, '^' if it contains in one
                of the match_words if aligned to sent, ' ' otherwise.
                Returns None if matching failed, e.g. due to words in match_words, which are not present
                in sent, or of the words are specified not in the same order they appear in the sentence.
        Example:
            sent = 'Lanny Flaherty is an American actor born on December 18, 1949, in Pensacola, Florida.'
            match_words = ['Lanny', 'Flaherty', 'born', 'on', 'December', '18', '1949']
            return '^^^^^ ^^^^^^^^                      ^^^^ ^^ ^^^^^^^^ ^^  ^^^^                        '
        """

        last = 0  # pointer to the sentence
        last_match = 0  # pointer to the match_words list
        # Iteratively construct match_str with highlighted symbols, start with empty string
        match_str = ""
        while last < len(sent):
            cur_word = match_words[last_match]
            # Check if current word cur_word can be located in sent[last:last + len(cur_word)]:
            # 1. check if symbols around word position are not letters
>>>>>>> origin/claim_level
            check_boundaries = False
            if last == 0 or not sent[last - 1].isalpha():
                check_boundaries = True
            if check_boundaries and last_match < len(match_words):
<<<<<<< HEAD
                right_idx = last + len(match_words[last_match])
                if right_idx < len(sent):
                    check_boundaries = not sent[right_idx].isalpha()

            if last_match < len(match_words) and check_boundaries:
                if sent[last:].startswith(match_words[last_match]):
                    # match at sent[last] and match_words[last_match]
                    len_w = len(match_words[last_match])
                    last += len_w
                    match_str += "^" * len_w
                    last_match += 1
                    continue
            # no match at sent[last]
=======
                right_idx = last + len(cur_word)
                if right_idx < len(sent):
                    check_boundaries = not sent[right_idx].isalpha()
            if last_match < len(match_words) and check_boundaries:
                # 2. check if symbols in word position are the same as cur_word
                if sent[last:].startswith(cur_word):
                    # Found match at sent[last] with cur_word
                    len_w = len(cur_word)
                    last += len_w
                    # Highlight this position in match string
                    match_str += "^" * len_w
                    last_match += 1
                    continue
            # No match at sent[last], continue with the next position
>>>>>>> origin/claim_level
            last += 1
            match_str += " "

        if last_match < len(match_words):
<<<<<<< HEAD
            # didn't match all words to the sentence
=======
            # Didn't match all words to the sentence.
            # Possibly because the match words are in the wrong order or are not present in sentence.
>>>>>>> origin/claim_level
            return None

        return match_str

    def _align(
        self,
        sent: str,
        match_str: str,
        sent_tokens: List[int],
        tokenizer,
    ) -> List[int]:
<<<<<<< HEAD
        last = 0
        last_token = 0
        aligned_tokens = []
        while last < len(sent):
            if last_token >= len(sent_tokens):
                return aligned_tokens
            cur_token = tokenizer.decode(sent_tokens[last_token])
            if len(cur_token) > 0 and sent[last:].startswith(cur_token):
                # if the match string corresponding to the token contains matches, add to answer
=======
        """
        Get positions of the tokens which intersects with match string match_str corresponding to the sentence sent.
        """
        last = 0
        last_token = 0
        # Iteratively find position of each new token.
        aligned_tokens = []
        while last < len(sent):
            if last_token >= len(sent_tokens):
                # Already aligned all tokens.
                return aligned_tokens
            cur_token = tokenizer.decode(sent_tokens[last_token])
            # Try to find the position of cur_token in sentence, possibly in sent[last]
            if len(cur_token) > 0 and sent[last:].startswith(cur_token):
                # If the match string corresponding to the token contains matches, add to answer
>>>>>>> origin/claim_level
                if any(t == "^" for t in match_str[last : last + len(cur_token)]):
                    aligned_tokens.append(last_token)
                last_token += 1
                last += len(cur_token)
            else:
<<<<<<< HEAD
=======
                # Continue with the same token and next position in the sentence.
>>>>>>> origin/claim_level
                last += 1
        return aligned_tokens
