import re

from typing import Dict, List, Optional
from dataclasses import dataclass

from .stat_calculator import StatCalculator
from lm_polygraph.utils.openai_chat import OpenAIChat
from lm_polygraph.utils.model import WhiteboxModel
from .claim_level_prompts import CLAIM_EXTRACTION_PROMPTS, MATCHING_PROMPTS


@dataclass
class Claim:
    claim_text: str
    # The sentence of the generation, from which the claim was extracted
    sentence: str
    # Indices in the original generation of the tokens, which are related to the current claim
    aligned_token_ids: List[int]


class ClaimsExtractor(StatCalculator):
    """
    Extracts claims from the text of the model generation.
    """

    def __init__(
        self,
        openai_chat: OpenAIChat,
        sent_separators: str = ".?!。？！\n",
        language: str = "en",
    ):
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
        self.language = language
        self.openai_chat = openai_chat
        self.sent_separators = sent_separators

    def __call__(
        self,
        dependencies: Dict[str, object],
        texts: List[str],
        model: WhiteboxModel,
        *args,
        **kwargs,
    ) -> Dict[str, List]:
        """
        Extracts the claims out of each generation text.
        Parameters:
            dependencies (Dict[str, object]): input statistics, which includes:
                * 'greedy_log_probs' (List[List[float]]): log-probabilities of the generation tokens.
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
        Returns:
            Dict[str, List]: dictionary with :
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
                self.claims_from_text(greedy_text, greedy_tok, model.tokenizer)
            )
            # Iterate over newly added claims to concatenate into list
            for c in claims[-1]:
                claim_texts_concatenated.append(c.claim_text)
                claim_input_texts_concatenated.append(inp_text)

        return {
            "claims": claims,
            "claim_texts_concatenated": claim_texts_concatenated,
            "claim_input_texts_concatenated": claim_input_texts_concatenated,
        }

    def claims_from_text(self, text: str, tokens: List[int], tokenizer) -> List[Claim]:
        sentences = []
        for s in re.split(f"[{self.sent_separators}]", text):
            if len(s) > 0:
                sentences.append(s)
        if len(text) > 0 and text[-1] not in self.sent_separators:
            # Remove last unfinished sentence, because extracting claims
            # from unfinished sentence may lead to hallucinated claims.
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

            # Iteratively decode tokenized text until decoded sequence length is
            # greater or equal to the starting position of current sentence.
            # Find sentence location in tokens: tokens[sent_start_token_idx:sent_end_token_idx]
            while len(tokenizer.decode(tokens[:sent_start_token_idx])) < sent_start_idx:
                sent_start_token_idx += 1
            while len(tokenizer.decode(tokens[:sent_end_token_idx])) < sent_end_idx:
                sent_end_token_idx += 1

            # Extract claims from current sentence
            for c in self._claims_from_sentence(
                s, tokens[sent_start_token_idx:sent_end_token_idx], tokenizer
            ):
                # Correct aligned tokens positions from sentence-level to generation-level
                for i in range(len(c.aligned_token_ids)):
                    c.aligned_token_ids[i] += sent_start_token_idx
                claims.append(c)
        return claims

    def _claims_from_sentence(
        self,
        sent: str,
        sent_tokens: List[int],
        tokenizer,
    ) -> List[Claim]:
        # Extract claims with specific prompt
        extracted_claims = self.openai_chat.ask(
            CLAIM_EXTRACTION_PROMPTS[self.language].format(sent=sent)
        )
        claims = []
        for claim_text in extracted_claims.split("\n"):
            # Bad claim_text example:
            # - There aren't any claims in this sentence.
            if not claim_text.startswith("- "):
                continue
            if "there aren't any claims" in claim_text.lower():
                continue
            # remove '- ' in the beginning
            claim_text = claim_text[2:].strip()
            # Get words which matches the claim using specific prompt
            # Example:
            # sent = 'Lanny Flaherty is an American actor born on December 18, 1949, in Pensacola, Florida.'
            # claim = 'Lanny Flaherty was born on December 18, 1949.'
            # GPT response: 'Lanny, Flaherty, born, on, December, 18, 1949'
            # match_words = ['Lanny', 'Flaherty', 'born', 'on', 'December', '18', '1949']
            chat_ask = MATCHING_PROMPTS[self.language].format(
                sent=sent,
                claim=claim_text,
            )
            match_words = self.openai_chat.ask(chat_ask)
            # comma has a different form in Chinese and space works better
            if self.language == "zh":
                match_words = match_words.strip().split(" ")
            else:
                match_words = match_words.strip().split(",")
            match_words = list(map(lambda x: x.strip(), match_words))
            # Try to highlight matched symbols in sent
            if self.language == "zh":
                match_string = self._match_string_zh(sent, match_words)
            else:
                match_string = self._match_string(sent, match_words)
            if match_string is None:
                continue
            # Get token positions which intersect with highlighted regions, that is, correspond to the claim
            aligned_token_ids = self._align(sent, match_string, sent_tokens, tokenizer)
            if len(aligned_token_ids) == 0:
                continue
            claims.append(
                Claim(
                    claim_text=claim_text,
                    sentence=sent,
                    aligned_token_ids=aligned_token_ids,
                )
            )
        return claims

    def _match_string(self, sent: str, match_words: List[str]) -> Optional[str]:
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

        sent_pos = 0  # pointer to the sentence
        match_words_pos = 0  # pointer to the match_words list
        # Iteratively construct match_str with highlighted symbols, start with empty string
        match_str = ""
        while sent_pos < len(sent):
            # Check if current word cur_word can be located in sent[sent_pos:sent_pos + len(cur_word)]:
            # 1. check if symbols around word position are not letters
            check_boundaries = False
            if sent_pos == 0 or not sent[sent_pos - 1].isalpha():
                check_boundaries = True
            if check_boundaries and match_words_pos < len(match_words):
                cur_match_word = match_words[match_words_pos]
                right_idx = sent_pos + len(cur_match_word)
                if right_idx < len(sent):
                    check_boundaries = not sent[right_idx].isalpha()
                # 2. check if symbols in word position are the same as cur_word
                if check_boundaries and sent[sent_pos:].startswith(cur_match_word):
                    # Found match at sent[sent_pos] with cur_word
                    len_w = len(cur_match_word)
                    sent_pos += len_w
                    # Highlight this position in match string
                    match_str += "^" * len_w
                    match_words_pos += 1
                    continue
            # No match at sent[sent_pos], continue with the next position
            sent_pos += 1
            match_str += " "

        if match_words_pos < len(match_words):
            # Didn't match all words to the sentence.
            # Possibly because the match words are in the wrong order or are not present in sentence.
            return None

        return match_str

    def _match_string_zh(self, sent: str, match_words: List[str]) -> Optional[str]:
        # Greedily matching characters from `match_words` to `sent` for Chinese.
        # Returns None if matching failed, e.g. due to characters in match_words, which are not present
        # in sent, or if the characters are not in the same order they appear in the sentence.
        #
        # Example:
        # sent = '爱因斯坦也是一位和平主义者。'
        # match_words = ['爱因斯坦', '是', '和平', '主义者']
        # return '^^^^ ^  ^^^^'

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

    def _align(
        self,
        sent: str,
        match_str: str,
        sent_tokens: List[int],
        tokenizer,
    ) -> List[int]:
        """
        Identifies token indices in `sent_tokens` that align with matching characters (marked by '^')
        in `match_str`. All tokens, which textual representations intersect with any of matching
        characters, are included. Partial intersections should be uncommon in practice.

        Args:
            sent: the original sentence.
            match_str: a string of the same length as `sent` where '^' characters indicate matches.
            sent_tokens: a list of token ids representing the tokenized version of `sent`.
            tokenizer: the tokenizer used to decode tokens.

        Returns:
            A list of integers representing the indices of tokens in `sent_tokens` that align with
            matching characters in `match_str`.
        """
        sent_pos = 0
        cur_token_i = 0
        # Iteratively find position of each new token.
        aligned_token_ids = []
        while sent_pos < len(sent) and cur_token_i < len(sent_tokens):
            cur_token_text = tokenizer.decode(sent_tokens[cur_token_i])
            # Try to find the position of cur_token_text in sentence, possibly in sent[sent_pos]
            if len(cur_token_text) == 0:
                # Skip non-informative token
                cur_token_i += 1
                continue
            if sent[sent_pos:].startswith(cur_token_text):
                # If the match string corresponding to the token contains matches, add to answer
                if any(
                    t == "^"
                    for t in match_str[sent_pos : sent_pos + len(cur_token_text)]
                ):
                    aligned_token_ids.append(cur_token_i)
                cur_token_i += 1
                sent_pos += len(cur_token_text)
            else:
                # Continue with the same token and next position in the sentence.
                sent_pos += 1
        return aligned_token_ids
