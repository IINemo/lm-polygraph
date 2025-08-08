from tqdm import tqdm
from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from lm_polygraph.stat_calculators.extract_claims import Claim, WhiteboxModel


class StepsExtractor(StatCalculator):
    def __init__(
        self,
        sent_separators: str = "\n",
        skip_starts: list[str] = ["Reasoning Steps:"],
        progress_bar: bool = True,
    ):
        super().__init__()
        self.sent_separators = sent_separators
        self.skip_starts = skip_starts
        self.progress_bar = progress_bar

    @staticmethod
    def meta_info() -> tuple[list[str], list[str]]:
        return (
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

    def __call__(
        self,
        dependencies: dict[str, object],
        texts: list[str],
        model: WhiteboxModel,
        *args,
        **kwargs,
    ) -> dict[str, list]:
        claims: list[list[Claim]] = []
        claim_texts_concatenated: list[str] = []
        claim_input_texts_concatenated: list[str] = []

        data = zip(
            texts,
            dependencies["greedy_texts"],
            dependencies["greedy_tokens"],
        )
        if self.progress_bar:
            data = tqdm(data, total=len(texts), desc="Extracting steps")
        for input_text, greedy_text, greedy_tokens in data:
            steps: list[Claim] = self.split_to_steps(
                greedy_text, greedy_tokens, model.tokenizer
            )
            claims.append(steps)
            claim_texts_concatenated += [c.claim_text for c in steps]
            claim_input_texts_concatenated += [input_text for c in steps]

        return {
            "claims": claims,
            "claim_texts_concatenated": claim_texts_concatenated,
            "claim_input_texts_concatenated": claim_input_texts_concatenated,
        }

    def filter_claim_texts(self, claim_text: str) -> bool:
        claim_text = claim_text.strip()
        return len(claim_text) > 0 and not any(
            claim_text.lower().startswith(b.lower()) for b in self.skip_starts
        )

    def split_to_steps(
        self,
        text: str,
        tokens: list[int],
        tokenizer,
    ) -> list[Claim]:
        if not tokenizer.decode(tokens).startswith(text):
            return []
        prev_token_i, token_i = 0, 0
        prev_text_i = 0
        claims: list[Claim] = []
        for text_i in range(len(text)):
            if text[text_i] in self.sent_separators and self.filter_claim_texts(
                text[prev_text_i : text_i + 1]
            ):
                end = token_i if token_i > prev_token_i else prev_token_i + 1
                claims.append(
                    Claim(
                        claim_text=text[prev_text_i : text_i + 1].strip(),
                        sentence=text[prev_text_i : text_i + 1],
                        aligned_token_ids=list(range(prev_token_i, min(end, len(tokens)))),
                    )
                )
            while (
                token_i < len(tokens)
                and tokenizer.decode(tokens[: token_i + 1]) in text[: text_i + 1]
            ):
                token_i += 1
            if text[text_i] in self.sent_separators:
                prev_text_i = text_i + 1
                prev_token_i = token_i
        if self.filter_claim_texts(text[prev_text_i:]):
            end = token_i if token_i > prev_token_i else prev_token_i + 1
            claims.append(
                Claim(
                    claim_text=text[prev_text_i:].strip(),
                    sentence=text[prev_text_i:],
                    aligned_token_ids=list(range(prev_token_i, min(end, len(tokens)))),
                )
            )
        return claims


def load_stat_calculator(config, builder):
    return StepsExtractor(
        sent_separators=getattr(config, "sent_separators", "\n"),
        progress_bar=getattr(config, "progress_bar", False),
    )
