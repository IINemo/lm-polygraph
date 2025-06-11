import torch
import numpy as np

from typing import Dict, List, Optional, Tuple

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel


class BasePromptCalculator(StatCalculator):
    """
    Calculates the probability for a specific token to be generated from the specific prompt.
    Used for P(True)-based methods.
    """

    def __init__(
        self,
        prompt: str,
        expected: str,
        method: str,
        input_text_dependency: str = "input_texts",
        sample_text_dependency: Optional[str] = None,
        generation_text_dependency: str = "greedy_texts",
    ):
        """
        Parameters:
            prompt (str): Prompt to use for estimating the answer of.
                The following values can be used in the prompt:
                - q: input text
                - a: generation text
                - s: list of several generation samples.
                Prompt example: 'Question: {q}. Is the following answer true? {a}'.
            expected (str): string to measure probability of. Must be decoded into one token,
                otherwise an exception will be raised.
            method (str): the name of the statistics to calculate with this calculator.
        """
        super().__init__()
        self.method = method
        self.prompt = prompt
        self.expected = expected
        self.input_text_dependency = input_text_dependency
        self.sample_text_dependency = sample_text_dependency
        self.generation_text_dependency = generation_text_dependency

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the probability for `expected` to be generated from `prompt`.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, consisting of:
                - 'greedy_texts' (List[str]): model generations for this batch,
                - 'sample_texts' (List[List[str]]): several sampling generations for each input text.
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with the following items:
                - `method` (List[float]): logarithms of probability of generating `expected` from prompt formatted
                    at each input text.
        """
        expected_tokens = model.tokenizer([self.expected])["input_ids"][0]
        expected_tokens = [
            t
            for t in expected_tokens
            if t != model.tokenizer.eos_token_id and t != model.tokenizer.bos_token_id
        ]
        assert len(expected_tokens) == 1
        expected_token = expected_tokens[0]

        answers = dependencies[self.generation_text_dependency]
        samples = [[] for _ in range(len(answers))]
        if self.sample_text_dependency is not None:
            samples = dependencies[self.sample_text_dependency]
        input_texts = dependencies[self.input_text_dependency]
        inp_texts = [
            self.prompt.format(q=text, s=", ".join(sample), a=ans)
            for text, ans, sample in zip(input_texts, answers, samples)
        ]

        if len(inp_texts) == 0:
            return {self.method: np.array([])}

        batch: Dict[str, torch.Tensor] = model.tokenize(inp_texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}

        with torch.no_grad():
            out = model.generate(
                **batch,
                output_scores=True,
                return_dict_in_generate=True,
                min_new_tokens=1,
                max_new_tokens=1,
                num_beams=1,
            )

        logits = torch.stack(out.scores, dim=1)
        log_probs = logits[:, -1, expected_token].cpu().numpy()

        return {self.method: log_probs}


class PromptCalculator(BasePromptCalculator):

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the PromptCalculator.
        """

        return ["p_true"], ["greedy_texts"]

    def __init__(self):
        super().__init__(
            "Question: {q}\n Possible answer:{a}\n "
            "Is the possible answer:\n (A) True\n (B) False\n The possible answer is:",
            "True",
            "p_true",
        )


class SamplingPromptCalculator(BasePromptCalculator):

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the SamplingPromptCalculator.
        """

        return ["p_true_sampling"], ["greedy_texts", "sample_texts"]

    def __init__(self):
        super().__init__(
            "Question: {q}\n Here are some ideas that were brainstormed: {s}\n Possible answer:{a}\n "
            "Is the possible answer:\n (A) True\n (B) False\n The possible answer is:",
            "True",
            "p_true_sampling",
            sample_text_dependency="sample_texts",
        )


class ClaimPromptCalculator(BasePromptCalculator):

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the ClaimPromptCalculator.
        """

        return ["p_true_claim"], ["greedy_texts"]

    def __init__(self):
        super().__init__(
            "Question: {q}\n Possible answer:{a}\n "
            "Is the possible answer True or False? The possible answer is: ",
            "True",
            "p_true_claim",
            input_text_dependency="claim_input_texts_concatenated",
            sample_text_dependency=None,
            generation_text_dependency="claim_texts_concatenated",
        )


class BaseAttentionPromptCalculator(StatCalculator):
    """
    Calculates the attention weights for a specific token to be generated from the specific prompt.
    Used for CSL method.
    """

    def __init__(
        self,
        prompt: str,
        suffix_prompt: str,
        method: str,
        input_text_dependency: str = "input_texts",
        generation_text_dependency: str = "greedy_texts",
        output_attentions: bool = False,
    ):
        """
        Parameters:
            prompt (str): Prompt to use for estimating the answer of.
                The following values can be used in the prompt:
                - q: input text
                - a: generation text
                Prompt example: 'Question: {q}. Is the following answer true? {a}'.
            method (str): the name of the statistics to calculate with this calculator.
        """
        super().__init__()
        self.method = method
        self.prompt = prompt
        self.suffix_prompt = suffix_prompt
        self.input_text_dependency = input_text_dependency
        self.generation_text_dependency = generation_text_dependency
        self.output_attentions = output_attentions

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the probability for `expected` to be generated from `prompt`.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, consisting of:
                - 'greedy_texts' (List[str]): model generations for this batch,
                - 'sample_texts' (List[List[str]]): several sampling generations for each input text.
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with the following items:
                - `method` (List[float]): logarithms of probability of generating `expected` from prompt formatted
                    at each input text.
        """

        greedy_tokens = dependencies["greedy_tokens"]

        inp_texts = [self.prompt.format(q=text) for text in texts]

        prompt_tokens = []
        for i, text in enumerate(inp_texts):
            input_ids = model.tokenizer([text])["input_ids"][0]
            suffix_ids = model.tokenizer(
                [self.suffix_prompt], add_special_tokens=False
            )["input_ids"][0]
            combined_ids = input_ids + greedy_tokens[i] + suffix_ids
            prompt_tokens.append(combined_ids)
        prompt_tokens = torch.tensor(prompt_tokens).to(model.device())

        if len(inp_texts) == 0:
            return {self.method: np.array([])}

        with torch.no_grad():
            out = model(
                input_ids=prompt_tokens,
                output_attentions=self.output_attentions,
            )

        attentions = out.attentions
        attn_mask = []
        for i in range(len(texts)):
            greedy_tokens_i = dependencies["greedy_tokens"][i]
            n_tokens = len(greedy_tokens_i)
            input_ids = prompt_tokens[i].cpu().detach().numpy()
            len_input_ids = len(input_ids)

            start, end = None, None
            for pos in range(len_input_ids - n_tokens, -1, -1):
                if np.array_equal(input_ids[pos : pos + n_tokens], greedy_tokens_i):
                    start, end = pos, pos + n_tokens
                    break

            # If greedy_tokens not found, use default positions
            if start is None or end is None:
                start, end = -4 - n_tokens, -4

            stacked_attention = tuple(attention.to("cpu") for attention in attentions)
            stacked_attention = torch.cat(stacked_attention).float().numpy()
            if stacked_attention.dtype == torch.bfloat16:
                stacked_attention = stacked_attention.to(
                    torch.float16
                )  # numpy does not support bfloat16

            attn_mask.append(
                stacked_attention[:, :, len_input_ids - 1, start:end].reshape(
                    -1, n_tokens
                )
            )

        return {f"attention_weights_{self.method}": attn_mask}


class AttentionElicitingPromptCalculator(BaseAttentionPromptCalculator):

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the PromptCalculator.
        """

        return ["attention_weights_eliciting_prompt"], ["greedy_texts"]

    def __init__(self):
        super().__init__(
            """Read the following question with optional context and decide if the answer correctly answer the question. Focus on the answer, and reply Y or N.
            Context: Luxor International Airport is a airport near Luxor in Egypt (EG). It is 353 km away from the nearest seaport (Duba). The offical IATA for this airport is LXR.
            Question: Luxor international airport is in which country?
            Answer: It is in the United States, and its IATA is LXR.
            Decision: N. (The airport is in Egypt, not the United States.)

            Context: Harry is a good witcher.
            Question: How old is Harry?
            Answer: Harry practices witchcraft.
            Decision: N. (The answer does not mention Harry\'s age.)

            Question: What is the capital of Kenya?
            Answer: Nairobi is the capital of Kenya.
            Decision: Y.

            Question: Who has won the most Premier League titles since 2015?
            Answer: Manchester City have win the most Premier League title after 2015.
            Decision: Y. (Grammar errors are ignored.)

            Question: {q}
            Answer: """,
            "\n\nDecision:",
            "eliciting_prompt",
            output_attentions=True,
        )
