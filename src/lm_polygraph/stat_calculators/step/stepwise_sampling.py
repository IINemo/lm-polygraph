import torch
from typing import List, Dict
from transformers import StoppingCriteria, StoppingCriteriaList

from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from .steps_extractor import StepsExtractor


class StopOnNewline(StoppingCriteria):
    def __init__(self, tokenizer, start_length):
        self.tokenizer = tokenizer
        self.start_length = start_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        generated_ids = input_ids[0][self.start_length :]
        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return any(x in decoded for x in ["\n- Step ", "\n<Answer>: "])


class StepwiseSamplingCalculator(StatCalculator):
    def __init__(
        self,
        candidates_per_step: int = 10,
        max_tokens_per_step: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        verbose: bool = False,
    ):
        super().__init__()
        self.steps_extractor = StepsExtractor()
        self.candidates_per_step = candidates_per_step
        self.max_tokens_per_step = max_tokens_per_step
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.verbose = verbose

    @staticmethod
    def meta_info() -> tuple[list[str], list[str]]:
        return (
            [
                "sample_steps_texts",
                "sample_steps_tokens",
                "sample_steps_log_likelihoods",
                "sample_steps_log_probs",
            ],
            [
                "input_tokens",
                "greedy_texts",
                "greedy_tokens",
                "claims",
            ],
        )

    def prepare_claims(self, claims, input_len, full_len):
        all_claim_tensors = []
        for claim in claims:
            mask = torch.zeros((1, full_len), dtype=int)
            mask[0, (input_len + torch.as_tensor(claim.aligned_token_ids)).int()] = 1
            all_claim_tensors.append(mask[:, 1:])  # ignoring <s>
        return all_claim_tensors

    def generate_step_candidates(self, model: WhiteboxModel, prompt_tokens: list[int]):
        llm_inputs = {
            "input_ids": torch.LongTensor([prompt_tokens]).to(model.device()),
            "attention_mask": torch.ones(1, len(prompt_tokens))
            .bool()
            .to(model.device()),
        }
        start_len = llm_inputs["input_ids"].shape[-1]
        stopping_criteria = StoppingCriteriaList(
            [StopOnNewline(model.tokenizer, start_len)]
        )

        llm_outputs = model.generate(
            **llm_inputs,
            max_new_tokens=self.max_tokens_per_step,
            do_sample=True,
            top_p=self.top_p,
            top_k=self.top_k,
            temperature=self.temperature,
            num_return_sequences=self.candidates_per_step,
            eos_token_id=model.tokenizer.eos_token_id,
            pad_token_id=model.tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            output_scores=True,
            return_dict_in_generate=True,
            output_attentions=False,
        )

        return llm_inputs, llm_outputs

    def extract_last_position(self, model: WhiteboxModel, new_tokens: torch.Tensor):
        new_text = model.tokenizer.decode(new_tokens, skip_special_tokens=True)
        steps = self.steps_extractor.split_to_steps(
            new_text, new_tokens, model.tokenizer
        )
        if len(steps) == 0:
            return 0
        return max(steps[0].aligned_token_ids) + 1

    def __call__(
        self,
        dependencies: Dict[str, object],
        texts: List[str],
        model: WhiteboxModel,
        **kwargs,
    ) -> Dict[str, List]:
        all_claims = dependencies["claims"]
        results = {
            "sample_steps_texts": [[[] for _ in claims] for claims in all_claims],
            "sample_steps_tokens": [[[] for _ in claims] for claims in all_claims],
            "sample_steps_log_likelihoods": [
                [[] for _ in claims] for claims in all_claims
            ],
            "sample_steps_log_probs": [[[] for _ in claims] for claims in all_claims],
        }
        for i in range(len(texts)):
            input_tokens = dependencies["input_tokens"][i]
            greedy_tokens = dependencies["greedy_tokens"][i]
            claims = dependencies["claims"][i]

            last_claim_pos = 0

            for claim_pos in range(len(claims)):
                if len(claims[claim_pos].aligned_token_ids) > 0:
                    first_claim_pos = min(claims[claim_pos].aligned_token_ids)
                else:
                    first_claim_pos = last_claim_pos
                last_claim_pos = first_claim_pos
                cur_tokens = input_tokens + greedy_tokens[:first_claim_pos]
                if self.verbose:
                    print(
                        'Generating from: "{}"'.format(
                            model.tokenizer.decode(cur_tokens).split("</think>\n\n")[-1]
                        )
                    )
                inputs, outputs = self.generate_step_candidates(model, cur_tokens)
                scores = torch.stack(outputs.scores, dim=1)

                sample_steps_texts: list[str] = []
                sample_steps_tokens: list[list[int]] = []
                sample_steps_log_likelihoods: list[list[float]] = []
                sample_steps_log_probs: list[float] = []
                for o, logprobs in zip(
                    outputs.sequences[:, inputs["input_ids"].shape[-1] :], scores
                ):
                    last_pos = self.extract_last_position(model, o)
                    sample_step_tokens = o[:last_pos].tolist()
                    sample_step_text = model.tokenizer.decode(
                        sample_step_tokens, skip_special_tokens=True
                    )
                    sample_steps_texts.append(sample_step_text)
                    sample_steps_tokens.append(sample_step_tokens)
                    sample_steps_log_likelihoods.append(
                        [
                            logprobs[i, sample_step_tokens[i]].item()
                            for i in range(last_pos)
                        ]
                    )
                    sample_steps_log_probs.append(sum(sample_steps_log_likelihoods[-1]))

                if self.verbose:
                    print("Sample Steps:")
                    for sample_step in sample_steps_texts:
                        print(f'"{sample_step}"')

                results["sample_steps_texts"][i][claim_pos] = sample_steps_texts
                results["sample_steps_tokens"][i][claim_pos] = sample_steps_tokens
                results["sample_steps_log_likelihoods"][i][
                    claim_pos
                ] = sample_steps_log_likelihoods
                results["sample_steps_log_probs"][i][claim_pos] = sample_steps_log_probs

        return results
