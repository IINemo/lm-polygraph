from typing import Dict, List, Optional

import numpy as np
import torch

from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel


class EnsembleProbsCalculator(StatCalculator):
    """
    Compute per-class probabilities for each prompt instruction using the underlying
    white-box model. This replaces the previous placeholder implementation that
    returned random numbers.
    """

    def __init__(
        self,
        instructions: List[str],
        class_labels: List[str],
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        prompt_formatting: Optional[str] = None,
    ):
        self.instructions = instructions
        self.class_labels = class_labels
        self.few_shot_examples = few_shot_examples
        self.prompt_formatting = prompt_formatting

    def _format_prompt(self, instruction: str, input_text: str) -> str:
        """Build a prompt that lists options and optional few-shot examples."""
        options = ", ".join(self.class_labels)
        examples_block = ""
        if self.few_shot_examples:
            formatted_examples = [
                f"Input: {ex['text']}\nLabel: {ex['label']}"
                for ex in self.few_shot_examples
            ]
            examples_block = "\n\n".join(formatted_examples)

        if self.prompt_formatting:
            return self.prompt_formatting.format(
                instruction=instruction,
                input_text=input_text,
                options=options,
                examples=examples_block,
            )

        prompt_parts = [instruction.strip(), f"Options: {options}"]
        if examples_block:
            prompt_parts.append("Examples:")
            prompt_parts.append(examples_block)
        prompt_parts.append(f"Input: {input_text}\nLabel:")
        return "\n".join(prompt_parts)

    def _label_logprob(
        self,
        prompt_ids: List[int],
        label_ids: List[int],
        model: WhiteboxModel,
    ) -> float:
        """
        Compute the log-probability of `label_ids` generated after `prompt_ids`
        using teacher-forcing cross-entropy (sum over label tokens).
        """
        if len(label_ids) == 0:
            return float("-inf")

        device = model.device()
        input_ids = torch.tensor([prompt_ids + label_ids], device=device)
        attention_mask = torch.ones_like(input_ids)

        labels = torch.full_like(input_ids, -100)
        labels[0, len(prompt_ids) :] = torch.tensor(label_ids, device=device)

        with torch.no_grad():
            outputs = model.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

        # outputs.loss is mean per label token; convert back to sum log-probability
        log_prob = -outputs.loss * len(label_ids)
        return float(log_prob.detach().cpu())

    def __call__(
        self,
        batch_stats: Dict,
        inp_texts: List[str],
        model: WhiteboxModel,
        max_new_tokens=None,
        *args,
        **kwargs,
    ):
        n_samples = len(inp_texts)
        n_classes = len(self.class_labels)
        n_instructions = len(self.instructions)

        ensemble_probs = np.zeros(
            (n_samples, n_classes, n_instructions), dtype=np.float32
        )

        tokenizer = model.tokenizer

        for i, text in enumerate(inp_texts):
            for j, instruction in enumerate(self.instructions):
                prompt = self._format_prompt(instruction, text)
                prompt_ids = (
                    tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
                    .input_ids[0]
                    .tolist()
                )

                label_logprobs = []
                for label in self.class_labels:
                    label_ids = tokenizer.encode(label, add_special_tokens=False)
                    label_logprobs.append(
                        self._label_logprob(prompt_ids, label_ids, model)
                    )

                label_probs = torch.softmax(
                    torch.tensor(label_logprobs, dtype=torch.float32), dim=-1
                )
                ensemble_probs[i, :, j] = label_probs.cpu().numpy()

        return {"ensemble_probs": ensemble_probs}

    @staticmethod
    def meta_info():
        return (["ensemble_probs"], ["input_texts"])
