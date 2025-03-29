import openai
from .generation_metric import GenerationMetric
import numpy as np
import logging
from typing import Dict, List
import re 
log = logging.getLogger("lm_polygraph")
import os 

class GptAccuracyMetric(GenerationMetric):
    """
    Uses GPT to compare generated text with target and return 1 if semantically equivalent, else 0.
    """

    def __init__(self, model="gpt-4o-mini", sample=False, sample_strategy="First", api_key=None):
        if sample:
            super().__init__([
                "first_sample_texts",
                "best_sample_texts",
                "best_normalized_sample_texts",
                "input_texts"],
                "sequence")
        else:
            super().__init__(["greedy_texts", "input_texts"], "sequence")
        self.sample = sample
        self.sample_strategy = sample_strategy
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key

    def __str__(self):
        if self.sample == True:
            return f"GptAccuracy_{self.model}_{self.sample_strategy}"
        return f"GptAccuracy_{self.model}"
    
    def _filter_input(self, input):
        matches = re.findall(r"Question:\s*(.*?)\nAnswer:", input, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return input
    def _gpt_compare(self, output: str, target: str, question: str) -> int:
        prompt = (
            f"You are a text evaluator. The model was asked the following question: {question.strip()}.\n"
            "The 'Generated' text is a model's response. The 'Target' is the correct answer.\n"
            "If the generated answer correctly answers the question based on the target, return 1.\n"
            "If it is wrong, return 0.\n"
            "Respond ONLY with a single digit: 1 or 0.\n\n"
            f"Generated: {output.strip()}\n"
            f"Target: {target.strip()}"
        )

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a strict evaluator of text similarity."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1,
                n=1
            )

            raw_reply = response['choices'][0]['message']['content'].strip()
            return int(raw_reply) if raw_reply in ['0', '1'] else 0

        except Exception as e:
            log.error(f"GPT comparison failed: {e}")
            return 0  # Safe default

    def __call__(self, stats: Dict[str, np.ndarray], target_texts: List[str]) -> np.ndarray:
        if self.sample:
            if self.sample_strategy == "First":
                gen_texts = stats["first_sample_texts"]
            elif self.sample_strategy == "Best":
                gen_texts = stats["best_sample_texts"]
            elif self.sample_strategy == "BestNormalized":
                gen_texts = stats["best_normalized_sample_texts"]
            else:
                raise ValueError(f"Invalid sample strategy: {self.sample_strategy}")
        else:
            gen_texts = stats["greedy_texts"]

        results = []
        input_texts = [self._filter_input(text) for text in stats["input_texts"]]
        for output, target, input in zip(gen_texts, target_texts, input_texts):
            score = self._gpt_compare(output, target,input)
            results.append(score)

        return np.array(results)
