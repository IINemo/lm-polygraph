import re
import numpy as np

from typing import List, Dict
from .generation_metric import GenerationMetric
from transformers import AutoTokenizer
from .x_metric_utils import MT5ForRegression
import torch 
import datasets 
from transformers import TrainingArguments, DataCollatorWithPadding, Trainer

class XMetric(GenerationMetric):
    """
    Calculates X-MERTIC (https://aclanthology.org/2023.wmt-1.63/)
    between model-generated texts and ground truth texts.
    """

    def __init__(self, model ,tokenizer,
                 source_ignore_regex=None, translation_ignore_regex=None, sample: bool = False, sample_strategy: str = "First"):
        if sample:
            super().__init__([
                "first_sample_texts",
                "best_sample_texts",
                "best_normalized_sample_texts",
                "input_texts"],
            "sequence")
        else:
            super().__init__(["greedy_texts", "input_texts"], "sequence")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model

        self.tokenizer = tokenizer
        self.source_ignore_regex = (
            re.compile(source_ignore_regex) if source_ignore_regex else None
        )
        self.translation_ignore_regex = (
            re.compile(translation_ignore_regex) if translation_ignore_regex else None
        )

        self.training_args = TrainingArguments(
            output_dir=".",
            per_device_eval_batch_size=1,
            dataloader_pin_memory=False,
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            data_collator=data_collator
        )
        self.sample = sample
        self.sample_strategy=sample_strategy


    def __str__(self):
        if self.sample:
            if self.sample_strategy == "First":
                return f"Samplexmetric"
            else:
                return f"{self.sample_strategy}Samplexmetric"
        return "xmetric"

    def _filter_source(self, text: str, ignore_regex: re.Pattern) -> str:
        if ignore_regex is not None:
            try:
                return ignore_regex.findall(text)[-1]
            except IndexError:
                raise ValueError(
                    f"Source text '{text}' does not match the ignore regex '{ignore_regex}'"
                )
        return text

    def _filter_translation(self, text: str, ignore_regex: re.Pattern) -> str:
        return ignore_regex.sub("", text).strip() if ignore_regex else text.strip()

    def _filter_text(self, text: str, ignore_regex: re.Pattern) -> str:
        if ignore_regex is not None:
            processed_text = ignore_regex.search(text)
            if processed_text:
                return processed_text.group(1)
            else:
                raise ValueError(
                    f"Source text {text} does not match the ignore regex {ignore_regex}"
                )
        return text

    def _prepare_inputs(self, translations: List[str], references: List[str], sources: List[str],):
        """Prepares the input data for X-MERTIC scoring."""
        inputs = [
            f"source: {source} candidate: {hyp} reference: {ref}" 
            for hyp, ref, source in zip(translations, references, sources)
        ]
        tokenized = self.tokenizer(
            inputs, 
            max_length=512, 
            truncation=True, 
            padding=False
        )
        
        # Convert to Hugging Face Dataset
        dataset = datasets.Dataset.from_dict({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "input":inputs
        }).with_format("torch")  
        
        def remove_eos(example):
            example["input_ids"] = example["input_ids"][:-1]
            example["attention_mask"] = example["attention_mask"][:-1]
            return example

        dataset = dataset.map(remove_eos)
        return dataset

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
    ) -> np.ndarray:
        """
        Calculates X-MERTIC between stats['greedy_texts'] and target_texts.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, including:
                * model-generated texts in 'greedy_texts'
            target_texts (List[str]): ground-truth texts
            input_texts (List[str]): input texts before translation

        Returns:
            np.ndarray: list of X-MERTIC scores for each sample.
        """
        references = [
            src
            for src in stats["target_texts"]
        ]
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

        translations = [
            self._filter_translation(tr, self.source_ignore_regex)
            for tr in gen_texts
        ]

        sources = [
            self._filter_text(src, self.source_ignore_regex)
            for src in stats["input_texts"]
        ]

        inputs = self._prepare_inputs(translations, references, sources)
        scores, _, _ = self.trainer.predict(test_dataset=inputs)
        for i, score in enumerate(scores):
            scores[i] = (25 - score) / 25
        return scores