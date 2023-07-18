import torch

from typing import List, Dict
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration

from utils.generation_parameters import GenerationParameters


@dataclass
class Model:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    model_path: str
    parameters: GenerationParameters = GenerationParameters()

    def device(self):
        return self.model.device

    @staticmethod
    def from_pretrained(model_path: str, device: str = 'cpu'):
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, max_length=256).to(device)
        except ValueError:
            model = T5ForConditionalGeneration.from_pretrained(model_path, max_length=256).to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", add_bos_token=True,
                                                  model_max_length=256)
        model.eval()
        return Model(model, tokenizer, model_path)

    @staticmethod
    def load(model_path: str, tokenizer_path: str, device: str = 'cpu'):
        model = torch.load(model_path).to(device)
        tokenizer = torch.load(tokenizer_path, padding_side="left")
        model.eval()
        return Model(model, tokenizer, model_path)

    def tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        tokenized = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
        return tokenized
