import torch

from typing import List, Dict
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


@dataclass
class Model:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer

    def device(self):
        return self.model.device

    @staticmethod
    def from_pretrained(model_path: str, device: str = 'cpu'):
        model = AutoModelForCausalLM.from_pretrained(model_path, max_length=256).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", add_bos_token=True)
        model.eval()
        return Model(model, tokenizer)

    @staticmethod
    def load(model_path: str, tokenizer_path: str, device: str = 'cpu'):
        model = torch.load(model_path).to(device)
        tokenizer = torch.load(tokenizer_path, padding_side="left")
        model.eval()
        return Model(model, tokenizer)

    def tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        tokenized = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
        return tokenized
