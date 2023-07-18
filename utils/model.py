import torch

from typing import List, Dict
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig

@dataclass
class Model:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    model_path: str
    model_type: str

    def device(self):
        return self.model.device

    @staticmethod
    def from_pretrained(model_path: str, device: str = 'cpu'):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if any(["CausalLM" in architecture for architecture in config.architectures]):
            model_type = "CausalLM"
            model = AutoModelForCausalLM.from_pretrained(model_path, max_length=256, trust_remote_code=True).to(device)
        elif any([("Seq2SeqLM" in architecture) or ("ConditionalGeneration" in architecture)
                  for architecture in config.architectures]):
            model_type = "Seq2SeqLM"
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, max_length=256).to(device)
        else:
            raise ValueError(f'Model {model_path} is not adopted for the sequence generation task')
            
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", add_bos_token=True,
                                                  model_max_length=256)
        
        model.eval()
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token 
            
        return Model(model, tokenizer, model_path, model_type)

    @staticmethod
    def load(model_path: str, tokenizer_path: str, device: str = 'cpu'):
        model = torch.load(model_path).to(device)
        tokenizer = torch.load(tokenizer_path, padding_side="left")
        model.eval()
        return Model(model, tokenizer, model_path)

    def tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        if ("falcon" in self.model.config._name_or_path) or ("llama" in self.model.config._name_or_path):
            tokenized = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt', return_token_type_ids=False)
        else:
            tokenized = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
        return tokenized
