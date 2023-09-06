import json
import requests
import torch
import sys
import openai

from typing import List, Dict
from abc import abstractmethod, ABC
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig, BartForCausalLM

from lm_polygraph.utils.generation_parameters import GenerationParameters


class Model(ABC):
    def __init__(self, model_path: str, model_type: str):
        self.model_path = model_path
        self.model_type = model_type

    @abstractmethod
    def generate_texts(self, input_texts: List[str], **args) -> List[str]:
        raise Exception("Not implemented")

    @abstractmethod
    def generate(self, **args):
        raise Exception("Not implemented")

    @abstractmethod
    def __call__(self, **args):
        raise Exception("Not implemented")


class BlackboxModel(Model):
    def __init__(self, openai_api_key: str = None, model_path: str = None, 
                 hf_api_token: str = None, hf_model_id: str = None,
                 parameters: GenerationParameters = GenerationParameters()):
        super().__init__(model_path, 'Blackbox')
        self.parameters = parameters
        openai.api_key = openai_api_key
        self.hf_api_token = hf_api_token
        self.hf_model_id = hf_model_id

    def query(self, payload):
        API_URL = f"https://api-inference.huggingface.co/models/{self.hf_model_id}"
        headers = {"Authorization": f"Bearer {self.hf_api_token}"}
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json() 

    @staticmethod
    def from_huggingface(hf_api_token: str, hf_model_id: str, **kwargs):
        return BlackboxModel(hf_api_token=hf_api_token, hf_model_id=hf_model_id, model_path = hf_model_id)
    
    @staticmethod
    def from_openai(openai_api_key: str, model_path: str, **kwargs):
        return BlackboxModel(openai_api_key = openai_api_key, model_path = model_path)
    

    def generate_texts(self, input_texts: List[str], **args) -> List[str]:
        if any(args.get(arg, False) for arg in ['output_scores', 'output_attentions', 'output_hidden_states']):
            raise Exception("Cannot access logits for blackbox model")

        for delete_key in ['do_sample', 'min_length', 'top_k', 'repetition_penalty']:
            args.pop(delete_key, None)
        for key, replace_key in [
            ('num_return_sequences', 'n'),
            ('max_length', 'max_tokens'),
            ('max_new_tokens', 'max_tokens'),
        ]:
            if key in args.keys():
                args[replace_key] = args[key]
                args.pop(key)
        texts = []

        openai = False
        hf = False
        try:
            openai_api_key = self.openai_api_key
            model_path = self.model_path
            openai = True
        except:
            hf_api_token = self.hf_api_token
            hf_model_id = self.hf_model_id
            hf = True
        
        if openai:
            for prompt in input_texts:
                response = openai.ChatCompletion.create(
                    model=self.model_path, messages=[{"role": "user", "content": prompt}], **args)
                if args['n'] == 1:
                    texts.append(response.choices[0].message.content)
                else:
                    texts.append([resp.message.content for resp in response.choices])
        elif hf:
            for prompt in input_texts:
                output = self.query({"inputs": prompt})
                texts.append(output[0]['generated_text'])
        return texts

    def generate(self, **args):
        raise Exception("Cannot access logits of blackbox model")

    def __call__(self, **args):
        raise Exception("Cannot access logits of blackbox model")

    def tokenizer(self, *args, **kwargs):
        raise Exception("Cannot access logits of blackbox model")


def _valdate_args(args):
    if 'presence_penalty' in args.keys() and args['presence_penalty'] != 0.0:
        sys.stderr.write('Skipping requested argument presence_penalty={}'.format(args['presence_penalty']))
    args.pop('presence_penalty', None)
    return args


class WhiteboxModel(Model):
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, model_path: str, model_type: str,
                 parameters: GenerationParameters = GenerationParameters()):
        super().__init__(model_path, model_type)
        self.model = model
        self.tokenizer = tokenizer
        self.parameters = parameters

    def generate(self, **args):
        args = _valdate_args(args)
        print('WhiteboxModel.generate args:', args)
        return self.model.generate(**args)

    def generate_texts(self, input_texts: List[str], **args) -> List[str]:
        args = _valdate_args(args)
        args['return_dict_in_generate'] = True
        print('WhiteboxModel.generate_texts args:', args)
        batch: Dict[str, torch.Tensor] = self.tokenize(input_texts)
        batch = {k: v.to(self.device()) for k, v in batch.items()}
        texts = [self.tokenizer.decode(x) for x in self.model.generate(**batch, **args).sequences.cpu()]
        return texts

    def __call__(self, **args):
        return self.model(**args)

    def device(self):
        return self.model.device

    @staticmethod
    def from_pretrained(model_path: str, device: str = 'cpu', **kwargs):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if any(["CausalLM" in architecture for architecture in config.architectures]):
            model_type = "CausalLM"
            model = AutoModelForCausalLM.from_pretrained(
                model_path, max_length=256, trust_remote_code=True, **kwargs)
        elif any([("Seq2SeqLM" in architecture) or ("ConditionalGeneration" in architecture)
                  for architecture in config.architectures]):
            model_type = "Seq2SeqLM"
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, max_length=256, **kwargs)
            if 'falcon' in model_path:
                model.transformer.alibi = True
        elif any(["BartModel" in architecture for architecture in config.architectures]):
            model_type = "CausalLM"
            model = BartForCausalLM.from_pretrained(model_path, max_length=256, **kwargs)
        else:
            raise ValueError(f'Model {model_path} is not adapted for the sequence generation task')
        if not kwargs.get('load_in_8bit', False) and not kwargs.get('load_in_4bit', False):
            model = model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, padding_side="left", add_bos_token=True, model_max_length=256)

        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return WhiteboxModel(model, tokenizer, model_path, model_type)

    @staticmethod
    def load(model_path: str, tokenizer_path: str, device: str = 'cpu'):
        model = torch.load(model_path).to(device)
        tokenizer = torch.load(tokenizer_path, padding_side="left")
        model.eval()
        return WhiteboxModel(model, tokenizer, model_path)

    def tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        if ("falcon" in self.model.config._name_or_path) or ("Llama" in self.model.config._name_or_path):
            tokenized = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt',
                                       return_token_type_ids=False)
        else:
            tokenized = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
        return tokenized
