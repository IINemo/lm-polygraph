from typing import List, Dict
from .generation_metric import GenerationMetric
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from transformers import RobertaPreTrainedModel, RobertaModel, PreTrainedModel
import torch
from torch import nn
import numpy as np
import tqdm
import pdb




class RobertaClassificationHeadPruned(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj_adjusted = nn.Linear(config.hidden_size*2, 1)
        
    def forward_direction(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return x

    def forward(self, features_forward, features_backward, **kwargs):
        
        x_forward = self.forward_direction(features_forward)
        x_backward = self.forward_direction(features_backward)
        
        concatenated_hs = torch.cat([x_forward, x_backward], -1)
                
        x = self.out_proj_adjusted(concatenated_hs)
        return x


class TwoFoldRoberta(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHeadPruned(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        batch_forward, batch_backward,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs_forward = self.roberta(
            **batch_forward
        )[0]
        
        outputs_backward = self.roberta(
            **batch_backward
        )[0]
                
        logits = self.classifier(outputs_forward, outputs_backward)
                
        return logits

class PairsDatasetInference(Dataset):
    def __init__(self,texts_first, texts_second):
        self.texts_first =texts_first
        self.texts_second = texts_second
    def __len__(self):
        return len(self.texts_first)
    def __getitem__(self, idx):
        return self.texts_first[idx], self.texts_second[idx]

def cos(vecs1, vecs2):
    return np.sum(vecs1 * vecs2, axis=1) / np.sqrt(np.sum(vecs1**2, axis=1) * np.sum(vecs1**2, axis=1))


class MIS_Simcse_Metric(GenerationMetric):
    def __init__(self, batch_size):
        super().__init__(['greedy_texts'], 'sequence')
        self.mis_model = TwoFoldRoberta.from_pretrained('SkolkovoInstitute/Mutual_Implication_Score')
        self.mis_tokenizer = AutoTokenizer.from_pretrained('SkolkovoInstitute/Mutual_Implication_Score')
        self.batch_size = batch_size
        self.simcse = self.make_simcse()
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.mis_model.to(self.device)

    def __str__(self):
        return f'MIS_Simcse'

    def make_simcse(self):
        model_path = 'princeton-nlp/sup-simcse-roberta-large'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            model.to(self.device)

        # Access batch_size using self.batch_size
        batch_size = self.batch_size

        def apply(texts1, texts2, batch_size=batch_size):
            results = []
            for i in range(0, len(texts1), batch_size):
                vecs = []
                for source in texts1, texts2:
                    inputs = tokenizer(
                        source[i:i+batch_size],
                        padding=True, truncation=True, return_tensors="pt"
                    ).to(model.device)
                    with torch.inference_mode():
                        result = model(**inputs).pooler_output.cpu().numpy()
                    vecs.append(result)
                results.extend(cos(*vecs))
            return results

        return apply


    def __call__(self, stats: Dict[str, np.ndarray], target_texts: List[str],
                 target_tokens: List[List[int]]) -> np.ndarray:

        texts1 = stats['greedy_texts']
        texts2 = target_texts

        dataset_direct = PairsDatasetInference(texts1, texts2)
        dataloader_direct = DataLoader(dataset_direct, batch_size = self.batch_size)

        dataset_reverse= PairsDatasetInference(texts2, texts1)
        dataloader_reverse = DataLoader(dataset_reverse, batch_size = self.batch_size)

        preds = []

        for b1,b2 in zip(dataloader_direct, dataloader_reverse):

            with torch.no_grad():
                tokenized1 = self.mis_tokenizer(*b1, padding=True, truncation='longest_first', return_tensors="pt")

                if torch.cuda.is_available():
                    tokenized1['input_ids'] = tokenized1['input_ids'].to(self.device)
                    tokenized1['attention_mask'] = tokenized1['attention_mask'].to(self.device)

                tokenized2 = self.mis_tokenizer(*b2, padding=True, truncation='longest_first', return_tensors="pt")
                if torch.cuda.is_available():
                    tokenized2['input_ids'] = tokenized2['input_ids'].to(self.device)
                    tokenized2['attention_mask'] = tokenized2['attention_mask'].to(self.device)

                merged_prob = self.mis_model(tokenized1, tokenized2)

                merged_prob = torch.sigmoid(merged_prob)

            merged_prob = merged_prob.cpu().numpy()
            preds.extend(merged_prob)

        mis_scores = np.array([float(e) for e in preds])
        simcse_scores = np.array(self.simcse(texts1=texts1, texts2=texts2))
        result = (mis_scores + simcse_scores) / 2
        return result




