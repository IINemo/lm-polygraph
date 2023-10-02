import numpy as np

from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.estimators.common import DEBERTA
import torch.nn as nn

softmax = nn.Softmax(dim=1)

class SemanticMatrixCalculator(StatCalculator):
    def __init__(self):
        super().__init__(['semantic_matrix_entail',
                          'semantic_matrix_contra',
                          'semantic_matrix_classes'],
                         ['blackbox_sample_texts'])
        DEBERTA.setup()

    def __call__(self, dependencies: Dict[str, np.array], texts: List[str], model: WhiteboxModel, max_new_tokens: int = 100) -> Dict[str, np.ndarray]:
        batch_texts = dependencies['blackbox_sample_texts']

        batch_pairs = []
        for texts in batch_texts:
            batch_pairs.append(itertools.product(texts, texts))

        W = np.eye(len(texts))
        device = DEBERTA.device 
        softmax = nn.Softmax(dim=1)

        for pairs in batch_pairs:
            encoded = DEBERTA.deberta_tokenizer.batch_encode_plus(pairs, padding=True, return_tensors='pt').to(device)
            logits = DEBERTA.deberta(**encoded).logits.detach().to(device)
            probs = softmax(logits)

            entail_probs = probs[:, DEBERTA.deberta.config.label2id['ENTAILMENT']]
            contra_probs = probs[:, DEBERTA.deberta.config.label2id['CONTRADICTION']]
            class_preds = probs.argmax(-1)

            mat_shape = (len(texts), len(texts))

            E = entail_probs.view(mat_shape)
            C = contra_probs.view(mat_shape)
            P = class_preds.view(mat_shape)

        return {'semantic_matrix_entail': E,
                'semantic_matrix_contra': C,
                'semantic_matrix_classes': P}
