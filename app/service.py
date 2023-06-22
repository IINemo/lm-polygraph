import numpy as np

from typing import Optional, Dict, Tuple, List

from flask import Flask, request, abort
from utils.model import Model
from estimators import *
from utils.manager import UEManager
from utils.processor import Processor
from utils.dataset import Dataset

app = Flask(__name__)

model: Optional[Model] = None
ue_methods: Dict[str, Estimator] = {}


def create_method(method_name: str) -> Estimator:
    match method_name:
        case 'Maximum probability':
            return MaxProbabilityToken()
        case 'Entropy':
            return EntropyToken()
        case _:
            raise Exception(f'Unknown method: {method_name}')


class ResultProcessor(Processor):
    def __init__(self):
        self.stats = {}
        self.ue_estimations = {}

    def on_batch(
            self,
            batch_stats: Dict[str, np.ndarray],
            batch_gen_metrics: Dict[Tuple[str, str], List[float]],
            batch_estimations: Dict[Tuple[str, str], List[float]],
    ):
        self.stats = batch_stats
        self.ue_estimations = batch_estimations


@app.route('/chat/completions', methods=['GET', 'POST'])
def generate():
    data = request.get_json()
    print(f'Request data: {data}')

    global model
    if model is None or model.model_path != data['model']:
        model = Model.from_pretrained(data['model'])

    ue_method_name = data['ue']
    text = data['messages'][0]['content']

    if ue_method_name not in ue_methods.keys():
        ue_methods[ue_method_name] = create_method(ue_method_name)

    dataset = Dataset([text], [''], batch_size=1)
    processor = ResultProcessor()
    man = UEManager(dataset, model, [ue_methods[ue_method_name]], [], [], [processor])
    man()

    if len(processor.ue_estimations) != 1:
        abort(500,
              description=f'Internal: expected single uncertainty estimator, got: {processor.ue_estimations.keys()}')
    uncertainty = [-x for x in processor.ue_estimations[next(iter(processor.ue_estimations.keys()))]]
    print(' Generation: {}'.format(processor.stats['greedy_texts'][0]))
    print(' Uncertainty: {}'.format(uncertainty))
    tokens = []
    for t in processor.stats['greedy_tokens'][0][:-1]:
        tokens.append(model.tokenizer.decode([t]))
    if len(tokens) > 0:
        tokens[0] = tokens[0].lstrip()
        tokens[-1] = tokens[-1].rstrip()

    return {'generation': tokens, 'uncertainty': uncertainty}


if __name__ == '__main__':
    app.run(host='localhost', port=5000)
