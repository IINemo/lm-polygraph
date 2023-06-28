import numpy as np
import argparse

from typing import Optional, Dict, Tuple, List

from flask import Flask, request, abort
from utils.model import Model
from utils.manager import UEManager
from utils.processor import Processor
from utils.dataset import Dataset

from app.parsers import parse_model, parse_ue_method, Estimator, normalize

app = Flask(__name__)

model: Optional[Model] = None
ue_methods: Dict[str, Estimator] = {}
cache_path: str = '/Users/ekaterinafadeeva/cache'


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
    model_path = parse_model(data['model'])

    global model
    if model is None or model.model_path != model_path:
        model = Model.from_pretrained(model_path)

    ue_method_name = data['ue']
    text = data['messages'][0]['content']

    if ue_method_name not in ue_methods.keys():
        ue_methods[ue_method_name] = parse_ue_method(ue_method_name, model_path, cache_path)

    dataset = Dataset([text], [''], batch_size=1)
    processor = ResultProcessor()
    method = ue_methods[ue_method_name]
    man = UEManager(dataset, model, [method], [], [], [processor])
    man()

    if len(processor.ue_estimations) != 1:
        abort(500,
              description=f'Internal: expected single uncertainty estimator, got: {processor.ue_estimations.keys()}')
    uncertainty = [normalize(method, x) for x in processor.ue_estimations[next(iter(processor.ue_estimations.keys()))]]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5239)
    parser.add_argument("--cache-path", type=str, default='/Users/ekaterinafadeeva/cache')
    args = parser.parse_args()
    cache_path = args.cache_path
    app.run(host='localhost', port=args.port)
