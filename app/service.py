import numpy as np
import argparse
import torch

from typing import Optional, Dict, Tuple, List

from flask import Flask, request, abort
from utils.model import Model
from utils.generation_parameters import GenerationParameters
from utils.manager import UEManager
from utils.processor import Processor
from utils.dataset import Dataset
from utils.normalize import normalize_from_bounds, has_norm_bound

from app.parsers import parse_model, parse_seq_ue_method, parse_tok_ue_method, Estimator, parse_ensemble

app = Flask(__name__)

model: Optional[Model] = None
tok_ue_methods: Dict[str, Estimator] = {}
seq_ue_methods: Dict[str, Estimator] = {}
cache_path: str = '/Users/ekaterinafadeeva/cache'
device: str = 'cpu'


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


def _get_uncertainty(processor: ResultProcessor, methods: List[Estimator], level: str) -> Tuple[List, str]:
    if len(methods) == 0:
        return []
    uncertainties, normalized_uncertainties = [], []
    normalization = 'bounds'
    for method in methods:
        uncertainties.append([])
        normalized_uncertainties.append([])
        for x in processor.ue_estimations[level, str(method)]:
            uncertainties[-1].append(x)
            if not has_norm_bound(method):
                normalization = 'none'
            normalized_uncertainties[-1].append(normalize_from_bounds(method, x))
        print(' {} Uncertainty: {}'.format(str(method), uncertainties[-1]))
    if normalization != 'none':
        uncertainties = normalized_uncertainties
    uncertainties = np.array(uncertainties)
    uncertainties = uncertainties.reshape(len(methods), len(uncertainties[0]))
    ue_list = np.mean(uncertainties, axis=0).tolist() if len(uncertainties[0]) != 0 else []
    return ue_list, normalization


def _add_spaces_to_tokens(tokenizer, stats, tokens):
    curr_len = 0
    tokens_with_spaces = []
    sequence = tokenizer.batch_decode(stats['greedy_tokens'], skip_special_tokens=True)[0]
    for token in tokens:
        if ((len(token) + curr_len) < len(sequence)) and (sequence[len(token) + curr_len]) == " ":
            tokens_with_spaces.append(token + " ")
            curr_len += 1
        else:
            tokens_with_spaces.append(token)
        curr_len += len(token)
    return tokens_with_spaces


def _align_tokenwise_uncertainty(tokens, uncertainties):
    if len(uncertainties) == 0:
        return []
    uncertainties_grouped = np.zeros_like(uncertainties)
    word_len = 0
    for i, token in enumerate(tokens):
        uncertainties_grouped[i] = uncertainties[i]
        if (" " in token) or ((word_len > 0) and ((len(tokens) - 1) == i)):
            uncertainties_grouped[i - word_len: i + 1] = np.min(uncertainties_grouped[i - word_len: i + 1])
            word_len = 0
        else:
            word_len += 1
    return uncertainties_grouped.tolist()


@app.route('/chat/completions', methods=['GET', 'POST'])
def generate():
    data = request.get_json()
    print(f'Request data: {data}')

    parameters = GenerationParameters(
        temperature=float(data['parameters']['temperature']),
        topk=int(data['parameters']['topk']),
        topp=float(data['parameters']['topp']),
        do_sample=(data['parameters']['do_sample'] == 'on'),
        num_beams=int(data['parameters']['num_beams']),
    )
    global model
    ensemble_model = None
    if data['model'] == 'Ensemble':
        model_path = data['ensembles']
        model, ensemble_model = parse_ensemble(model_path, device=device)
    elif model is None or model.model_path != parse_model(data['model']):
        model_path = parse_model(data['model'])
        model = Model.from_pretrained(model_path, device=device)
    else:
        model_path = parse_model(data['model'])
    model.parameters = parameters

    tok_ue_method_names = data['tok_ue'] if 'tok_ue' in data.keys() and data['tok_ue'] is not None else []
    seq_ue_method_names = data['seq_ue'] if 'seq_ue' in data.keys() and data['seq_ue'] is not None else []
    text = data['messages'][0]['content']

    for ue_method_name in tok_ue_method_names:
        if ue_method_name not in tok_ue_methods.keys():
            tok_ue_methods[ue_method_name] = parse_tok_ue_method(ue_method_name, model_path, cache_path)

    for ue_method_name in seq_ue_method_names:
        if ue_method_name not in seq_ue_methods.keys():
            seq_ue_methods[ue_method_name] = parse_seq_ue_method(ue_method_name, model_path, cache_path)

    dataset = Dataset([text], [''], batch_size=1)
    processor = ResultProcessor()

    tok_methods = [tok_ue_methods[ue_method_name] for ue_method_name in tok_ue_method_names]
    seq_methods = [seq_ue_methods[ue_method_name] for ue_method_name in seq_ue_method_names]

    man = UEManager(dataset, model, tok_methods + seq_methods, [], [],
                    [processor],
                    ensemble_model=ensemble_model,
                    ignore_exceptions=False)
    man()

    if len(processor.ue_estimations) != len(tok_methods) + len(seq_methods):
        abort(500,
              description=f'Internal: expected {len(tok_methods) + len(seq_methods)} estimations, '
                          f'got: {processor.ue_estimations.keys()}')
    print(' Generation: {}'.format(processor.stats['greedy_texts'][0]))

    tok_uncertainty, tok_norm = _get_uncertainty(processor, tok_methods, 'token')
    seq_uncertainty, seq_norm = _get_uncertainty(processor, seq_methods, 'sequence')
    tokens = []
    for t in processor.stats['greedy_tokens'][0][:-1]:
        tokens.append(model.tokenizer.decode([t]))
    if len(tokens) > 0:
        tokens[0] = tokens[0].lstrip()
        tokens[-1] = tokens[-1].rstrip()

    if model.model_type == "Seq2SeqLM":
        tokens = _add_spaces_to_tokens(model.tokenizer, processor.stats, tokens)
        tok_uncertainty = _align_tokenwise_uncertainty(tokens, tok_uncertainty)

    return {
        'generation': tokens,
        'token_uncertainty': tok_uncertainty,
        'sequence_uncertainty': seq_uncertainty,
        'token_normalization': tok_norm,
        'sequence_normalization': seq_norm,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5239)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cache-path", type=str, default='/Users/romanvashurin/cache')
    args = parser.parse_args()
    cache_path = args.cache_path
    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda:0'
    app.run(host='localhost', port=args.port)
