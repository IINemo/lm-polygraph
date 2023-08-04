import numpy as np
import argparse
import torch

from typing import Optional, Dict, Tuple, List

from flask import Flask, request, abort
from lm_polygraph.utils.model import WhiteboxModel, BlackboxModel
from lm_polygraph.utils.generation_parameters import GenerationParameters
from lm_polygraph.utils.manager import UEManager
from lm_polygraph.utils.processor import Processor
from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.utils.normalize import normalize_from_bounds, has_norm_bound

from .parsers import parse_model, parse_seq_ue_method, parse_tok_ue_method, Estimator, parse_ensemble

app = Flask(__name__)

model: Optional[WhiteboxModel] = None
tok_ue_methods: Dict[str, Estimator] = {}
seq_ue_methods: Dict[str, Estimator] = {}
cache_path: str = '/Users/ekaterinafadeeva/cache'
density_based_names: List[str] = ["Mahalanobis Distance", "Mahalanobis Distance - encoder",
                                  "RDE", "RDE - encoder"]
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


def _get_confidence(processor: ResultProcessor, methods: List[Estimator], level: str) -> Tuple[List, str]:
    if len(methods) == 0:
        return [], 'none'
    condifences, normalized_confidences = [], []
    normalization = 'bounds'
    for method in methods:
        condifences.append([])
        normalized_confidences.append([])
        for x in processor.ue_estimations[level, str(method)]:
            condifences[-1].append(-x)
            if not has_norm_bound(method):
                normalization = 'none'
            normalized_confidences[-1].append(1 - normalize_from_bounds(method, x))
        print(' {} Confidence: {}'.format(str(method), condifences[-1]))
    if normalization != 'none':
        condifences = normalized_confidences

    condifences = np.array(condifences)
    condifences = condifences.reshape(len(methods), len(condifences[0]))
    conf_list = np.mean(condifences, axis=0).tolist() if len(condifences[0]) != 0 else []
    return conf_list, normalization


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


def _align_tokenwise_confidences(tokens, confidences):
    if len(confidences) == 0:
        return []
    confidences_grouped = np.zeros_like(confidences)
    word_len = 0
    for i, token in enumerate(tokens):
        confidences_grouped[i] = confidences[i]
        if (" " in token) or ((word_len > 0) and ((len(tokens) - 1) == i)):
            confidences_grouped[i - word_len: i + 1] = np.min(confidences_grouped[i - word_len: i + 1])
            word_len = 0
        else:
            word_len += 1
    return confidences_grouped.tolist()


@app.route('/chat/completions', methods=['GET', 'POST'])
# @app.route('/get-prompt-result', methods=['GET', 'POST'])
def generate():
    data = request.get_json()
    print(f'Request data: {data}')

    topk = int(data['parameters']['topk'])
    parameters = GenerationParameters(
        temperature=float(data['parameters']['temperature']),
        topk=topk,
        topp=float(data['parameters']['topp']),
        do_sample=(topk > 1),
        num_beams=int(data['parameters']['num_beams']),
        repetition_penalty=float(data['parameters']['repetition_penalty']),
    )
    global model
    ensemble_model = None
    if data['model'] == 'Ensemble':
        model_path = data['ensembles']
        model, ensemble_model = parse_ensemble(model_path, device=device)
    elif model is None or model.model_path != parse_model(data['model']):
        model_path = parse_model(data['model'])
        if model_path.startswith('openai-'):
            model = BlackboxModel(data['openai_key'], model_path[len('openai-'):])
        else:
            model = WhiteboxModel.from_pretrained(model_path, device=device)
    else:
        model_path = parse_model(data['model'])
    model.parameters = parameters

    tok_ue_method_names = data['tok_ue'] if 'tok_ue' in data.keys() and data['tok_ue'] is not None else []
    seq_ue_method_names = data['seq_ue'] if 'seq_ue' in data.keys() and data['seq_ue'] is not None else []
    text = data['messages'][0]['content']

    for ue_method_name in tok_ue_method_names:
        if (ue_method_name not in tok_ue_methods.keys()) or (ue_method_name in density_based_names):
            tok_ue_methods[ue_method_name] = parse_tok_ue_method(ue_method_name, model_path, cache_path)

    for ue_method_name in seq_ue_method_names:
        if (ue_method_name not in seq_ue_methods.keys()) or (ue_method_name in density_based_names):
            seq_ue_methods[ue_method_name] = parse_seq_ue_method(ue_method_name, model_path, cache_path)

    dataset = Dataset([text], [''], batch_size=1)
    processor = ResultProcessor()

    tok_methods = [tok_ue_methods[ue_method_name] for ue_method_name in tok_ue_method_names]
    seq_methods = [seq_ue_methods[ue_method_name] for ue_method_name in seq_ue_method_names]

    try:
        man = UEManager(dataset, model, tok_methods + seq_methods, [], [],
                        [processor],
                        ensemble_model=ensemble_model,
                        ignore_exceptions=False)
        man()
    except Exception as e:
        abort(400, str(e))

    if len(processor.ue_estimations) != len(tok_methods) + len(seq_methods):
        abort(500, f'Internal: expected {len(tok_methods) + len(seq_methods)} estimations, '
                   f'got: {processor.ue_estimations.keys()}')
    greedy_text = processor.stats.get('greedy_texts', processor.stats.get('blackbox_greedy_texts', None))[0]
    print(' Generation: {}'.format(greedy_text))

    tok_conf, tok_norm = _get_confidence(processor, tok_methods, 'token')
    seq_conf, seq_norm = _get_confidence(processor, seq_methods, 'sequence')
    if 'greedy_tokens' in processor.stats.keys():
        tokens = []
        for t in processor.stats['greedy_tokens'][0][:-1]:
            tokens.append(model.tokenizer.decode([t]))
    else:
        tokens = [greedy_text]
    if len(tokens) > 0:
        tokens[0] = tokens[0].lstrip()
        tokens[-1] = tokens[-1].rstrip()

    if model.model_type == "Seq2SeqLM":
        tokens = _add_spaces_to_tokens(model.tokenizer, processor.stats, tokens)
        tok_conf = _align_tokenwise_confidences(tokens, tok_conf)

    return {
        'generation': tokens,
        'token_confidence': tok_conf,
        'sequence_confidence': seq_conf,
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
