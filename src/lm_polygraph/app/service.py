import os
import numpy as np
import argparse
import torch
import string

from typing import Optional, Dict, Tuple, List

from flask import Flask, request, abort, send_from_directory, render_template
from lm_polygraph.utils.model import WhiteboxModel, BlackboxModel
from lm_polygraph.utils.generation_parameters import GenerationParameters
from lm_polygraph.utils.manager import UEManager
from lm_polygraph.utils.processor import Processor
from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.utils.normalize import normalize_ue, can_normalize_ue

from .parsers import parse_model, parse_seq_ue_method, parse_tok_ue_method, Estimator, parse_ensemble

# static_folder = 'src/lm_polygraph/app/client'
static_folder = 'client'
app = Flask(__name__, static_folder=static_folder)

model: Optional[WhiteboxModel] = None
tok_ue_methods: Dict[str, Estimator] = {}
seq_ue_methods: Dict[str, Estimator] = {}
cache_path: str = None
density_based_names: List[str] = ["Mahalanobis Distance", "Mahalanobis Distance - encoder",
                                  "RDE", "RDE - encoder"]
device: str = 'cpu'


@app.route('/')
def serve_index():
    return send_from_directory(os.path.join(app.root_path, static_folder), 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.path.join(app.root_path, static_folder), filename)


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


def _get_confidence(processor: ResultProcessor, methods: List[Estimator], level: str, model_path: str) -> Tuple[
    List, str]:
    if len(methods) == 0:
        return [], 'none'
    condifences, normalized_confidences = [], []
    normalization = 'bounds'
    for method in methods:
        condifences.append([])
        normalized_confidences.append([])
        for x in processor.ue_estimations[level, str(method)]:
            condifences[-1].append(-x)
            if not can_normalize_ue(method, model_path):
                normalization = 'none'
            else:
                normalized_confidences[-1].append(1 - normalize_ue(method, model_path, x))
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


def _split_spaces(tokens, conf, model_path, split=string.punctuation.replace("'", '') + " \n", strip=(' ', '\n')):
    new_tokens, new_conf = [], []
    for i, (t, c) in enumerate(zip(tokens, conf)):
        while any(t.startswith(s) for s in split):
            new_tokens.append(t[0])
            new_conf.append(1.0)
            t = t[1:]
        stack_tokens, stack_conf = [], []
        while any(t.endswith(s) for s in split):
            stack_tokens.append(t[-1])
            stack_conf.append(1.0)
            t = t[:-1]
        if len(t) > 0:
            new_tokens.append(t)
            new_conf.append(c)
        new_tokens += stack_tokens[::-1]
        new_conf += stack_conf[::-1]
        if 'llama' in model_path.lower():
            new_tokens.append(' ')
            new_conf.append(1.0)
    while len(new_tokens) > 0 and new_tokens[0] in strip:
        new_tokens = new_tokens[1:]
        new_conf = new_conf[1:]
    while len(new_tokens) > 0 and new_tokens[-1] in strip:
        new_tokens = new_tokens[:-1]
        new_conf = new_conf[:-1]
    return new_tokens, new_conf


def _merge_into_words(tokens, confidences, split=string.punctuation.replace("'", '') + " \n"):
    if len(confidences) == 0:
        return tokens, []
    words, word_conf = [], []
    word_start = 0
    for i, (token, conf) in enumerate(zip(tokens, confidences)):
        if token in split:
            word_conf.append(conf)
            words.append(token)
            word_start = i + 1
        elif i + 1 == len(tokens) or tokens[i + 1] in split:
            word_conf.append(np.min(confidences[word_start:i + 1]))
            words.append(''.join(tokens[word_start:i + 1]))
            word_start = i + 1
    return words, word_conf


@app.route('/get-prompt-result', methods=['GET', 'POST'])
def generate():
    data = request.get_json()
    print(f'Request data: {data}')

    topk = int(data['topk'])
    parameters = GenerationParameters(
        temperature=float(data['temperature']),
        topk=topk,
        topp=float(data['topp']),
        do_sample=(topk > 1),
        num_beams=int(data['num_beams']),
        repetition_penalty=float(data['repetition_penalty']),
        allow_newlines=('Dolly' not in data['model']),
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
            load_in_8bit = ('cuda' in device) and any(s in model_path for s in ['7b', '12b', '13b'])
            model = WhiteboxModel.from_pretrained(
                model_path, device=device, load_in_8bit=load_in_8bit)
    else:
        model_path = parse_model(data['model'])
    model.parameters = parameters

    tok_ue_method_names = data['tok_ue'] if 'tok_ue' in data.keys() and data['tok_ue'] is not None else []
    seq_ue_method_names = data['seq_ue'] if 'seq_ue' in data.keys() and data['seq_ue'] is not None else []
    text = data['prompt']

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

    tok_conf, tok_norm = _get_confidence(processor, tok_methods, 'token', model_path)
    seq_conf, seq_norm = _get_confidence(processor, seq_methods, 'sequence', model_path)
    if 'greedy_tokens' in processor.stats.keys():
        tokens = []
        for t in processor.stats['greedy_tokens'][0][:-1]:
            if t not in [model.tokenizer.bos_token_id, model.tokenizer.eos_token_id, model.tokenizer.pad_token_id]:
                tokens.append(model.tokenizer.decode([t]))
    else:
        tokens = [greedy_text]

    if type(model) == WhiteboxModel and len(tok_methods) > 0:
        if model.model_type == "Seq2SeqLM": 
            tokens = _add_spaces_to_tokens(model.tokenizer, processor.stats, tokens)
        tokens, tok_conf = _split_spaces(tokens, tok_conf, model_path)
        tokens, tok_conf = _merge_into_words(tokens, tok_conf)

    return {
        'generation': tokens,
        'token_confidence': tok_conf,
        'sequence_confidence': seq_conf,
        'token_normalization': tok_norm,
        'sequence_normalization': seq_norm,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cache-path", type=str,
                        default=os.path.expanduser('~') + '/.cache')

    args = parser.parse_args()
    cache_path = args.cache_path
    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda:0'
    app.run(host='0.0.0.0', port=args.port, debug=True)
