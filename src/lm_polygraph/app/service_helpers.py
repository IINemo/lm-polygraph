import numpy as np
import string

from typing import Optional, Dict, Tuple, List

from flask import abort
from lm_polygraph.utils.model import WhiteboxModel, BlackboxModel
from lm_polygraph.utils.generation_parameters import GenerationParameters
from lm_polygraph.utils.manager import UEManager
from lm_polygraph.utils.processor import Processor
from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.utils.normalize import (
    normalize_ue,
    can_normalize_ue,
    can_get_calibration_conf,
    calibration_confidence,
)

from .parsers import (
    parse_model,
    parse_seq_ue_method,
    parse_tok_ue_method,
    Estimator,
    parse_ensemble,
)


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


class Responder:
    def __init__(self, cache_path, device):
        self.model: Optional[WhiteboxModel] = None
        self.tok_ue_methods: Dict[str, Estimator] = {}
        self.seq_ue_methods: Dict[str, Estimator] = {}
        self.density_based_names: List[str] = [
            "Mahalanobis Distance",
            "Mahalanobis Distance - encoder",
            "RDE",
            "RDE - encoder",
        ]
        self.cache_path: str = cache_path
        self.device = device

    def _get_confidence(
        self,
        processor: ResultProcessor,
        methods: List[Estimator],
        level: str,
        model_path: str,
    ) -> Tuple[List, str]:
        if len(methods) == 0:
            return [], "none"
        condifences, normalized_confidences = [], []
        normalization = "bounds"
        for method in methods:
            condifences.append([])
            normalized_confidences.append([])
            for x in processor.ue_estimations[level, str(method)]:
                condifences[-1].append(-x)
                if not ("openai" in model_path):
                    if not can_get_calibration_conf(
                        method, model_path, self.cache_path
                    ):
                        normalization = "none"
                    else:
                        normalized_confidences[-1].append(
                            calibration_confidence(
                                method, model_path, x, self.cache_path
                            )
                        )
                else:
                    if not can_normalize_ue(method, model_path, self.cache_path):
                        normalization = "none"
                    else:
                        normalized_confidences[-1].append(
                            1 - normalize_ue(method, model_path, x, self.cache_path)
                        )

            print(" {} Confidence: {}".format(str(method), condifences[-1]))
        if normalization != "none":
            condifences = normalized_confidences

        condifences = np.array(condifences)
        condifences = condifences.reshape(len(methods), len(condifences[0]))
        conf_list = (
            np.mean(condifences, axis=0).tolist() if len(condifences[0]) != 0 else []
        )
        return conf_list, normalization

    def _add_spaces_to_tokens(self, tokenizer, stats, tokens):
        curr_len = 0
        tokens_with_spaces = []
        sequence = tokenizer.batch_decode(
            stats["greedy_tokens"], skip_special_tokens=True
        )[0]
        for token in tokens:
            if ((len(token) + curr_len) < len(sequence)) and (
                sequence[len(token) + curr_len]
            ) == " ":
                tokens_with_spaces.append(token + " ")
                curr_len += 1
            else:
                tokens_with_spaces.append(token)
            curr_len += len(token)
        return tokens_with_spaces

    def _split_spaces(
        self,
        tokens,
        conf,
        split=string.punctuation.replace("'", "") + " \n",
        strip=(" ", "\n"),
    ):
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
        while len(new_tokens) > 0 and new_tokens[0] in strip:
            new_tokens = new_tokens[1:]
            new_conf = new_conf[1:]
        while len(new_tokens) > 0 and new_tokens[-1] in strip:
            new_tokens = new_tokens[:-1]
            new_conf = new_conf[:-1]
        return new_tokens, new_conf

    def _merge_into_words(
        self, tokens, confidences, split=string.punctuation.replace("'", "") + " \n"
    ):
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
                word_conf.append(np.min(confidences[word_start : i + 1]))
                words.append("".join(tokens[word_start : i + 1]))
                word_start = i + 1
        return words, word_conf

    def _generate_response(self, data):
        print(f"Request data: {data}")

        topk = int(data["topk"])
        parameters = GenerationParameters(
            temperature=float(data["temperature"]),
            topk=topk,
            topp=float(data["topp"]),
            do_sample=(topk > 1),
            num_beams=int(data["num_beams"]),
            presence_penalty=float(data["presence_penalty"]),
            repetition_penalty=float(data["repetition_penalty"]),
            allow_newlines=("Dolly" not in data["model"]),
        )
        ensemble_model = None
        if data["model"] == "Ensemble":
            model_path = data["ensembles"]
            self.model, ensemble_model = parse_ensemble(model_path, device=self.device)
        elif self.model is None or self.model.model_path != parse_model(data["model"]):
            model_path = parse_model(data["model"])
            if model_path.startswith("openai-"):
                self.model = BlackboxModel(
                    data["openai_key"], model_path[len("openai-") :]
                )
            else:
                # load_in_8bit = ('cuda' in self.device) and any(s in model_path for s in ['7b', '12b', '13b'])
                load_in_8bit = False
                self.model = WhiteboxModel.from_pretrained(
                    model_path, device=self.device, load_in_8bit=load_in_8bit
                )
        else:
            model_path = parse_model(data["model"])
        self.model.parameters = parameters

        # TODO this is temporary fix for single instead of multi-choice, should be changed with katya proper version
        # As back expects list and front sends the string at the moment
        tok_ue_method_names = (
            [data.get("tok_ue")] if data.get("tok_ue") is not None else []
        )
        seq_ue_method_names = (
            [data.get("seq_ue")] if data.get("seq_ue") is not None else []
        )
        # tok_ue_method_names = data['tok_ue'] if 'tok_ue' in data.keys() and data['tok_ue'] is not None else []
        # seq_ue_method_names = data['seq_ue'] if 'seq_ue' in data.keys() and data['seq_ue'] is not None else []

        text = data["prompt"]

        if "dolly" in model_path.lower():
            text = (
                f"Below is an instruction that describes a task. Write a response that appropriately completes "
                f"the request.\n### Instruction:\n{text}\n### Response:\n"
            )
        if any(x in model_path.lower() for x in ["vicuna", "llama"]):
            text += "\n\n"

        for ue_method_name in tok_ue_method_names:
            if (ue_method_name not in self.tok_ue_methods.keys()) or (
                ue_method_name in self.density_based_names
            ):
                self.tok_ue_methods[ue_method_name] = parse_tok_ue_method(
                    ue_method_name, model_path, self.cache_path
                )

        for ue_method_name in seq_ue_method_names:
            if (ue_method_name not in self.seq_ue_methods.keys()) or (
                ue_method_name in self.density_based_names
            ):
                self.seq_ue_methods[ue_method_name] = parse_seq_ue_method(
                    ue_method_name, model_path, self.cache_path
                )

        dataset = Dataset([text], [""], batch_size=1)
        processor = ResultProcessor()

        tok_methods = [
            self.tok_ue_methods[ue_method_name]
            for ue_method_name in tok_ue_method_names
        ]
        seq_methods = [
            self.seq_ue_methods[ue_method_name]
            for ue_method_name in seq_ue_method_names
        ]

        try:
            man = UEManager(
                dataset,
                self.model,
                tok_methods + seq_methods,
                [],
                [],
                [processor],
                ensemble_model=ensemble_model,
                ignore_exceptions=False,
            )
            man()
        except Exception as e:
            import traceback

            trace = traceback.format_exc()
            print("Error: " + str(e))
            print("Stack trace:\n" + trace)
            abort(400, str(e))

        if len(processor.ue_estimations) != len(tok_methods) + len(seq_methods):
            abort(
                500,
                f"Internal: expected {len(tok_methods) + len(seq_methods)} estimations, "
                f"got: {processor.ue_estimations.keys()}",
            )
        greedy_text = processor.stats.get(
            "greedy_texts", processor.stats.get("blackbox_greedy_texts", None)
        )[0]
        print(" Generation: {}".format(greedy_text))

        tok_conf, tok_norm = self._get_confidence(
            processor, tok_methods, "token", model_path
        )
        seq_conf, seq_norm = self._get_confidence(
            processor, seq_methods, "sequence", model_path
        )
        if "greedy_tokens" in processor.stats.keys():
            tokens = []
            for t in processor.stats["greedy_tokens"][0][:-1]:
                if t not in [
                    self.model.tokenizer.bos_token_id,
                    self.model.tokenizer.eos_token_id,
                    self.model.tokenizer.pad_token_id,
                ]:
                    tokens.append(self.model.tokenizer.decode([t]))
        else:
            tokens = [greedy_text]

        for i in range(len(tokens)):
            if "### End" in tokens[i]:  # for dolly
                tokens = tokens[:i]
                tok_conf = tok_conf[:i]
                break

        if isinstance(self.model, WhiteboxModel):
            if len(tok_methods) == 0:
                tok_conf = [0 for _ in tokens]
            if self.model.model_type == "Seq2SeqLM" or any(
                x in self.model.model_path.lower() for x in ["llama", "vicuna"]
            ):
                tokens = self._add_spaces_to_tokens(
                    self.model.tokenizer, processor.stats, tokens
                )
            tokens, tok_conf = self._split_spaces(tokens, tok_conf)
            tokens, tok_conf = self._merge_into_words(tokens, tok_conf)
            if len(tok_methods) == 0:
                tok_conf = []

        for i in range(1, len(tokens)):
            if tokens[i] == "\n" and tokens[i - 1] == "\n":
                tokens[i - 1] = ""  # prevent multiple newlines (for vicuna)

        return {
            "generation": tokens,
            "token_confidence": tok_conf,
            "sequence_confidence": seq_conf,
            "token_normalization": tok_norm,
            "sequence_normalization": seq_norm,
        }
