import warnings
from dataclasses import dataclass
from typing import Optional, Union, Dict, List, Tuple
from scipy.stats import entropy

import torch
import torch.distributed as dist
from torch import nn
from transformers import GenerationMixin

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import (
    SampleOutput,
    SampleDecoderOnlyOutput,
    ModelOutput,
)


class EnsembleSampleMixin(GenerationMixin):
    def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:

            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation_utils.SampleDecoderOnlyOutput`], [`~generation_utils.SampleEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation_utils.SampleDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation_utils.SampleEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     TopKLogitsWarper,
        ...     TemperatureLogitsWarper,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
        >>> model.config.pad_token_id = model.config.eos_token_id

        >>> input_prompt = "Today is a beautiful day, and"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(15, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )
        >>> # instantiate logits processors
        >>> logits_warper = LogitsProcessorList(
        ...     [
        ...         TopKLogitsWarper(50),
        ...         TemperatureLogitsWarper(0.7),
        ...     ]
        ... )

        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> torch.manual_seed(0)  # doctest: +IGNORE_RESULT
        >>> outputs = model.sample(
        ...     input_ids,
        ...     logits_processor=logits_processor,
        ...     logits_warper=logits_warper,
        ...     stopping_criteria=stopping_criteria,
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Today is a beautiful day, and a wonderful day.\n\nI was lucky enough to meet the']
        ```"""
        if getattr(self, "models", None) is None:
            self._models_list = []
        # init values
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria
            if stopping_criteria is not None
            else StoppingCriteriaList()
        )
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(
                stopping_criteria, max_length
            )
        logits_warper = (
            logits_warper if logits_warper is not None else LogitsProcessorList()
        )
        pad_token_id = (
            pad_token_id
            if pad_token_id is not None
            else self.generation_config.pad_token_id
        )
        eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else self.generation_config.eos_token_id
        )
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = (
            torch.tensor(eos_token_id).to(input_ids.device)
            if eos_token_id is not None
            else None
        )
        output_scores = (
            output_scores
            if output_scores is not None
            else self.generation_config.output_scores
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        models_scores = [] if (return_dict_in_generate and output_scores) else None
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        cross_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = (
                model_kwargs["encoder_outputs"][0].get("attentions")
                if output_attentions
                else None
            )
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"][0].get("hidden_states")
                if output_hidden_states
                else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(
            input_ids.shape[0], dtype=torch.long, device=input_ids.device
        )

        this_peer_finished = False  # used by synced_gpus only

        encoder_outputs = model_kwargs.pop("encoder_outputs")
        calculate_entropies = getattr(self, "calculate_entropies", True)

        self.models_beam_tokens_iter = None
        models_beam_next_token_logits = []

        pe_uncertainties = {}
        ep_uncertainties = {}
        if calculate_entropies:
            pe_uncertainties["total_uncertainty"] = []
            pe_uncertainties["data_uncertainty"] = []
            pe_uncertainties["mutual_information"] = []
            pe_uncertainties["epkl_total_uncertainty"] = []
            pe_uncertainties["epkl"] = []
            pe_uncertainties["rmi"] = []

            ep_uncertainties["total_uncertainty"] = []
            ep_uncertainties["data_uncertainty"] = []
            ep_uncertainties["mutual_information"] = []
            ep_uncertainties["epkl_total_uncertainty"] = []
            ep_uncertainties["epkl"] = []
            ep_uncertainties["rmi"] = []

        if self.mc:
            num_models = self.mc_models_num
        else:
            num_models = len(self.models)

        self.models_beam_logits_iter = None

        # auto-regressive generation
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(
                    0.0 if this_peer_finished else 1.0
                ).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = []
            if self.mc:
                for i in range(self.mc_models_num):
                    torch.manual_seed(self.mc_seeds[i])
                    model_inputs.append(
                        self.prepare_inputs_for_generation(
                            input_ids,
                            encoder_outputs=encoder_outputs[i],
                            **model_kwargs,
                        )
                    )
                torch.manual_seed(self.base_seed)
            else:
                for i in range(num_models):
                    dev = self.models[i].device
                    input_ids.to(dev)
                    model_kwargs = {
                        k: v.to(dev)
                        for k, v in model_kwargs.items()
                        if hasattr(v, "to")
                    }
                    model_inputs.append(
                        self.prepare_inputs_for_generation(
                            input_ids.to(dev),
                            encoder_outputs=encoder_outputs[i],
                            **model_kwargs,
                        )
                    )

            # forward pass to get next token
            models_next_token_probas = []
            models_next_token_logits = []
            models_entropies = []
            models_outputs = []
            if self.mc:
                for i in range(self.mc_models_num):
                    torch.manual_seed(self.mc_seeds[i])
                    models_outputs.append(
                        self(
                            **model_inputs[i],
                            return_dict=True,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                        )
                    )

                    if synced_gpus and this_peer_finished:
                        continue  # don't waste resources running the code we don't need

                torch.manual_seed(self.base_seed)
            else:
                for i, model in enumerate(self.models):
                    models_outputs.append(
                        model(
                            **model_inputs[i],
                            return_dict=True,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                        )
                    )

                    if synced_gpus and this_peer_finished:
                        continue  # don't waste resources running the code we don't need

            for outputs in models_outputs:
                model_next_token_logits = outputs.logits[:, -1, :].to(self.device)
                model_next_token_scores = nn.functional.log_softmax(
                    model_next_token_logits, dim=-1
                )  # (batch_size * num_beams, vocab_size)
                models_next_token_logits.append(model_next_token_scores)
                models_next_token_probas.append(
                    model_next_token_scores.exp()
                )  # probas of one model
                if calculate_entropies:
                    model_entropy = torch.tensor(
                        entropy(models_next_token_probas[-1].cpu().numpy(), axis=-1)
                    ).to(input_ids.device)
                    models_entropies.append(model_entropy)

            pe_next_token_scores = (
                torch.stack(models_next_token_logits).logsumexp(dim=0)
                - torch.tensor(num_models).log()
            )

            if self.models_beam_logits_iter is None:
                self.models_beam_logits_iter = torch.zeros(
                    (num_models, input_ids.shape[0], 1)
                ).to(input_ids.device)
                models_beam_logits = self.models_beam_logits_iter

            denom = models_beam_logits.logsumexp(dim=0)
            num = (
                torch.stack(models_next_token_logits) + models_beam_logits
            ).logsumexp(dim=0)
            ep_next_token_scores = num - denom

            pe_next_token_probas = pe_next_token_scores.exp()
            ep_next_token_probas = ep_next_token_scores.exp()

            if calculate_entropies:
                pe_token_total_unc = torch.tensor(
                    entropy(pe_next_token_probas.cpu().numpy(), axis=-1)
                ).to(input_ids.device)
                pe_token_data_unc = torch.stack(models_entropies).mean(0)
                pe_token_mi = pe_token_total_unc - pe_token_data_unc
                pe_token_av_logs = torch.stack(models_next_token_logits).mean(0)
                pe_token_epkl_total_unc = -(
                    pe_token_av_logs * pe_next_token_probas
                ).sum(-1)
                pe_token_epkl = pe_token_epkl_total_unc - pe_token_data_unc
                pe_token_rmi = pe_token_epkl_total_unc - pe_token_total_unc

                ep_token_total_unc = torch.tensor(
                    entropy(ep_next_token_probas.cpu().numpy(), axis=-1)
                ).to(input_ids.device)
                ep_token_data_unc = torch.stack(models_entropies).mean(0)
                ep_token_mi = ep_token_total_unc - ep_token_data_unc
                ep_token_av_logs = torch.stack(models_next_token_logits).mean(0)
                ep_token_epkl_total_unc = -(
                    ep_token_av_logs * ep_next_token_probas
                ).sum(-1)
                ep_token_epkl = ep_token_epkl_total_unc - ep_token_data_unc
                ep_token_rmi = ep_token_epkl_total_unc - ep_token_total_unc

            if self.ensembling_mode == "pe":
                next_token_scores = pe_next_token_scores
            elif self.ensembling_mode == "ep":
                next_token_scores = ep_next_token_scores
            else:
                raise NotImplementedError

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            iter_models_scores = []
            for model_scores in models_next_token_logits:
                model_scores_processed = logits_processor(input_ids, model_scores)
                model_scores_processed = logits_warper(input_ids, model_scores)
                iter_models_scores.append(model_scores_processed)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                    models_scores.append(iter_models_scores)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )
                if calculate_entropies:
                    pe_uncertainties["total_uncertainty"].append(pe_token_total_unc)
                    pe_uncertainties["data_uncertainty"].append(pe_token_data_unc)
                    pe_uncertainties["mutual_information"].append(pe_token_mi)
                    pe_uncertainties["epkl_total_uncertainty"].append(
                        pe_token_epkl_total_unc
                    )
                    pe_uncertainties["epkl"].append(pe_token_epkl)
                    pe_uncertainties["rmi"].append(pe_token_rmi)

                    ep_uncertainties["total_uncertainty"].append(ep_token_total_unc)
                    ep_uncertainties["data_uncertainty"].append(ep_token_data_unc)
                    ep_uncertainties["mutual_information"].append(ep_token_mi)
                    ep_uncertainties["epkl_total_uncertainty"].append(
                        ep_token_epkl_total_unc
                    )
                    ep_uncertainties["epkl"].append(ep_token_epkl)
                    ep_uncertainties["rmi"].append(ep_token_rmi)

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                    )
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            token_models_beam_logits = torch.stack(models_next_token_logits)
            token_models_beam_logits = torch.gather(
                token_models_beam_logits,
                -1,
                next_tokens.repeat((num_models), 1).unsqueeze(-1),
            )

            self.models_beam_logits_iter = torch.cat(
                (self.models_beam_logits_iter, token_models_beam_logits), -1
            )
            models_beam_logits = self.models_beam_logits_iter.sum(-1, keepdims=True)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                    .ne(eos_token_id_tensor.unsqueeze(1))
                    .prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return SampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    models_scores=models_scores,
                    models_beam_next_token_logits=models_beam_next_token_logits,
                    pe_uncertainties=pe_uncertainties,
                    ep_uncertainties=ep_uncertainties,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                # TODO: Same as with beam search, need to add GPT support
                return SampleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids


@dataclass
class SampleEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using beam search. Hidden states and attention weights
    of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the encoder_hidden_states
    attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        sequences_scores (`torch.FloatTensor` of shape `(batch_size*num_return_sequences)`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Final beam scores of the generated `sequences`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Beam transition scores for each vocabulary token at each generation step. Beam transition scores consisting
            of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam.
            `(max_length-1,)`-shaped tuple of `torch.FloatTensor` with each tensor of shape `(batch_size*num_beams,
            config.vocab_size)`).
        beam_indices (`tuple(tuple(torch.LongTensor))`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
            `(batch_size*num_return_sequences, max_length-1)`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer of the decoder) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size*num_beams*num_return_sequences, sequence_length, hidden_size)`.
        decoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, num_heads, generated_length,
            sequence_length)`.
        cross_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)`.
    """

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    models_scores: Optional[Tuple[List[torch.FloatTensor]]] = None
    models_beam_next_token_logits: Optional[Tuple[torch.FloatTensor]] = None
    pe_uncertainties: Optional[Dict[str, List[torch.FloatTensor]]] = None
    ep_uncertainties: Optional[Dict[str, List[torch.FloatTensor]]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
