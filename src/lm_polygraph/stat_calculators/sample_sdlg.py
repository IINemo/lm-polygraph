import numpy as np
import torch
from collections import defaultdict

from typing import Dict, List, Tuple

from .stat_calculator import StatCalculator
from lm_polygraph.model_adapters import WhiteboxModel, BlackboxModel, WhiteboxModelvLLM


class Args:
    def __init__(self):
        self.num_total_generations = 10
        self.eos_token_ids = None
        self.invalid_ids = []

        # Default parameters taken from Aichberger et al. (2024)
        self.num_beams_sdlg = 5
        self.num_return_sequences_sdlg = 1
        self.do_sample_sdlg = False
        self.temperature_sdlg = 1.0
        self.top_p_sdlg = 1
        self.token_prob_threshold = 0.001
        self.alphas = (1/3, 1/3, 1/3)  # weighting of attribution, substitution, and importance scores


@torch.no_grad()
def remove_invalid_ids(generation, 
                       invalid_ids):
    for invalid in invalid_ids:
        if invalid in generation:
            generation = generation[:torch.where(generation == invalid)[0][0]]
    return generation


@torch.no_grad()
def clean_generation(generation):
    strings_to_filter_on = ['A:', 'A;', 'answer:',  'Answer:', 'Answers:', 'answers:', 'ANSWER:',
                            'Q:', 'Q;', 'question:', 'Question:', 'Questions:', 'questions:', 'QUESTION:']

    for stop_word in strings_to_filter_on:
        stop_word_index = generation.find(stop_word)
        if stop_word_index != -1:
            generation = generation[:stop_word_index]

    generation = generation.strip()

    return generation


@torch.no_grad()
def generate_text(args, 
                  model, 
                  tokenizer, 
                  input_ids, 
                  len_prompt, 
                  decoding_method, 
                  device):
    input_ids = input_ids.to(device).reshape(1, -1) if args.dataset == 'trivia_qa' else input_ids.to(device)

    generation_ids = model.generate(input_ids,
                                    num_beams=args.num_beams_sdlg * args.num_return_sequences_sdlg,
                                    num_return_sequences=args.num_return_sequences_sdlg,
                                    do_sample=args.do_sample_sdlg,
                                    temperature=args.temperature_sdlg,
                                    top_p=args.top_p_sdlg,
                                    max_length=len_prompt + args.max_length_of_generated_sequence,
                                    eos_token_id=args.eos_token_ids,)

    generation_ids = generation_ids.to('cpu')


@torch.no_grad()
def compute_likelihood(prompt, 
                       generation, 
                       model, 
                       device, 
                       compute_cleaned=False, 
                       store_logits=True):

    # Note: This computation of NLL follows the impementation of Kuhn et al. (2023)
    list_average_neg_log_likelihoods, list_neg_log_likelihood = [], []
    list_cleaned_average_neg_log_likelihood, list_cleaned_neg_log_likelihood = [], []
    list_generation_logits, list_cleaned_generation_logits = [], []

    # iterate over all generations -> "generation_ids" is list of generations
    for i in range(len(generation['generation_ids'])): 

        generation_ids = generation['generation_ids'][i]

        generation_input = torch.hstack([prompt, generation_ids]).to(device)

        target_ids = generation_input.clone()
        target_ids[:len(prompt)] = -100
        model_output = model(torch.reshape(generation_input, (1, -1)), labels=target_ids)
        average_neg_log_likelihood = model_output['loss'].item()
        neg_log_likelihood = average_neg_log_likelihood * (len(generation_ids))

        list_average_neg_log_likelihoods.append(average_neg_log_likelihood)
        list_neg_log_likelihood.append(neg_log_likelihood)

        # compute logits
        if store_logits:
            generation_logits = model_output["logits"][0, len(prompt)-1:-1, :].to('cpu') 
            # shift by 1 since token probs at last token of prompt already belong to first token of generation
            list_generation_logits.append(generation_logits)
            assert generation_logits.shape[0] == generation_ids.shape[0]

        if compute_cleaned:

            cleaned_generation_ids = generation['cleaned_generation_ids'][i]

            if torch.equal(cleaned_generation_ids, generation_ids) or \
                generation['cleaned_generation_text'][i] == generation['generation_text'][i]:
                cleaned_average_neg_log_likelihood = average_neg_log_likelihood
                cleaned_neg_log_likelihood = neg_log_likelihood
                if store_logits:
                    cleaned_generation_logits = generation_logits
            elif generation['cleaned_generation_text'][i] == '':
                # Note: setting nll to ngative infinity (zero likelihood) if cleaned generation is empty
                cleaned_average_neg_log_likelihood = float('-inf')
                cleaned_neg_log_likelihood = float('-inf')
                if store_logits:
                    cleaned_generation_logits = []
            else:
                # Note: computation of NNL follows tutorial: https://huggingface.co/docs/transformers/perplexity
                generation_input = torch.hstack([prompt, cleaned_generation_ids]).to(device)
                target_ids = generation_input.clone()
                target_ids[:len(prompt)] = -100
                model_output = model(torch.reshape(generation_input, (1, -1)), labels=target_ids)
                cleaned_average_neg_log_likelihood = model_output['loss'].item()
                cleaned_neg_log_likelihood = cleaned_average_neg_log_likelihood * (len(cleaned_generation_ids))

                # compute logits
                if store_logits:
                    cleaned_generation_logits = model_output["logits"][0, len(prompt)-1:-1, :].to('cpu')

            if store_logits:
                list_cleaned_generation_logits.append(cleaned_generation_logits)
            list_cleaned_average_neg_log_likelihood.append(cleaned_average_neg_log_likelihood)
            list_cleaned_neg_log_likelihood.append(cleaned_neg_log_likelihood)

    return {
        'average_neg_log_likelihood': list_average_neg_log_likelihoods,
        'neg_log_likelihood': list_neg_log_likelihood,
        'generation_logits': list_generation_logits,

        'cleaned_average_neg_log_likelihood': list_cleaned_average_neg_log_likelihood,
        'cleaned_neg_log_likelihood': list_cleaned_neg_log_likelihood,
        'cleaned_generation_logits': list_cleaned_generation_logits,
    }


latest_grads = [None]
latest_embeddings = [None]
def forward_hook(module, input, output):
    latest_embeddings[0] = output.detach()
    output.register_hook(lambda grad: latest_grads.__setitem__(0, grad))


def get_word_indices(text_ids, tokenizer):
    all_word_indices = []
    word_indices = [0]
    lookback = 1
    for t in range(1, len(text_ids)):
        if len(tokenizer.decode(text_ids[t]).strip()) == 0:
            word_indices.append(t)
            lookback += 1 # increase lookback in case token is empty
        elif len(tokenizer.decode(text_ids[t-lookback:t+1]).split()) == 1:
            word_indices.append(t)
            lookback = 1
        else:
            all_word_indices.append(torch.tensor(word_indices))
            word_indices = [t]
            lookback = 1
    all_word_indices.append(torch.tensor(word_indices))

    return all_word_indices


def rank_tensor(t, descending=True):
    t = torch.tensor(t)
    sorted_vals, sorted_idx = torch.sort(t, descending=descending)
    unique_vals, inverse_indices = torch.unique_consecutive(sorted_vals, return_inverse=True)
    ranks = inverse_indices + 1
    corrected_ranks = torch.empty_like(sorted_idx)
    corrected_ranks[sorted_idx] = ranks

    return corrected_ranks


# algorithm 2 (according to paper)
def compute_token_score_ranking(deberta_model, 
                                deberta_tokenizer, 
                                device_deberta, 
                                question, 
                                initial_generation_ids, 
                                initial_generation_text, 
                                additional_generated_text, 
                                generation_logits, 
                                deberta_embeddings, 
                                args):

    # Define a hook to store the gradients of the token embeddings
    handle = deberta_model.deberta.embeddings.word_embeddings.register_forward_hook(forward_hook)

    ce_loss_fn = torch.nn.CrossEntropyLoss()

    encoded_question = deberta_tokenizer.encode(question, padding=True, return_tensors='pt').squeeze()[:-1] # remove SEP token (last)
    encoded_answer = deberta_tokenizer.encode(' ' + initial_generation_text, padding=True, return_tensors='pt').squeeze()[1:-1] # remove CLS token (first) and SEP (last) tokens
    all_word_indices = get_word_indices(text_ids=encoded_answer, tokenizer=deberta_tokenizer)

    qa_initial = question + ' ' + initial_generation_text
    input_sequence = qa_initial + ' [SEP] ' + qa_initial
    model_input = [input_sequence]

    for additional_a in additional_generated_text:
        input_sequence = qa_initial + ' [SEP] ' + question + ' ' + additional_a
        model_input.append(input_sequence)

    encoded_input = deberta_tokenizer(model_input, return_tensors='pt', padding=True).to(device_deberta)
    deberta_model.zero_grad()
    prediction = deberta_model(**encoded_input)['logits']


    target = torch.tensor([0] + [0] * len(additional_generated_text)).to(device_deberta)
    loss = ce_loss_fn(prediction, target)
    loss.backward()
    
    assert encoded_input["input_ids"].shape[1] == latest_grads[0].shape[1] == latest_embeddings[0].shape[1]
    
    for i in range(len(additional_generated_text) + 1):
        if encoded_question.tolist() != encoded_input["input_ids"][i, :len(encoded_question)].tolist():
            print(f"Error: {encoded_question.tolist()} vs. {encoded_input['input_ids'][i, :len(encoded_question)].tolist()}")
            return False
    for word, word_token_indices in zip(initial_generation_text.split(), all_word_indices):
        if word.strip() != deberta_tokenizer.decode(encoded_answer[word_token_indices]).strip():
            print(f'Error: words do not match ({word.strip()} vs. {deberta_tokenizer.decode(encoded_answer[all_word_indices]).strip()})')
            return False
    if len(encoded_answer) != initial_generation_ids.shape[0]:
        # Example: encoded_answer: [' the', ' _', 'Sel', 'ache', '_.'] vs. initial_generation_ids: [' the', ' _', 'Sel', 'ache', '_', '.']
        print(f"Error: {[deberta_tokenizer.decode(e) for e in encoded_answer]} vs. {[deberta_tokenizer.decode(e) for e in initial_generation_ids]}")
        return False
            
    handle.remove()
    token_info = {}
    with torch.no_grad():

        consider_gradients_from_both_sides = False  # set accordingly (similar empirical performance, thus set to False for more efficiency)

        qa1_gradients = latest_grads[0][:, len(encoded_question):len(encoded_question)+len(encoded_answer), :]
        qa1_embeddings = latest_embeddings[0][:, len(encoded_question):len(encoded_question)+len(encoded_answer), :]
        qa1_attributions = qa1_gradients * qa1_embeddings
        assert qa1_gradients.shape == qa1_embeddings.shape

        if consider_gradients_from_both_sides:
            qa2_gradients = latest_grads[0][0, -(len(encoded_answer)+1):-1, :]
            qa2_embeddings = latest_embeddings[0][0, -(len(encoded_answer)+1):-1, :]
            qa2_attributions = qa2_gradients * qa2_embeddings
            assert qa1_gradients.shape == qa2_gradients.shape == qa1_embeddings.shape == qa2_embeddings.shape
        
            token_attributions = torch.abs(qa1_attributions + qa2_attributions) / 2
            # token_attributions.shape = [num_tokens, deberta_embedding_dim]
        else:
            token_attributions = torch.abs(qa1_attributions)

        # (1) calculate attribution scores
        all_word_gradient_magnitudes = []
        for i in range(len(additional_generated_text) + 1):
            word_attributions = torch.vstack([token_attributions[i, word_token_indices, :].mean(dim=0) for word_token_indices in all_word_indices])
            assert word_attributions.shape[0] == len(initial_generation_text.split())
            
            all_word_gradient_magnitudes.append(torch.norm(word_attributions, dim=-1).tolist())
            # word_gradient_magnitude.shape = [num_words]

        word_gradient_magnitude = torch.tensor(all_word_gradient_magnitudes).mean(dim=0)
        assert word_gradient_magnitude.shape[0] == len(all_word_indices)

        # (2+3) calculate substitution and importance scores
        if consider_gradients_from_both_sides:
            deberta_gradients = (qa1_gradients + qa2_gradients) / 2
        else:
            deberta_gradients = qa1_gradients

        for initial_gen_word_idx, word_token_indices in enumerate(all_word_indices):

            initial_gen_token_idx = word_token_indices[0] # index at generation level
            initial_voc_token_idx = initial_generation_ids[word_token_indices[0]]
            if args.token_prob_threshold is None:
                other_voc_token_indices = torch.tensor(range(len(generation_logits[initial_gen_token_idx])))
            else:
                other_voc_token_indices = torch.where(generation_logits[initial_gen_token_idx] > args.token_prob_threshold)[0] # indices at vocabulary level

            delta_embeddings = deberta_embeddings[initial_voc_token_idx] - deberta_embeddings[other_voc_token_indices]
            # deberta_embeddings.shape = torch.Size([vocab_size, deberta_embedding_dim])
            # delta_embeddings.shape = torch.Size([num_tokens, deberta_embedding_dim])

            all_substitution_scores = []
            for i in range(len(additional_generated_text) + 1):
                all_substitution_scores.append(torch.nn.functional.cosine_similarity(delta_embeddings, deberta_gradients[i, initial_gen_token_idx].unsqueeze(0)).tolist())

            all_substitution_scores = torch.tensor(all_substitution_scores).mean(dim=0)
            assert all_substitution_scores.shape[0] == len(other_voc_token_indices)

            for new_token_idx, substitution_score in zip(other_voc_token_indices, all_substitution_scores):

                attribution_score = word_gradient_magnitude[initial_gen_word_idx]

                importance_score = generation_logits[initial_gen_token_idx][new_token_idx]

                if new_token_idx != initial_voc_token_idx and new_token_idx not in args.invalid_ids:
                    token_info[(initial_gen_word_idx, initial_gen_token_idx.item(), new_token_idx.item())] = (attribution_score.item(), substitution_score.item(), importance_score.item())
                    # keys:
                    # (1) initial_gen_word_idx: index of word in original generation
                    # (2) initial_gen_token_idx: index of token in original generation that is replaced
                    # (3) new_token_idx: index of token in vocabulary that is used as replacement
                    # values:
                    # (1) attribution_score: gradient magnitude on a word level -> bigger is better (higher gradient)
                    # (2) substitution_score: gradient direction on token level -> bigger is better (same direction)
                    # (3) importance_score: probability on token level -> bigger is better (higher probability)

    # sort token_info
    ranking_attribution_score = rank_tensor([v[0] for v in token_info.values()], descending=True)
    ranking_substitution_score = rank_tensor([v[1] for v in token_info.values()], descending=True)
    ranking_importance_score = rank_tensor([v[2] for v in token_info.values()], descending=True)

    sorted_indices = torch.argsort(args.alphas[0] * ranking_attribution_score + 
                                   args.alphas[1] * ranking_substitution_score + 
                                   args.alphas[2] * ranking_importance_score, 
                                   descending=False)

    return sorted_indices, token_info


# algorithm 1 (according to paper)
def generate_semantically_diverse_output_sequences(results_dict, 
                                                   deberta_model, 
                                                   deberta_tokenizer, 
                                                   device_deberta, 
                                                   model, 
                                                   tokenizer, 
                                                   device_llm,
                                                   input_ids, 
                                                   prompt, 
                                                   question, 
                                                   initial_generation, 
                                                   initial_likelihood, 
                                                   args):

    deberta_embeddings = deberta_model.deberta.embeddings.word_embeddings(
        torch.tensor([list(range(0, deberta_tokenizer.vocab_size))]).to(device_deberta)
    ).squeeze().detach()

    initial_generation_text = initial_generation['generation_text'][0]
    initial_generation_ids = initial_generation['generation_ids'][0]

    assert len(initial_likelihood["generation_logits"]) == 1
    generation_logits = initial_likelihood["generation_logits"][0].to(dtype=torch.float32)
    generation_logits = torch.nn.functional.softmax(generation_logits, dim=-1) 
    generation_logits += 1e-9

    assert generation_logits.shape[0] == initial_generation_ids.shape[0]
    # generation_logits.shape = [num_tokens, opt_vocab_size]

    single_word = False
    if initial_generation_ids.shape[0] == 0 or len(initial_generation_text.split()) == 0:
        print("Warning: initial generation is empty!")
        return results_dict
        
    if len(initial_generation_text.split()) == 1:
        single_word = True
        token_info = {}

        if args.token_prob_threshold is None:
            other_voc_token_indices = torch.tensor((range(len(generation_logits[0]))))
        else:
            other_voc_token_indices = torch.where(generation_logits[0] > args.token_prob_threshold)[0] # indices at vocabulary level

        for new_token_idx in other_voc_token_indices:
            if new_token_idx != initial_generation_ids[0] and new_token_idx not in args.invalid_ids:
                importance_score = generation_logits[0][new_token_idx]
                token_info[(0, 0, new_token_idx.item())] = (0, 0, importance_score.item())

        sorted_indices = torch.argsort(rank_tensor([v[2] for v in token_info.values()], descending=True), descending=False)

    additional_generated_text = []
    num_added_gens = 0

    if not single_word:
        sorted_indices, token_info = compute_token_score_ranking(deberta_model, 
                                                                 deberta_tokenizer, 
                                                                 device_deberta, 
                                                                 question, 
                                                                 initial_generation_ids, 
                                                                 initial_generation_text, 
                                                                 additional_generated_text, 
                                                                 generation_logits, 
                                                                 deberta_embeddings, 
                                                                 args)
        # return empty dict when error occured
        if not isinstance(sorted_indices, torch.Tensor):
            results_dict['sdlg']['generations'] = []
            results_dict['sdlg']['likelihoods'] = []
            return results_dict

    token_info_list = list(token_info.keys())
    with torch.no_grad():
        # iterate over words that should be changed
        for i, s in enumerate(sorted_indices):

            initial_gen_word_idx, initial_gen_token_idx, new_token_idx = token_info_list[s]

            if initial_gen_token_idx > 0:

                new_input_ids = initial_generation_ids[:initial_gen_token_idx]
                token_to_replace_id = initial_generation_ids[initial_gen_token_idx]

                all_input_ids = torch.hstack([input_ids.to(device_llm), new_input_ids.unsqueeze(0).to(device_llm)])

            else:
                initial_gen_token_idx = 0
                all_input_ids = input_ids.to(device_llm)
                token_to_replace_id = initial_generation_ids[0]

            token_to_replace_text = tokenizer.decode(token_to_replace_id)

            new_token_text = tokenizer.decode([new_token_idx])
            importance_score = generation_logits[initial_gen_token_idx][new_token_idx].item()

            # skip if token id is invalid or token id is the same
            if new_token_idx in args.invalid_ids or new_token_idx == token_to_replace_id.item():
                continue

            # check if added token is eos token
            if new_token_idx != args.eos_token_ids:

                final_input_ids = torch.hstack([all_input_ids, torch.tensor(new_token_idx).unsqueeze(0).unsqueeze(0).to(device_llm)])
                alternative_generation = generate_text(args=args, 
                                                    model=model, 
                                                    tokenizer=tokenizer, 
                                                    input_ids=final_input_ids, 
                                                    len_prompt=len(prompt), 
                                                    decoding_method="sdlg", 
                                                    device=device_llm)
            else:
                if initial_gen_word_idx == 0:
                    continue # skip if first predicted token is eos token 
                generation_to_add = torch.hstack([new_input_ids[0], torch.tensor(new_token_idx)])
                generation_text = tokenizer.decode(generation_to_add, skip_special_tokens=True).strip()
                cleaned_generation_text = clean_generation(generation_text)
                alternative_generation = {
                    'generation_ids': [generation_to_add],
                    'generation_text': [generation_text],

                    'cleaned_generation_ids': [generation_to_add if generation_text == cleaned_generation_text else tokenizer.encode(cleaned_generation_text, add_special_tokens=False, return_tensors='pt')[0]],
                    'cleaned_generation_text': [cleaned_generation_text],
                    'logits': None,
                }

            # skip empty generations
            if len(alternative_generation['generation_text'][0].strip()) == 0:
                continue

            # compute likelihood
            alternative_likelihoods = compute_likelihood(prompt, alternative_generation, model, device_llm, compute_cleaned=False, store_logits=True)

            # log additional information of alternative generation
            alternative_generation['word_idx'] = initial_gen_word_idx
            alternative_generation['token_idx'] = new_token_idx
            alternative_generation['initial_gen_token_idx'] = initial_gen_token_idx
            alternative_generation['token_text'] = new_token_text
            alternative_generation['token_likelihood'] = importance_score
            alternative_generation['num_computed_gen'] = i + 1

            alternative_generation['initial_generation_ids'] = initial_generation_ids
            alternative_generation['initial_generation_text'] = initial_generation_text

            # store alternative generation
            results_dict['sdlg']['generations'].append(alternative_generation)
            results_dict['sdlg']['likelihoods'].append(alternative_likelihoods)
            num_added_gens += 1

            additional_generated_text.append(alternative_generation['generation_text'][0])

            # breaking condition
            if num_added_gens >= args.num_total_generations:
                return results_dict

    return results_dict


class SamplingGenerationSDLGCalculator(StatCalculator):
    """
        This class is an interface to the algorithms 1 & 2 from the paper
        "Semantically Diverse Language Generation..." by Aichberger et al. (2024).
        https://arxiv.org/pdf/2406.04306
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """

        return [
            "sdlg_sample_likelihoods",
            "sdlg_sample_tokens",
            "sdlg_sample_texts",
        ], [
            "greedy_texts",
            "greedy_tokens",
            "greedy_log_likelihoods",
        ]

    def __init__(
        self,
        nli_model,
        samples_n: int = 10,
        alphas: List[float] = [0.33, 0.33, 0.33],
        token_prob_threshold: float = 0.001,
        invalid_token_ids: []
    ):
        self.nli_model = nli_model
        self.samples_n = samples_n
        self.alphas = alphas
        self.token_prob_threshold = token_prob_threshold
        self.invalid_ids = invalid_token_ids

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: Model,
        max_new_tokens: int = 100,
    ):
        sdlg_sample_texts = []
        sdlg_sample_tokens = []
        sdlg_sample_likelihoods = []

        args = Args(
            num_total_generations=self.samples_n,
            token_prob_threshold=self.token_prob_threshold,
            alphas=self.alphas,
            invalid_ids=self.invalid_ids
            eos_token_ids=model.eos_token_id if isinstance(model, (WhiteboxModel, WhiteboxModelvLLM)) else None,
        )

        for text in texts:
            results_dict = defaultdict(dict)

            batch: Dict[str, torch.Tensor] = model.tokenize([text])
            batch = {k: v.to(model.device()) for k, v in batch.items()}

            result = generate_semantically_diverse_output_sequences(
                results_dict=results_dict,
                deberta_model=self.nli_model.deberta,
                deberta_tokenizer=self.nli_model.deberta_tokenizer,
                device_deberta=self.nli_model.device,
                model=model.model,
                tokenizer=model.tokenizer,
                device_llm=model.device,
                input_ids=model.tokenizer(text, return_tensors='pt').input_ids.squeeze(0),
                prompt=text,
                question=text,
                initial_generation=dependencies['greedy_texts'][0],
                initial_likelihood=np.sum(dependencies['greedy_log_likelihoods'][0]),
                args=args
            )

            sdlg_sample_texts.extend(result['sdlg']['generations'])
            sdlg_sample_tokens.extend([gen['generation_ids'] for gen in result['sdlg']['generations']])
            sdlg_sample_likelihoods.extend([gen['neg_log_likelihood'] for gen in result['sdlg']['likelihoods']])

        return {
            "sdlg_sample_texts": sdlg_sample_texts,
            "sdlg_sample_tokens": sdlg_sample_tokens,
            "sdlg_sample_likelihoods": sdlg_sample_likelihoods,
        }
