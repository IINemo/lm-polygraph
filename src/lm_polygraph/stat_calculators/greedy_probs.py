import torch
import numpy as np

from typing import Dict, List, Tuple, Union

from .embeddings import get_embeddings_from_output
from .stat_calculator import StatCalculator
from lm_polygraph.model_adapters import WhiteboxModel, WhiteboxModelvLLM


class GreedyProbsCalculator(StatCalculator):
    """
    For Whitebox model (lm_polygraph.WhiteboxModel), at input texts batch calculates:
    * generation texts
    * tokens of the generation texts
    * probabilities distribution of the generated tokens
    * attention masks across the model (if applicable)
    * embeddings from the model
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """
        return [
            "input_texts",
            "input_tokens",
            "greedy_log_probs",
            "greedy_tokens",
            "greedy_tokens_alternatives",
            "greedy_texts",
            "greedy_log_likelihoods",
            "embeddings",
            "attention_all",
            "tokenizer",
        ], []

    def __init__(
        self,
        output_attentions: bool = True,
        output_hidden_states: bool = False,
        n_alternatives: int = 10,
    ):
        super().__init__()
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.n_alternatives = n_alternatives

    def _preprocess_attention(
        self,
        attentions: torch.Tensor,
        current_idx: int,
        start_idx: int,
        end_idx: int,
        prompt_len: int,
    ) -> torch.Tensor:
        """
        Preprocess attention weights before stacking.

        Parameters:
            attentions (torch.Tensor): Attention weights from a specific layer and head for a current token
            current_idx (int): Current position in the sequence
            start_idx (int): Start index of the generated tokens
            end_idx (int): End index of the generated tokens for current position
            prompt_len (int): Length of the prompt

        Returns:
            torch.Tensor: Preprocessed attention weights
        """
        # Handle attention tensor processing for models with varying attention sizes (e.g. Gemma)
        n_attentions = attentions.shape[-1]

        # Handle empty tensor case
        if attentions.nelement() == 0:
            return torch.zeros(abs(current_idx), device=attentions.device)

        # Handle cases where attention size is smaller than expected
        if n_attentions < end_idx:
            if start_idx < 0:
                return attentions[start_idx:n_attentions]
            return attentions[n_attentions - current_idx : n_attentions]

        # Handle cases where attention spans beyond expected range
        if (n_attentions - current_idx) > end_idx and start_idx < 0:
            return attentions[prompt_len : prompt_len + current_idx]

        # Default case: return attention slice within expected range
        return attentions[start_idx:end_idx]

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: Union[WhiteboxModel, WhiteboxModelvLLM],
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the statistics of probabilities at each token position in the generation.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, can be empty (not used).
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with the following items:
                - 'input_tokens' (List[List[int]]): tokenized input texts,
                - 'greedy_log_probs' (List[List[np.array]]): logarithms of autoregressive
                        probability distributions at each token,
                - 'greedy_texts' (List[str]): model generations corresponding to the inputs,
                - 'greedy_tokens' (List[List[int]]): tokenized model generations,
                - 'attention' (List[List[np.array]]): attention maps at each token, if applicable to the model,
                - 'greedy_log_likelihoods' (List[List[float]]): log-probabilities of the generated tokens.
        """
        # Check if tool calling is enabled and enhance prompts if needed
        import sys
        import logging
        log = logging.getLogger("lm_polygraph")
        
        print(f"[DEBUG GreedyProbsCalculator] ENTRY: Checking for tool calling, has tool_manager attr: {hasattr(model, 'tool_manager')}", file=sys.stderr, flush=True)
        if hasattr(model, 'tool_manager'):
            print(f"[DEBUG GreedyProbsCalculator] tool_manager is not None: {model.tool_manager is not None}", file=sys.stderr, flush=True)
            if model.tool_manager is not None:
                print(f"[DEBUG GreedyProbsCalculator] tool_manager.has_tools(): {model.tool_manager.has_tools()}", file=sys.stderr, flush=True)
        if hasattr(model, 'tool_mandatory'):
            print(f"[DEBUG GreedyProbsCalculator] tool_mandatory: {model.tool_mandatory}", file=sys.stderr, flush=True)
        if hasattr(model, 'tool_name'):
            print(f"[DEBUG GreedyProbsCalculator] tool_name: {model.tool_name}", file=sys.stderr, flush=True)
        
        enhanced_texts = texts
        if hasattr(model, 'tool_manager') and model.tool_manager is not None and model.tool_manager.has_tools():
            print(f"[DEBUG GreedyProbsCalculator] Tools enabled! Enhancing {len(texts)} prompts with tool calling", file=sys.stderr, flush=True)
            log.info(f"GreedyProbsCalculator: Tools enabled, enhancing prompts with tool calling for {len(texts)} inputs")
            
            # For tool calling, we need to get the enhanced prompts (with tool responses)
            # but we still need to call generate() to get scores/logits for uncertainty estimation.
            # We use enhance_prompt_with_tool() to get enhanced prompts, then generate with those.
            # This is the same tool calling logic used by execute_tool_calling_workflow(),
            # but we don't generate the final answer here - we just enhance the prompt.
            enhanced_texts = []
            for i, text in enumerate(texts):
                print(f"[DEBUG GreedyProbsCalculator] Processing input {i+1}/{len(texts)}", file=sys.stderr, flush=True)
                print(f"[DEBUG GreedyProbsCalculator] Checking tool_mandatory: hasattr={hasattr(model, 'tool_mandatory')}, value={getattr(model, 'tool_mandatory', None)}", file=sys.stderr, flush=True)
                if hasattr(model, 'tool_mandatory') and model.tool_mandatory:
                    # Mandatory tool usage - enhance prompt with tool results
                    # Use the shared enhance_prompt_with_tool function (same logic as execute_tool_calling_workflow)
                    from lm_polygraph.utils.tool_calling import enhance_prompt_with_tool
                    
                    print(f"[DEBUG GreedyProbsCalculator] Enhancing prompt {i+1}/{len(texts)} with tool calling", file=sys.stderr, flush=True)
                    
                    try:
                        enhanced_prompt, tool_name_used, tool_was_used = enhance_prompt_with_tool(
                            model=model,
                            question=text,
                            tool_manager=model.tool_manager,
                            mandatory=True,
                            tool_name=model.tool_name,
                            max_new_tokens=max_new_tokens,
                            use_tools=False  # Disable tool calling in nested calls
                        )
                        enhanced_texts.append(enhanced_prompt)
                        print(f"[DEBUG GreedyProbsCalculator] Enhanced prompt created for input {i+1} (tool_was_used={tool_was_used}, tool_name={tool_name_used})", file=sys.stderr, flush=True)
                        print(f"[DEBUG GreedyProbsCalculator] Enhanced texts: {enhanced_texts}")
                    except Exception as e:
                        print(f"[DEBUG GreedyProbsCalculator] ERROR in enhance_prompt_with_tool: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                        import traceback
                        traceback.print_exc(file=sys.stderr)
                        # Fallback to original text on error
                        log.warning(f"Tool calling failed for input {i+1}, using original text: {e}")
                        enhanced_texts.append(text)
                else:
                    # Optional tool usage - for now, use original text
                    # TODO: Implement optional tool calling logic
                    enhanced_texts.append(text)
        else:
            enhanced_texts = texts
        
        # Process enhanced texts in batches (now that we're using only top 1 document, prompts are shorter)
        # This applies to both tool-enhanced and non-tool prompts
        print(f"[DEBUG GreedyProbsCalculator] Processing {len(enhanced_texts)} prompts in batch", file=sys.stderr, flush=True)
        if hasattr(model, 'tool_manager') and model.tool_manager is not None and model.tool_manager.has_tools():
            print(f"[DEBUG GreedyProbsCalculator] Tool-enhanced prompts will be processed in batch (using top 1 document)", file=sys.stderr, flush=True)
        
        # Batch processing for all prompts (tool-enhanced or not)
        batch: Dict[str, torch.Tensor] = model.tokenize(enhanced_texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}
        
        print(f"[DEBUG GreedyProbsCalculator] Batch tokenized: input_ids shape={batch['input_ids'].shape}", file=sys.stderr, flush=True)
        
        with torch.no_grad():
            out = model.generate(
                **batch,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
                min_new_tokens=2,
                output_attentions=self.output_attentions,
                output_hidden_states=self.output_hidden_states,
                num_return_sequences=1,
                suppress_tokens=(
                    []
                    if model.generation_parameters.allow_newlines
                    else [
                        t
                        for t in range(len(model.tokenizer))
                        if "\n" in model.tokenizer.decode([t])
                    ]
                ),
            )
        
        batch_for_tokens = batch
        logits = torch.stack(out.scores, dim=1)
        if model.model_type == "vLLMCausalLM":
            logits = logits.transpose(1, 0)
        sequences = out.sequences
        if self.output_attentions:
            attentions = out.attentions
        if self.output_hidden_states:
            embeddings_encoder, embeddings_decoder = get_embeddings_from_output(
                out, batch, model.model_type
            )
            if embeddings_decoder.dtype == torch.bfloat16:
                embeddings_decoder = embeddings_decoder.to(
                    torch.float16
                )  # numpy does not support bfloat16

        cut_logits = []
        cut_sequences = []
        cut_texts = []
        cut_alternatives = []
        # Use enhanced_texts length for iteration (should be same as texts, but enhanced_texts is what we generated with)
        for i in range(len(enhanced_texts)):
            if model.model_type == "CausalLM":
                idx = batch_for_tokens["input_ids"].shape[1]
                seq = sequences[i, idx:].cpu()
            elif model.model_type == "vLLMCausalLM":
                seq = sequences[i].cpu()
            else:
                seq = sequences[i, 1:].cpu()
            length, text_length = len(seq), len(seq)
            for j in range(len(seq)):
                if seq[j] == model.tokenizer.eos_token_id:
                    length = j + 1
                    text_length = j
                    break
            cut_sequences.append(seq[:length].tolist())
            cut_texts.append(model.tokenizer.decode(seq[:text_length]))
            cut_logits.append(logits[i, :length, :].cpu().numpy())
            cut_alternatives.append([[] for _ in range(length)])
            for j in range(length):
                lt = logits[i, j, :].cpu().numpy()
                best_tokens = np.argpartition(lt, -self.n_alternatives)
                ln = len(best_tokens)
                best_tokens = best_tokens[ln - self.n_alternatives : ln]
                for t in best_tokens:
                    cut_alternatives[i][j].append((t.item(), lt[t].item()))
                cut_alternatives[i][j].sort(
                    key=lambda x: x[0] == cut_sequences[i][j],
                    reverse=True,
                )

        ll = []
        for i in range(len(enhanced_texts)):
            log_probs = cut_logits[i]
            tokens = cut_sequences[i]
            assert len(tokens) == len(log_probs)
            ll.append([log_probs[j, tokens[j]] for j in range(len(log_probs))])

        attention_all = []
        if self.output_attentions and (model.model_type != "vLLMCausalLM"):
            prompt_len = batch_for_tokens["input_ids"].shape[1]
            for i in range(len(enhanced_texts)):
                c = len(cut_sequences[i])
                attn_mask = np.zeros(
                    shape=(
                        model.model.config.num_attention_heads
                        * model.model.config.num_hidden_layers,
                        c,
                        c,
                    )
                )
                for j in range(1, c):
                    # Get attention dimensions
                    current_attention_len = attentions[j][i].shape[-1]

                    # Default case: use relative indexing from end
                    start_idx = -j
                    end_idx = current_attention_len

                    # Special case for models like Gemma that maintain consistent attention lengths
                    if attentions[0][i].shape[-1] == current_attention_len:
                        start_idx = prompt_len
                        end_idx = prompt_len + j

                    stacked_attention = torch.vstack(
                        [
                            self._preprocess_attention(
                                attentions[j][layer][i][head][0],
                                j,
                                start_idx,
                                end_idx,
                                prompt_len,
                            )
                            for layer in range(len(attentions[j]))
                            for head in range(len(attentions[j][layer][i]))
                        ]
                    )
                    if stacked_attention.dtype == torch.bfloat16:
                        stacked_attention = stacked_attention.to(
                            torch.float16
                        )  # numpy does not support bfloat16

                    attn_mask[:, j, :j] = stacked_attention.cpu().numpy()
                attention_all.append(attn_mask)

        if not self.output_hidden_states:
            embeddings_dict = {}
        elif model.model_type == "CausalLM":
            embeddings_dict = {
                "embeddings_decoder": embeddings_decoder.cpu().detach().numpy(),
            }
        elif model.model_type == "Seq2SeqLM":
            embeddings_dict = {
                "embeddings_encoder": embeddings_encoder.cpu().detach().numpy(),
                "embeddings_decoder": embeddings_decoder.cpu().detach().numpy(),
            }
        else:
            raise NotImplementedError
        
        # Clear CUDA cache after batch processing (optional, but good practice)
        torch.cuda.empty_cache()

        # Create result dictionary
        result_dict = {
            "input_tokens": batch_for_tokens["input_ids"].to("cpu").tolist(),
            "greedy_log_probs": cut_logits,
            "greedy_tokens": cut_sequences,
            "greedy_tokens_alternatives": cut_alternatives,
            "greedy_texts": cut_texts,
            "greedy_log_likelihoods": ll,
        }
        result_dict.update(embeddings_dict)
        if self.output_attentions:
            result_dict.update({"attention_all": attention_all})
            result_dict.update({"tokenizer": model.tokenizer})
        return result_dict
