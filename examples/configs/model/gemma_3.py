from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str, device_map: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device_map,
        attn_implementation="eager",
    )
    if not hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = model.config.text_config.num_hidden_layers
    if not hasattr(model.config, "num_attention_heads"):
        model.config.num_attention_heads = model.config.text_config.num_attention_heads
    model.eval()

    return model


def load_tokenizer(model_path: str, add_bos_token: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        add_bos_token=add_bos_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_path.split("-")[-1] == "it":
        end_token_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        tokenizer.eos_token_id = end_token_id

    return tokenizer
