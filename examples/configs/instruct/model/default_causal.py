from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str, device_map: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, device_map=device_map
    )
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

    return tokenizer
