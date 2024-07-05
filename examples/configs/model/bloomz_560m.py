from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL_PATH = "bigscience/bloomz-560m"

def load_model():
    config = get_config()
    model_type = "CausalLM"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True
    )
    model.eval()

    return model, model_type

def load_tokenizer():
    config = get_config()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        padding_side="left",
        add_bos_token=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def get_config():
    return AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
