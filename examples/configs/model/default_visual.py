from transformers import AutoModelForVision2Seq, AutoProcessor


def load_model(model_path: str, device_map: str):
    model = AutoModelForVision2Seq.from_pretrained(
        model_path, trust_remote_code=True, device_map=device_map
    )
    model.eval()

    return model


def load_tokenizer(model_path: str):
    processor_visual = AutoProcessor.from_pretrained(
        model_path,
        padding_side="left",
        add_bos_token=True,
    )
    if processor_visual.tokenizer.pad_token is None:
        processor_visual.tokenizer.pad_token = processor_visual.tokenizer.eos_token

    return processor_visual
