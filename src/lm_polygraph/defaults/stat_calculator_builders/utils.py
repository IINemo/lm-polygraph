import logging
from typing import Dict, Optional

from lm_polygraph.utils.deberta import Deberta, MultilingualDeberta
from lm_polygraph.utils.sentence_embedder import SentenceEmbedder

log = logging.getLogger("lm_polygraph")


def load_nli_model(
    deberta_path="microsoft/deberta-large-mnli",
    batch_size=10,
    device=None,
    hf_cache: str = None,
):
    if deberta_path.startswith("microsoft"):
        nli_model = Deberta(deberta_path, batch_size, device=device, hf_cache=hf_cache)
    else:
        nli_model = MultilingualDeberta(
            deberta_path, batch_size, device=device, hf_cache=hf_cache
        )
    log.info(
        f"Initialized {nli_model.deberta_path} on {nli_model.device} with batch_size={nli_model.batch_size}"
    )
    return nli_model


def load_sentence_embedding_model(
    model_name: str = "all-mpnet-base-v2",
    device: Optional[str] = None,
    cache_folder: Optional[str] = None,
    encoding_arguments: Optional[Dict] = None,
) -> SentenceEmbedder:
    model = SentenceEmbedder(
        model_name,
        device=device,
        cache_folder=cache_folder,
        encoding_arguments=encoding_arguments,
    )
    log.info(f"Initialized SentenceEmbedder({model_name}) on {model.device}")
    return model
