import logging

from lm_polygraph.utils.deberta import Deberta, MultilingualDeberta

log = logging.getLogger("lm_polygraph")


def load_nli_model(
    deberta_path="microsoft/deberta-large-mnli",
    batch_size=10,
    device=None,
):
    if deberta_path.startswith("microsoft"):
        nli_model = Deberta(deberta_path, batch_size, device)
    else:
        nli_model = MultilingualDeberta(deberta_path, batch_size, device)
    log.info(
        f"Initialized {nli_model.deberta_path} on {nli_model.device} with batch_size={nli_model.batch_size}"
    )
    return nli_model