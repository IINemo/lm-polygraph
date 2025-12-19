from .model import WhiteboxModel, BlackboxModel
from .manager import UEManager
from .estimate_uncertainty import estimate_uncertainty
from .dataset import Dataset

# Optional vLLM support (requires vllm package)
try:
    from .vllm_with_uncertainty import VLLMWithUncertainty
except ImportError:
    VLLMWithUncertainty = None
