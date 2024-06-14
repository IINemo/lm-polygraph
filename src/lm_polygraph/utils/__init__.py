import inspect

from .model import WhiteboxModel, BlackboxModel
from .manager import UEManager, estimate_uncertainty
from .dataset import Dataset


def polygraph_module_init(func):
    def wrapper(*args, **kwargs):
        breakpoint()
        is_method = inspect.ismethod(func)
        return func(*args, **kwargs)

    return wrapper
