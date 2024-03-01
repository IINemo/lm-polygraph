import torch
from torch import nn, Tensor
import types


class ConsistentDropout(nn.Dropout):
    def forward_share_across_tokens(self, input: Tensor) -> Tensor:
        shape = input.shape
        mask_shape = shape[2:]
        mask = torch.empty(
            mask_shape, dtype=torch.bool, device=input.device
        ).bernoulli_(self.p)
        mask = mask.repeat(list(shape[:2]) + [1] * len(mask_shape))
        return input.masked_fill(mask, 0) / (1 - self.p)

    def forward(self, input: Tensor) -> Tensor:
        shape = input.shape
        mask_shape = shape[1:]
        mask = torch.empty(
            mask_shape, dtype=torch.bool, device=input.device
        ).bernoulli_(self.p)
        mask = mask.repeat(list(shape[:1]) + [1] * len(mask_shape))
        return input.masked_fill(mask, 0) / (1 - self.p)

    def identity(self, input: Tensor) -> Tensor:
        return input


def functional_dropout_share(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    if not training:
        return input
    shape = input.shape
    mask_shape = shape[2:]
    mask = torch.empty(mask_shape, dtype=torch.bool, device=input.device).bernoulli_(p)
    mask = mask.repeat(list(shape[:2]) + [1] * len(mask_shape))
    return input.masked_fill(mask, 0) / (1 - p)


def functional_dropout(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    if not training:
        return input
    shape = input.shape
    mask_shape = shape[1:]
    mask = torch.empty(mask_shape, dtype=torch.bool, device=input.device).bernoulli_(p)
    mask = mask.repeat(list(shape[:1]) + [1] * len(mask_shape))
    return input.masked_fill(mask, 0) / (1 - p)


def replace_with_identity(module):
    children = module.children()
    for c_mod in children:
        if type(c_mod).__name__ == "Dropout":
            method = getattr(ConsistentDropout, "identity")
            setattr(c_mod, "forward", types.MethodType(method, c_mod))
        else:
            replace_with_identity(c_mod)


def replace_dropout(model_name, module, p=0.1, share_across_tokens=True):
    if "bart" in model_name.lower():
        if share_across_tokens:
            torch.nn.functional.dropout = functional_dropout_share
        else:
            torch.nn.functional.dropout = functional_dropout
    else:
        children = module.children()
        for c_mod in children:
            if type(c_mod).__name__ == "Dropout":
                if share_across_tokens:
                    method = getattr(ConsistentDropout, "forward_share_across_tokens")
                else:
                    method = getattr(ConsistentDropout, "forward")
                setattr(c_mod, "forward", types.MethodType(method, c_mod))
                c_mod.p = p
            else:
                replace_dropout(
                    model_name, c_mod, p=p, share_across_tokens=share_across_tokens
                )
