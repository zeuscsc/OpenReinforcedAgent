from torch.nn import functional as F
import torch

def pad_and_truncate(inputs: torch.Tensor, max_length, value=-100):
    # truncate
    if inputs.shape[-1] >= max_length:
        inputs = inputs[..., :max_length]
    # pad
    else:
        inputs = F.pad(inputs, (max_length - inputs.shape[-1], 0), value=value)

    return inputs