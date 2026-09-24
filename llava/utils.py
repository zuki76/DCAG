def disable_torch_init():
    """Skip default Linear and LayerNorm initialization before loading pretrained weights."""
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
