"""Training-surface controls for the frozen Phase-2 decoder bank."""

from __future__ import annotations

from torch import nn


def configure_decoder_operator_only_training(module: nn.Module) -> tuple[str, ...]:
    """Freeze an initialized model except for its decoder operator bank."""
    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch module")
    decoder_operator = getattr(module, "decoder_operator", None)
    if not isinstance(decoder_operator, nn.Module):
        raise ValueError("decoder_operator must be an initialized module")

    trainable = tuple(
        name
        for name, _ in module.named_parameters()
        if name.startswith("decoder_operator.")
    )
    if not trainable:
        raise ValueError("decoder_operator must contain trainable parameters")
    trainable_names = frozenset(trainable)
    for name, parameter in module.named_parameters():
        parameter.requires_grad_(name in trainable_names)
    return trainable


def enforce_decoder_operator_only_training_mode(module: nn.Module) -> None:
    """Keep the frozen backbone in eval while the decoder bank trains."""
    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch module")
    decoder_operator = getattr(module, "decoder_operator", None)
    if not isinstance(decoder_operator, nn.Module):
        raise ValueError("decoder_operator must be an initialized module")
    for name, child in module.named_children():
        if name != "decoder_operator":
            child.eval()
    decoder_operator.train(True)
