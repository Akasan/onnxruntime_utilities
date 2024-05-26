import torch
from typing import Any
from torch import nn


def export(
    model: nn.Module,
    inputs: tuple[Any],
    onnx_filename: str,
):
    torch.onnx.export(
        model,
        inputs,
        onnx_filename,
    )


def dynamo_export(
    model: nn.Module,
    inputs: tuple[Any],
    onnx_filename: str,
):
    _model = torch.onnx.dynamo_export(
        model,
        *inputs,
    )
    _model.save(onnx_filename)
