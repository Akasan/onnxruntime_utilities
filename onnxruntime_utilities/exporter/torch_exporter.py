from typing import Any

import torch
from packaging import version
from torch import nn

IS_TORCH_LESS_THAN_1_11 = version.parse(
    version.parse(torch.__version__).base_version
) < version.parse("1.11")


def export(
    model: nn.Module,
    inputs: tuple[Any],
    f: str,
    input_names: list[str] = [],
    output_names: list[str] = [],
    dynamic_axes: dict[dict[int, str]] = dict(),
    opset_version: int = 17,
    use_external_data_format: bool = True,
    do_constant_folding: bool = True,
    enable_onnx_checker: bool = True,
):
    kwargs = {
        "opset_version": opset_version,
        "do_constant_folding": do_constant_folding,
    }
    if not input_names == []:
        kwargs["input_names"] = input_names
    if not output_names == []:
        kwargs["output_names"] = output_names
    if not dynamic_axes == dict():
        kwargs["dynamic_axes"] = dynamic_axes

    if IS_TORCH_LESS_THAN_1_11:
        kwargs["use_external_data_format"] = use_external_data_format
        kwargs["enable_onnx_checker"] = enable_onnx_checker

    torch.onnx.export(
        model,
        inputs,
        f,
        **kwargs,
    )


def dynamo_export(
    model: nn.Module,
    inputs: tuple[Any],
    f: str,
):
    _model = torch.onnx.dynamo_export(
        model,
        *inputs,
    )
    _model.save(f)
