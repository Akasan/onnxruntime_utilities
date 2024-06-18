import torch
from torch import nn

from onnxruntime_utilities.exporter.torch_exporter import export


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20)

    def forward(self, x):
        return self.linear(x)


model = Model()
export(
    model,
    inputs=(torch.randn(1, 10),),
    f="model.onnx",
    input_names=["sample"],
    output_names=["output"],
    dynamic_axes={
        "sample": {0: "batch", 1: "channels"},
    },
)
