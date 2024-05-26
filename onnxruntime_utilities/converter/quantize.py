from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_dynamic(
    original_onnx_filename: str,
    quantized_onnx_filename: str,
    weight_type: QuantType = QuantType.QInt8
):
    quantize_dynamic(
        original_onnx_filename,
        quantized_onnx_filename,
        weight_type=weight_type
    )
