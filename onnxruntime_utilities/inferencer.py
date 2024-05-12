from typing import Any
from onnxruntime import InferenceSession


def is_valid_inputs(sess: InferenceSession, inputs: dict[str, Any]) -> bool:
    desired_input_names = set([i.name for i in sess.get_inputs()])
    specified_input_names = set([i for i in inputs.keys()])
    return desired_input_names == specified_input_names


def inference(sess: InferenceSession, inputs: dict[str, Any]) -> Any:
    if not is_valid_inputs(sess, inputs):
        raise ValueError("Invalid inputs are specified")

    result = sess.run(None, inputs)
    return result
