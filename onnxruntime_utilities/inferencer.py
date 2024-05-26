from typing import Any

from onnxruntime import InferenceSession


def is_valid_input_names(sess: InferenceSession, inputs: dict[str, Any]) -> bool:
    desired_input_names = set([i.name for i in sess.get_inputs()])
    specified_input_names = set([i for i in inputs.keys()])
    return desired_input_names == specified_input_names


def inference(sess: InferenceSession, inputs: dict[str, Any]) -> Any:
    if not is_valid_input_names(sess, inputs):
        raise ValueError("Invalid inputs are specified")

    result = sess.run(None, inputs)
    return result


def generate_inputs(sess: InferenceSession, items: list[Any]) -> dict[str, Any]:
    desired_inputs = sess.get_inputs()

    if not len(desired_inputs) == len(items):
        raise Exception("Not enough input items")

    inputs = dict()
    for _input, item in zip(desired_inputs, items):
        inputs[_input.name] = item

    return inputs
