from onnxruntime import InferenceSession


def show_input_summary(sess: InferenceSession):
    for i, _input in enumerate(sess.get_inputs()):
        name = _input.name
        shape = _input.shape
        _type = _input.type
        print(f"[{i}] name: {name}, shape: {shape}, type: {_type}")


def show_output_summary(sess: InferenceSession):
    for i, output in enumerate(sess.get_outputs()):
        name = output.name
        shape = output.shape
        _type = output.type
        print(f"[{i}] name: {name}, shape: {shape}, type: {_type}")
