from onnxruntime import InferenceSession


def show_input_summary(sess: InferenceSession):
    for i, _input in enumerate(sess.get_inputs()):
        print(f"[{i}] name: {_input.name}, shape: {_input.shape}, type: {_input.type}")


def show_output_summary(sess: InferenceSession):
    for i, output in enumerate(sess.get_outputs()):
        print(f"[{i}] name: {output.name}, shape: {output.shape}, type: {output.type}")
