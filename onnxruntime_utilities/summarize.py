from onnxruntime import InferenceSession


def show_input_summary(sess: InferenceSession):
    for _input in sess.get_inputs():
        print(f"name: {_input.name}, shape: {_input.shape}, type: {_input_type}")


def show_output_summary(sess: InferenceSession):
    for output in sess.get_outputs():
        print(f"name: {output.name}, shape: {output.shape}, type: {output_type}")
