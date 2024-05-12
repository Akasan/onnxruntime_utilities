from onnxruntime import InferenceSession


def load_inference_session(filename: str) -> InferenceSession:
    return InferenceSession(filename)
