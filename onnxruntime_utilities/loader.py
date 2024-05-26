from onnxruntime import InferenceSession, SessionOptions


def load_inference_session(
    filename: str, session_options: SessionOptions | None = None
) -> InferenceSession:
    if session_options is None:
        return InferenceSession(filename)
    else:
        return InferenceSession(filename, session_options=session_options)
