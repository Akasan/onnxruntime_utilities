# onnxruntime_utilities

`onnxruntime_utilities` is a utilities for onnxruntime. 

## Usage
### Load onnx as InferenceSession
```
from onnxruntime_utilities import load_inference_session

sess = load_inference_session("hoge.onnx")
```

### Show input/output summary
```
from onnxruntime_utilities import show_input_summary, show_output_summary

show_input_summary(sess)
show_output_summary(sess)
```

### Execute inference
```
from onnxruntime_utilities import inference

try:
    inputs = {}
    result = inference(sess, inputs)
except ValueError:
    print("Invalid inputs")
```
