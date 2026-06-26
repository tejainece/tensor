tensor Dart package should provide basic tensor support similar to pytorch so that other packages can use the tensors to build transformers, llms, diffusion models, etc.

Tensor and TensorGraph are the primitives that provide tensor operations. TensorGraph is base class of Tensor.

TensorGraph is a directed acyclic graph (DAG) of operations that can be executed to produce a tensor.

TensorGraph supports operations similar to Tensor but the operations build a graph instead of executing immediately. TensorGraph will automatically resolve to Tensor when an operation cannot be graphed by the TensorGraph being operated on.

TensorGraph supports both eager and graph execution modes. 