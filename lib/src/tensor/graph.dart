import 'package:tensor/tensor.dart';

abstract class TensorGraph {}

abstract class UnaryTensorOp {}

class NegTensorOp implements UnaryTensorOp {}

class UnaryTensorGraph {
  final Tensor input;

  final List<UnaryTensorOp> _ops = [];

  UnaryTensorGraph(this.input);

  UnaryTensorGraph operator -() {
    _ops.add(NegTensorOp());
    return this;
  }
}
