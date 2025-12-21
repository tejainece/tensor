import 'dart:math';

import 'package:tensor/tensor.dart';

// TODO Mish

abstract class Activation {
  String get name;

  Iterable<String> get aliases;

  const Activation();

  Tensor forward(Tensor x, {required Context context});

  static const ReLU relu = ReLU();
  static const QuickGeluActivation quickGelu = QuickGeluActivation();
  static const GeluActivation gelu = GeluActivation();
  static const GeluTanhActivation geluTanh = GeluTanhActivation();
  static const SiLU silu = SiLU();

  static const List<Activation> list = [quickGelu, gelu, silu, relu, geluTanh];

  static final Map<String, Activation> _byName = () {
    final ret = <String, Activation>{};
    for (final activation in list) {
      ret[activation.name] = activation;
      ret[activation.name.toLowerCase()] = activation;
      for (final alias in activation.aliases) {
        ret[alias.toLowerCase()] = activation;
      }
    }
    return ret;
  }();

  static Activation? fromName(String name) => _byName[name];
}

class ReLU implements Activation {
  @override
  String get name => "ReLU";

  const ReLU();

  @override
  Iterable<String> get aliases => [];

  @override
  Tensor forward(Tensor x, {required Context context}) {
    return x.relu();
  }
}

class QuickGeluActivation implements Activation {
  @override
  String get name => "QuickGeLU";

  const QuickGeluActivation();

  @override
  Iterable<String> get aliases => [];

  @override
  Tensor forward(Tensor x, {required Context context}) {
    return x * (x * 1.702).sigmoid();
  }
}

class GeluActivation implements Activation {
  @override
  String get name => "GeLU";

  const GeluActivation();

  @override
  Iterable<String> get aliases => [];

  @override
  Tensor forward(Tensor x, {required Context context}) {
    return x.gelu(GeluApporimate.none);
  }
}

/// Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
/// Also see the Gaussian Error Linear Units paper: https://huggingface.co/papers/1606.08415
class GeluTanhActivation implements Activation {
  @override
  String get name => "GeLU_New";

  @override
  final List<String> aliases = const ["gelu_tanh"];

  const GeluTanhActivation();

  @override
  Tensor forward(Tensor x, {required Context context}) {
    return x *
        0.5 *
        (((x + x.pow(3.0) * 0.044715) * sqrt(2.0 / pi)).tanh() + 1.0);
  }
}

/// Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
///
/// References:
///   https://arxiv.org/abs/1606.08415
///   https://arxiv.org/abs/1702.03118
///   https://arxiv.org/abs/1710.05941v1
class SiLU implements Activation {
  @override
  String get name => "SiLU";

  const SiLU();

  @override
  final List<String> aliases = const ["swish"];

  @override
  Tensor forward(Tensor x, {required Context context}) {
    return x.silu();
  }
}
