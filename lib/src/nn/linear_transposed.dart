import 'package:tensor/tensor.dart';

/// 1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
/// Basically acts like a Linear layer but the weights are transposed.
class LinearTransposed extends Module implements SimpleModule {
  /// The weight matrix of shape [numInFeatures, numOutFeatures].
  final Tensor weight;
  final Tensor? bias;

  LinearTransposed({
    super.name = 'linear_transposed',
    required this.weight,
    this.bias,
  });

  int get numInFeatures => weight.shape[0];
  int get numOutFeatures => weight.shape[1];

  @override
  Tensor forward(Tensor input, {required Context context}) {
    context.onloadModule(this);
    // Ensure input is on the same device as the weights
    final inputs = input.to(device: context.device); // TODO remove if possible
    Tensor output = inputs.matmul(weight);
    if (bias != null) {
      output = output + bias!;
    }
    return output;
  }

  @override
  void resetParameters({Generator? generator}) {
    // Transformers Conv1D uses normal distribution with std 0.02
    weight.normal_(mean: 0.0, std: 0.02, generator: generator);
    if (bias != null) {
      bias!.zeros_();
    }
  }

  @override
  Map<String, dynamic> get meta => {
    "numInFeatures": numInFeatures,
    "numOutFeatures": numOutFeatures,
    "hasBias": bias != null,
  };

  @override
  late final Iterable<Tensor> parameters = {weight, if (bias != null) bias!};

  @override
  final Iterable<Module> submodules = const [];

  Future<void> copyFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
  }) async {
    if (loader.hasTensor('${prefix}weight')) {
      final newWeight = await loader.loadByName('${prefix}weight');
      weight.copy_(newWeight);
    }
    if (bias != null && loader.hasTensor('${prefix}bias')) {
      final newBias = await loader.loadByName('${prefix}bias');
      bias!.copy_(newBias);
    }
  }

  static Future<LinearTransposed> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    String name = 'linear_transposed',
  }) async {
    final weight = await loader.loadByName('${prefix}weight');
    Tensor? bias;
    if (loader.hasTensor('${prefix}bias')) {
      bias = await loader.loadByName('${prefix}bias');
    }
    return LinearTransposed(name: name, weight: weight, bias: bias);
  }

  static LinearTransposed make({
    String name = 'linear_transposed',
    required int numInFeatures,
    required int numOutFeatures,
    bool hasBias = true,
  }) {
    final weight = Tensor.empty([numInFeatures, numOutFeatures]);
    final bias = hasBias ? Tensor.empty([numOutFeatures]) : null;
    return LinearTransposed(name: name, weight: weight, bias: bias)
      ..resetParameters();
  }
}
