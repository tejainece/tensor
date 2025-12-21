import 'dart:math';

import 'package:tensor/tensor.dart';

class Conv1D extends Module implements SimpleModule {
  /// [weight] is of shape (numOutChannels, numInChannels, kernelSize)
  Tensor weight;

  /// [bias] is of shape (numOutChannels)
  Tensor? bias;

  Conv1D({required super.name, required this.weight, this.bias});

  @override
  /// [input] is of shape (batch, numInChannels, input_seq_length)
  /// [output] is of shape (batch, numOutChannels, output_seq_length)
  Tensor forward(Tensor input, {required Context context}) {
    context.onloadModule(this);

    input = input.to(device: context.device); // TODO handle this better?

    Tensor output = input.matmul(weight);
    if (bias != null) {
      output = output + bias;
    }
    return output;
  }

  @override
  void resetParameters({Generator? generator}) {
    Init.kaimingUniform_(weight, a: sqrt(5), generator: generator);
    if (bias != null) {
      final fan = Init.calculateKaimingFan(weight);
      if (fan.fanIn != 0) {
        double bound = 1.0 / sqrt(fan.fanIn);
        bias!.uniform_(from: -bound, to: bound, generator: generator);
      }
    }
  }

  @override
  late final Iterable<Tensor> parameters = [weight, if (bias != null) bias!];

  int get numInChannels => weight.shape[1];
  int get numOutChannels => weight.shape[0];
  int get kernelSize => weight.shape[2];

  @override
  Map<String, dynamic> get meta => {
    'numInChannels': numInChannels,
    'numOutChannels': numOutChannels,
    'kernelSize': kernelSize,
  };

  @override
  late final Iterable<Module> submodules = [];

  static Conv1D make({
    required String name,
    required int numInChannels,
    required int numOutChannels,
    int kernelSize = 1,
    /* TODO int stride = 1,
    int padding = 0,
    int dilation = 1,
    int groups = 1,
    bool hasBias = true,
    PadMode? padMode,
    Generator? generator,
    DataType? dataType,
    Device? device,*/
  }) {
    final weight =
        Tensor.randn([numOutChannels, numInChannels, kernelSize]) * 0.02;
    final bias = Tensor.zeros([numOutChannels]);

    return Conv1D(name: name, weight: weight, bias: bias);
  }

  static Future<Conv1D> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required String prefix,
    required String name,
  }) async {
    final weight = await loader.loadByName('${prefix}weight');
    final bias = await loader.loadByName('${prefix}bias');

    return Conv1D(name: name, weight: weight, bias: bias);
  }
}
