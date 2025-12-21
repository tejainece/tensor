import 'package:tensor/tensor.dart';
import 'package:test/test.dart';

void main() {
  group('LinearTransposed', () {
    test('forward pass matches transformers.Conv1D', () {
      // Data generated from python script:
      /*
      Input:
      [[[1.0, 2.0], [3.0, 4.0]]]
      Weight:
      [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
      Bias:
      [0.1, 0.2, 0.3, 0.4]
      Output:
      [[[1.2, 1.6, 2.0, 2.4], [2.4, 3.2, 4.0, 4.8]]]
      */

      // Input shape: [1, 2, 2]
      final input = Tensor.from(
        [1.0, 2.0, 3.0, 4.0],
        [1, 2, 2],
        dataType: DataType.float32,
      );

      // inFeatures=2, outFeatures=4
      // Weight shape: [2, 4]
      final weight = Tensor.from(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [2, 4],
        dataType: DataType.float32,
      );

      // Bias shape: [4]
      final bias = Tensor.from(
        [0.1, 0.2, 0.3, 0.4],
        [4],
        dataType: DataType.float32,
      );

      final layer = LinearTransposed(name: 'test', weight: weight, bias: bias);

      final output = layer.forward(input, context: Context(device: Device.cpu));

      final expected = Tensor.from(
        [1.2, 1.6, 2.0, 2.4, 2.4, 3.2, 4.0, 4.8],
        [1, 2, 4],
        dataType: DataType.float32,
      );

      // Allow small floating point differences
      final diff = (output - expected).abs();
      expect(diff.max().scalar as double, lessThan(1e-6));
    });

    test('make factory creates correct shapes', () {
      final layer = LinearTransposed.make(
        name: 'test_make',
        numInFeatures: 5,
        numOutFeatures: 10,
      );

      expect(layer.weight.shape, [5, 10]);
      expect(layer.bias?.shape, [10]);
      expect(layer.numInFeatures, 5);
      expect(layer.numOutFeatures, 10);

      // Check initialization (std 0.02)
      // Hard to check strictly randomly, but values should be small
      expect(layer.weight.abs().mean().scalar as double, lessThan(1.0));
    });

    test('serialization meta', () {
      final layer = LinearTransposed.make(
        name: 'test_make',
        numInFeatures: 5,
        numOutFeatures: 10,
      );
      expect(layer.meta, {
        "numInFeatures": 5,
        "numOutFeatures": 10,
        "hasBias": true,
      });
    });
  });
}
