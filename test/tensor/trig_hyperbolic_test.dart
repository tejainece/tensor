import 'dart:math';

import 'package:tensor/tensor.dart';
import 'package:test/test.dart';

void main() {
  group('Trigonometric and Hyperbolic Functions', () {
    test('tan', () {
      final t = Tensor.from([0.0, pi / 4, pi], [3], dataType: DataType.float32);
      final result = t.tan();
      expect(result[0].scalar, closeTo(0.0, 1e-5));
      expect(result[1].scalar, closeTo(1.0, 1e-5));
      expect(result[2].scalar, closeTo(0.0, 1e-5));
    });

    test('tanh', () {
      final t = Tensor.from([0.0, 100.0], [2], dataType: DataType.float32);
      final result = t.tanh();
      expect(result[0].scalar, closeTo(0.0, 1e-5));
      expect(result[1].scalar, closeTo(1.0, 1e-5));
    });

    test('asin / arcsin', () {
      final t = Tensor.from([0.0, 0.5, 1.0], [3], dataType: DataType.float32);
      final expected = [0.0, asin(0.5), asin(1.0)];

      final result = t.asin();
      final resultAlias = t.arcsin();

      for (var i = 0; i < 3; i++) {
        expect(result[i].scalar, closeTo(expected[i], 1e-5));
        expect(resultAlias[i].scalar, closeTo(expected[i], 1e-5));
      }
    });

    test('atan / arctan', () {
      final t = Tensor.from([0.0, 1.0, 100.0], [3], dataType: DataType.float32);
      final expected = [0.0, atan(1.0), atan(100.0)];

      final result = t.atan();
      final resultAlias = t.arctan();

      for (var i = 0; i < 3; i++) {
        expect(result[i].scalar, closeTo(expected[i], 1e-5));
        expect(resultAlias[i].scalar, closeTo(expected[i], 1e-5));
      }
    });

    test('atan2 / arctan2', () {
      final y = Tensor.from([1.0, 0.0], [2], dataType: DataType.float32);
      final x = Tensor.from([1.0, 1.0], [2], dataType: DataType.float32);
      // atan2(1,1) = pi/4, atan2(0,1) = 0
      final result = y.atan2(x);
      final resultAlias = y.arctan2(x);

      expect(result[0].scalar, closeTo(pi / 4, 1e-5));
      expect(result[1].scalar, closeTo(0.0, 1e-5));

      expect(resultAlias[0].scalar, closeTo(pi / 4, 1e-5));
      expect(resultAlias[1].scalar, closeTo(0.0, 1e-5));
    });

    test('sinc', () {
      // sinc(0) = 1. normalized sinc usually used in DSP is sin(pi*x)/(pi*x), but pytorch sinc is sin(pi*x)/(pi*x).
      // PyTorch documentation says: sinc(x) = sin(pi*x)/(pi*x).
      // if x=0, 1. if x=0.5, sin(pi/2)/(pi/2) = 1/(pi/2) = 2/pi.
      final t = Tensor.from([0.0, 0.5], [2], dataType: DataType.float32);
      final result = t.sinc();

      expect(result[0].scalar, closeTo(1.0, 1e-5));
      expect(result[1].scalar, closeTo(2 / pi, 1e-5));
    });

    test('sinh', () {
      final t = Tensor.from([0.0, 1.0], [2], dataType: DataType.float32);
      final result = t.sinh();
      // sinh(0) = 0, sinh(1) = (e - 1/e)/2
      expect(result[0].scalar, closeTo(0.0, 1e-5));
      expect(result[1].scalar, closeTo((exp(1) - exp(-1)) / 2, 1e-5));
    });

    test('asinh / arcsinh', () {
      final t = Tensor.from(
        [0.0, 1.1752],
        [2],
        dataType: DataType.float32,
      ); // sinh(1) approx 1.1752
      final result = t.asinh();
      final resultAlias = t.arcsinh();

      expect(result[0].scalar, closeTo(0.0, 1e-4));
      expect(result[1].scalar, closeTo(1.0, 1e-4));
      expect(resultAlias[1].scalar, closeTo(1.0, 1e-4));
    });

    test('atanh / arctanh', () {
      final t = Tensor.from(
        [0.0, 0.76159],
        [2],
        dataType: DataType.float32,
      ); // tanh(1) approx 0.76159
      final result = t.atanh();
      final resultAlias = t.arctanh();

      expect(result[0].scalar, closeTo(0.0, 1e-4));
      expect(result[1].scalar, closeTo(1.0, 1e-4));
      expect(resultAlias[1].scalar, closeTo(1.0, 1e-4));
    });
  });
}
