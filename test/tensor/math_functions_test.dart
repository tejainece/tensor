import 'dart:math' as math;
import 'package:tensor/tensor.dart';
import 'package:test/test.dart';

void main() {
  group('Tensor Mathematical Functions', () {
    test('sin function', () {
      final t1 = Tensor.from(
        [0.0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2],
        [5],
        dataType: DataType.float64,
      );
      final result = t1.sin();

      expect(result.shape, [5]);
      expect(result.at([0]).scalar, closeTo(0.0, 1e-6));
      expect(result.at([1]).scalar, closeTo(0.5, 1e-6));
      expect(result.at([2]).scalar, closeTo(math.sqrt(2) / 2, 1e-6));
      expect(result.at([3]).scalar, closeTo(math.sqrt(3) / 2, 1e-6));
      expect(result.at([4]).scalar, closeTo(1.0, 1e-6));
    });

    test('cos function', () {
      final t1 = Tensor.from(
        [0.0, math.pi / 3, math.pi / 2, math.pi],
        [4],
        dataType: DataType.float64,
      );
      final result = t1.cos();

      expect(result.shape, [4]);
      expect(result.at([0]).scalar, closeTo(1.0, 1e-6));
      expect(result.at([1]).scalar, closeTo(0.5, 1e-6));
      expect(result.at([2]).scalar, closeTo(0.0, 1e-6));
      expect(result.at([3]).scalar, closeTo(-1.0, 1e-6));
    });

    test('exp function', () {
      final t1 = Tensor.from(
        [0.0, 1.0, 2.0, -1.0],
        [4],
        dataType: DataType.float64,
      );
      final result = t1.exp();

      expect(result.shape, [4]);
      expect(result.at([0]).scalar, closeTo(1.0, 1e-6));
      expect(result.at([1]).scalar, closeTo(math.e, 1e-6));
      expect(result.at([2]).scalar, closeTo(math.e * math.e, 1e-6));
      expect(result.at([3]).scalar, closeTo(1.0 / math.e, 1e-6));
    });

    test('rsqrt function', () {
      final t1 = Tensor.from(
        [1.0, 4.0, 9.0, 16.0],
        [4],
        dataType: DataType.float64,
      );
      final result = t1.rsqrt();

      expect(result.shape, [4]);
      expect(result.at([0]).scalar, closeTo(1.0, 1e-6));
      expect(result.at([1]).scalar, closeTo(0.5, 1e-6));
      expect(result.at([2]).scalar, closeTo(1.0 / 3.0, 1e-6));
      expect(result.at([3]).scalar, closeTo(0.25, 1e-6));
    });

    test('pow function with scalar exponent', () {
      final t1 = Tensor.from(
        [2.0, 3.0, 4.0, 5.0],
        [4],
        dataType: DataType.float64,
      );
      final result = t1.pow(2.0);

      expect(result.shape, [4]);
      expect(result.at([0]).scalar, closeTo(4.0, 1e-6));
      expect(result.at([1]).scalar, closeTo(9.0, 1e-6));
      expect(result.at([2]).scalar, closeTo(16.0, 1e-6));
      expect(result.at([3]).scalar, closeTo(25.0, 1e-6));
    });

    test('pow function with negative exponent', () {
      final t1 = Tensor.from([2.0, 4.0], [2], dataType: DataType.float64);
      final result = t1.pow(-1.0);

      expect(result.shape, [2]);
      expect(result.at([0]).scalar, closeTo(0.5, 1e-6));
      expect(result.at([1]).scalar, closeTo(0.25, 1e-6));
    });

    test('sin with multi-dimensional tensor', () {
      final t1 = Tensor.from(
        [0.0, math.pi / 2, math.pi, 3 * math.pi / 2],
        [2, 2],
        dataType: DataType.float64,
      );
      final result = t1.sin();

      expect(result.shape, [2, 2]);
      final flattened = result.flatten();
      expect(flattened.at([0]).scalar, closeTo(0.0, 1e-6));
      expect(flattened.at([1]).scalar, closeTo(1.0, 1e-6));
      expect(flattened.at([2]).scalar, closeTo(0.0, 1e-6));
      expect(flattened.at([3]).scalar, closeTo(-1.0, 1e-6));
    });

    test('ceil and floor', () {
      final t = Tensor.from(
        [1.1, 1.9, -1.1, -1.9],
        [4],
        dataType: DataType.float64,
      );

      final c = t.ceil();
      expect(c.at([0]).scalar, equals(2.0));
      expect(c.at([1]).scalar, equals(2.0));
      expect(c.at([2]).scalar, equals(-1.0));
      expect(c.at([3]).scalar, equals(-1.0));

      final f = t.floor();
      expect(f.at([0]).scalar, equals(1.0));
      expect(f.at([1]).scalar, equals(1.0));
      expect(f.at([2]).scalar, equals(-2.0));
      expect(f.at([3]).scalar, equals(-2.0));
    });

    test('ceil_ and floor_ (in-place)', () {
      final t1 = Tensor.from([1.1, 1.9], [2], dataType: DataType.float64);
      t1.ceil_();
      expect(t1.at([0]).scalar, equals(2.0));
      expect(t1.at([1]).scalar, equals(2.0));

      final t2 = Tensor.from([1.1, 1.9], [2], dataType: DataType.float64);
      t2.floor_();
      expect(t2.at([0]).scalar, equals(1.0));
      expect(t2.at([1]).scalar, equals(1.0));
    });

    test('clamp', () {
      final t = Tensor.from(
        [0.0, 0.5, 1.0, 1.5],
        [4],
        dataType: DataType.float64,
      );

      final c1 = t.clamp(min: 0.5, max: 1.0);
      expect(c1.at([0]).scalar, equals(0.5));
      expect(c1.at([1]).scalar, equals(0.5));
      expect(c1.at([2]).scalar, equals(1.0));
      expect(c1.at([3]).scalar, equals(1.0));

      final c2 = t.clamp(min: 0.8);
      expect(c2.at([0]).scalar, equals(0.8));
      expect(c2.at([1]).scalar, equals(0.8));
      expect(c2.at([2]).scalar, equals(1.0));
      expect(c2.at([3]).scalar, equals(1.5));

      final c3 = t.clamp(max: 0.8);
      expect(c3.at([0]).scalar, equals(0.0));
      expect(c3.at([1]).scalar, equals(0.5));
      expect(c3.at([2]).scalar, equals(0.8));
      expect(c3.at([3]).scalar, equals(0.8));
    });

    test('clamp_ (in-place)', () {
      final t = Tensor.from([0.0, 1.5], [2], dataType: DataType.float64);
      t.clamp_(min: 0.5, max: 1.0);
      expect(t.at([0]).scalar, equals(0.5));
      expect(t.at([1]).scalar, equals(1.0));
    });

    test('log', () {
      final t = Tensor.from([1.0, math.e], [2], dataType: DataType.float64);
      final l = t.log();
      expect(l.at([0]).scalar, closeTo(0.0, 1e-6));
      expect(l.at([1]).scalar, closeTo(1.0, 1e-6));
    });

    test('log_ (in-place)', () {
      final t = Tensor.from([1.0, math.e], [2], dataType: DataType.float64);
      t.log_();
      expect(t.at([0]).scalar, closeTo(0.0, 1e-6));
      expect(t.at([1]).scalar, closeTo(1.0, 1e-6));
    });

    test('in-place trig functions (sin_, cos_, tan_, tanh_)', () {
      final t1 = Tensor.from(
        [0.0, math.pi / 2],
        [2],
        dataType: DataType.float64,
      );
      t1.sin_();
      expect(t1.at([0]).scalar, closeTo(0.0, 1e-6));
      expect(t1.at([1]).scalar, closeTo(1.0, 1e-6));

      final t2 = Tensor.from([0.0, math.pi], [2], dataType: DataType.float64);
      t2.cos_();
      expect(t2.at([0]).scalar, closeTo(1.0, 1e-6));
      expect(t2.at([1]).scalar, closeTo(-1.0, 1e-6));

      final t3 = Tensor.from([0.0], [1], dataType: DataType.float64);
      t3.tan_();
      expect(t3.at([0]).scalar, closeTo(0.0, 1e-6));

      final t4 = Tensor.from([0.0], [1], dataType: DataType.float64);
      t4.tanh_();
      expect(t4.at([0]).scalar, closeTo(0.0, 1e-6));
    });

    test('exp with multi-dimensional tensor', () {
      final t1 = Tensor.from(
        [0.0, 1.0, 2.0, 3.0],
        [2, 2],
        dataType: DataType.float64,
      );
      final result = t1.exp();

      expect(result.shape, [2, 2]);
      final flattened = result.flatten();
      expect(flattened.at([0]).scalar, closeTo(1.0, 1e-6));
      expect(flattened.at([1]).scalar, closeTo(math.e, 1e-6));
    });

    test('chaining mathematical operations', () {
      final t1 = Tensor.from([1.0, 4.0, 9.0], [3], dataType: DataType.float64);
      // Test: sqrt(x) = 1/rsqrt(x), so x should equal (1/rsqrt(x))^2
      final rsqrtResult = t1.rsqrt();
      final squared = rsqrtResult.pow(-2.0);

      expect(squared.shape, [3]);
      expect(squared.at([0]).scalar, closeTo(1.0, 1e-6));
      expect(squared.at([1]).scalar, closeTo(4.0, 1e-6));
      expect(squared.at([2]).scalar, closeTo(9.0, 1e-6));
    });
  });
}
