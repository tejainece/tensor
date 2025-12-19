import 'dart:ffi';

import 'package:tensor/tensor.dart';
import 'package:test/test.dart';

void main() {
  group('Tensor.abs', () {
    test('abs of positive numbers', () {
      final t = Tensor.from([1.0, 2.0, 3.0], [3], dataType: DataType.float32);
      final result = t.abs();
      expect(result.dataPointer.cast<Float>().asTypedList(3), [1.0, 2.0, 3.0]);
    });

    test('abs of negative numbers', () {
      final t = Tensor.from(
        [-1.0, -2.0, -3.0],
        [3],
        dataType: DataType.float32,
      );
      final result = t.abs();
      expect(result.dataPointer.cast<Float>().asTypedList(3), [1.0, 2.0, 3.0]);
    });

    test('abs of mixed numbers', () {
      final t = Tensor.from([-1.0, 2.0, -3.0], [3], dataType: DataType.float32);
      final result = t.abs();
      expect(result.dataPointer.cast<Float>().asTypedList(3), [1.0, 2.0, 3.0]);
    });

    test('abs of zero', () {
      final t = Tensor.from([0.0], [1], dataType: DataType.float32);
      final result = t.abs();
      expect(result.dataPointer.cast<Float>().asTypedList(1), [0.0]);
    });
  });
}
