import 'package:test/test.dart';
import 'package:tensor/tensor.dart';

void main() {
  group('Tensor Reduction Operations', () {
    test('max/min global', () {
      final t = Tensor.from(
        [1.0, 5.0, 2.0, 8.0, 3.0],
        [5],
        datatype: DataType.float32,
      );
      expect(t.max().scalar, 8.0);
      expect(t.min().scalar, 1.0);
    });

    test('argmax/argmin global', () {
      final t = Tensor.from(
        [1.0, 5.0, 2.0, 8.0, 3.0],
        [5],
        datatype: DataType.float32,
      );
      expect(t.argmax().scalar, 3);
      expect(t.argmin().scalar, 0);
    });

    test('amax/amin with dim', () {
      // [[1, 2, 3],
      //  [4, 5, 6]]
      final t = Tensor.from(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [2, 3],
        datatype: DataType.float32,
      );

      // dim=0: [4, 5, 6]
      final max0 = t.amax(dim: [0]);
      expect(max0.shape, [3]);
      expect(max0.toList(), [4.0, 5.0, 6.0]);

      // dim=1: [3, 6]
      final max1 = t.amax(dim: [1]);
      expect(max1.shape, [2]);
      expect(max1.toList(), [3.0, 6.0]);

      // dim=0: [1, 2, 3]
      final min0 = t.amin(dim: [0]);
      expect(min0.toList(), [1.0, 2.0, 3.0]);
    });

    test('argmax/argmin with dim', () {
      // [[1, 5, 3],
      //  [4, 2, 6]]
      final t = Tensor.from(
        [1.0, 5.0, 3.0, 4.0, 2.0, 6.0],
        [2, 3],
        datatype: DataType.float32,
      );

      // dim=1 => indices of max in each row: [1, 2]
      final argmax1 = t.argmax(dim: 1);
      expect(argmax1.toList(), [1, 2]);

      // dim=0 => indices of min in each col: [0, 1, 0]
      // col 0: 1 vs 4 -> 0
      // col 1: 5 vs 2 -> 1
      // col 2: 3 vs 6 -> 0
      final argmin0 = t.argmin(dim: 0);
      expect(argmin0.toList(), [0, 1, 0]);
    });
  });
}
