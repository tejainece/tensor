import 'package:test/test.dart';
import 'package:tensor/tensor.dart';

void main() {
  group('Num extension test', () {
    test('int to tensor', () {
      final t = 5.to();
      expect(t.scalar, 5);
      expect(t.shape, isEmpty);
      expect(t.dataType, DataType.int64);
    });

    test('double to tensor', () {
      final t = 3.14.to();
      expect(t.scalar, closeTo(3.14, 0.0001));
      expect(t.shape, isEmpty);
      expect(t.dataType, DataType.float32);
    });

    test('int to float tensor', () {
      final t = 5.to(dataType: DataType.float32);
      expect(t.scalar, 5.0);
      expect(t.dataType, DataType.float32);
    });
  });
}
