import 'package:tensor/tensor.dart';
import 'package:test/test.dart';

void main() {
  test('calculateGain supports sigmoid', () {
    final gain = calculateGain(KaimingNonLinearity.sigmoid, null);
    expect(gain, 1.0);
  });
}
