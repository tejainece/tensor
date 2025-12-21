import 'package:tensor/tensor.dart';
import 'package:test/test.dart';

void main() {
  test('Generator.create works separately from default generator', () {
    final device = Device(deviceType: DeviceType.cpu, deviceIndex: -1);

    // Create two new generators
    final gen1 = Generator.create(device: device);
    final gen2 = Generator.create(device: device);

    // Set different seeds
    gen1.currentSeed = 12345;
    gen2.currentSeed = 67890;

    expect(gen1.currentSeed, 12345);
    expect(gen2.currentSeed, 67890);

    // Verify they produce different random numbers
    final t1 = Tensor.rand([10], generator: gen1);
    final t2 = Tensor.rand([10], generator: gen2);

    expect(t1.allClose(t2), isFalse); // Should be different

    // Verify repeatability
    gen1.currentSeed = 12345;
    final t3 = Tensor.rand([10], generator: gen1);
    expect(t1.allClose(t3), isTrue); // Should be same as t1
  });
}
