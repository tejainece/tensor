import 'package:tensor/tensor.dart';

void main() {
  int targetLength = 5;
  int kvLen = 5;
  int pastKeyValuesLength = 0;
  Device device = Device.cpu;

  final rangeRow = Tensor.arange(
    0,
    targetLength,
    device: device,
  ).view([targetLength, 1]);
  final rangeCol = Tensor.arange(0, kvLen, device: device).view([1, kvLen]);

  final maskCondition = rangeCol.gt(rangeRow + pastKeyValuesLength);

  print(maskCondition);
}
