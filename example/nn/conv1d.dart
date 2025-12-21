import 'package:tensor/tensor.dart';

void main() {
  final context = Context.best();

  final conv1d = Conv1D.make(
    name: 'conv1d',
    numInChannels: 7,
    numOutChannels: 5,
    // kernelSize: 3,
  );

  final input = Tensor.randn([2, 3, 7]);

  final output = conv1d.forward(input, context: context);
  print(output);
}
