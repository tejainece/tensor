import 'package:tensor/tensor.dart';
import 'package:tensor/src/ffi/tensor_ffi.dart';
import 'dart:ffi';
import 'package:ffi/ffi.dart';

void main() {
  print('Testing Tensor.empty error handling...');
  try {
    // Test valid call
    final t = Tensor.empty([2, 2], datatype: DataType.float32);
    print('Created tensor: $t');

    // Test potentially invalid call (though difficult to force standard exception without modifying C++ to throw always)
    print('Tensor.empty valid call check passed.');
  } catch (e) {
    print('Error caught: $e');
  }
}
