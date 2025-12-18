# Tensor: Dart Bindings for Libtorch

This package provides a Dart wrapper around `libtorch` (PyTorch C++ API), enabling powerful tensor operations, neural network modules, and GPU acceleration directly from Dart.

## Features

-   **Seamless Interop**: Call `libtorch` functions from Dart using FFI.
-   **Tensor Operations**: Create, manipulate, and compute with tensors (reshaping, slicing, math operations).
-   **Device Support**: Run computations on CPU, CUDA (Nvidia), and MPS (Apple Silicon).
-   **Neural Networks**: Building blocks for constructing neural network layers.

## Usage

Here is a basic example of creating a tensor and moving it to a device:

```dart
import 'package:tensor/tensor.dart';

void main() {
  // Create an Identity matrix of size 7x7
  // Automatically tries to use MPS (Apple Silicon) if available, otherwise CPU
  final tensor = Tensor.eye(
    7,
    device: Device(deviceType: DeviceType.mps, deviceIndex: -1),
  );
  
  print('Dimensions: ${tensor.dim}');
  print('Sizes: ${tensor.sizes}');
  print('Device: ${tensor.device}');
  print('Tensor Data:');
  print(tensor);
}
```

## Roadmap

### Feature Phases
- [ ] Phase 0: Save Module to disk as Safetensor
- [ ] Phase 1: DownSample2D, UpSample2D
- [ ] Phase 2: DownEncoderBlock2D, UpDecoderBlock2D
- [ ] Phase 4: VAE (Flux, SD 1.5, SDXL, Qwen Image)
- [ ] Phase 5: Unet
- [ ] Phase 6: Text encoding
- [ ] Phase 7: LowVRAM system & Graceful C++ exception handling

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
