import 'dart:ffi';
import 'package:tensor/src/ffi/torch_ffi.dart';
import 'package:universal_io/io.dart';

String getAoLibraryPath() {
  if (Platform.environment.containsKey('TORCHAO_LIBRARY_PATH')) {
    return Platform.environment['TORCHAO_LIBRARY_PATH']!;
  }

  String filename;
  if (Platform.isMacOS) {
    filename = 'libtorchao.dylib';
  } else if (Platform.isLinux) {
    filename = 'libtorchao.so';
  } else if (Platform.isWindows) {
    filename = 'libtorchao.dll';
  } else {
    throw UnsupportedError('Unsupported platform: ${Platform.operatingSystem}');
  }

  // Fallback to expecting it next to libtorchffi or in assets if embedded
  // For dev environment, we might default to aoffi/build/...
  // checking common build location:
  final scriptUri = Platform.script;
  final buildPath = scriptUri.resolve('../../../aoffi/build/$filename');
  if (File.fromUri(buildPath).existsSync()) {
    return buildPath.toFilePath();
  }

  return getEmbeddedLibraryPath(filename);
}

final DynamicLibrary aoLib = DynamicLibrary.open(getAoLibraryPath());

// --- C Function Signatures ---

typedef TorchAoInitC = Void Function();
typedef TorchAoInitDart = void Function();

typedef TorchAoFloat8LinearC =
    CTensor Function(
      CTensor input,
      CTensor input_scales,
      CTensor weight,
      CTensor weight_scales,
      CTensor bias,
      Int8 output_dtype,
    );

typedef TorchAoFloat8LinearDart =
    CTensor Function(
      CTensor input,
      CTensor input_scales,
      CTensor weight,
      CTensor weight_scales,
      CTensor bias,
      int output_dtype,
    );

// --- Dart API ---

class TorchAoFFI {
  static final init = aoLib.lookupFunction<TorchAoInitC, TorchAoInitDart>(
    'torchao_init',
  );

  static final float8Linear = aoLib
      .lookupFunction<TorchAoFloat8LinearC, TorchAoFloat8LinearDart>(
        'torchao_float8_linear',
      );
}

// Call init on load to ensure ops are registered
// (Actually better to call it explicitly or relying on load)
// void _init() => TorchAoFFI.init();
