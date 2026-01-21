import 'package:tensor/tensor.dart';
import 'package:test/test.dart';
import 'package:universal_io/io.dart';

void main() {
  group('CompositeSafeTensorLoader', () {
    test('loadSplitSafeTensors', () async {
      final baseDir = './testdata/models/llm/gemma/v1/2b';
      final loader = await CompositeSafeTensorLoader.loadSplitSafeTensors(
        Directory(baseDir),
      );
    });
  });
}
