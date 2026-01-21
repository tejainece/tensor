import 'package:test/test.dart';
import 'package:tensor/tensor.dart';

void main() {
  group('GGUF', () {
    test('load', () async {
      final path =
          './testdata/models/llm/gpt2/prunaai/Q5_K_M/gpt2.Q5_K_M.gguf';
      final file = await GGUFFile.load(path);

      expect(file.header.version, 3);
      expect(file.header.tensorCount, 149);
      expect(file.header.metadataKvCount, 22);
      expect(file.header.metadata['general.architecture'], 'gpt2');

      final loader = file.cpuLoader();
      expect(loader.tensorInfos.containsKey('output_norm.bias'), true);

      final tensor = await loader.loadByName('output_norm.bias');
      expect(tensor.shape, [768]);
      expect(tensor.dataType, DataType.float32);

      final list = tensor.toList();
      expect(list.length, 768);

      await loader.release();
    });
  });
}
