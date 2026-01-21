import 'package:tensor/tensor.dart';
import 'package:universal_io/io.dart';

export 'metadata.dart';
export 'storage.dart';

class GGUFFile {
  final GGUFHeader header;
  final String path;
  final int fileLength;

  GGUFFile({
    required this.header,
    required this.path,
    required this.fileLength,
  });

  GGUFLoader cpuLoader() {
    if (Platform.isLinux ||
        Platform.isMacOS ||
        Platform.isAndroid ||
        Platform.isIOS ||
        Platform.isFuchsia) {
      return MmapGGUFLoader.make(
        header: header,
        path: path,
        fileLength: fileLength,
      );
    }
    // Fallback? Currently only mmap loader implemented for GGUF.
    // SafeTensors has FileIOSafeTensorLoader.
    // For now throwing or implementing FileIO loader?
    // Plan didn't explicitly ask for FileIO loader, but it's good practice.
    // However, GGUF without mmap is slow and less useful for large models.
    // Let's stick to mmap for now as per plan focus on mmap.
    throw UnimplementedError(
      'GGUF loading on this platform is not yet supported (mmap required)',
    );
  }

  GGUFLoader cudaLoader() {
    return cpuLoader();
  }

  static Future<GGUFFile> load(String path) async {
    RandomAccessFile file = await File(path).open();
    final fileLength = await file.length();
    try {
      final header = await GGUFHeader.read(file);
      return GGUFFile(header: header, path: path, fileLength: fileLength);
    } finally {
      await file.close();
    }
  }
}
