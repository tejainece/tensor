import 'dart:async';
import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'package:tensor/tensor.dart';

abstract class GGUFLoader extends SafeTensorLoader {
  @override
  Map<String, String> get metadata;

  @override
  Map<String, SafeTensorInfo> get tensorInfos;

  Map<String, dynamic> get ggufMetadata;
}

class MmapGGUFLoader extends GGUFLoader {
  final GGUFHeader header;
  final int fd;
  final int mmapedLength;
  final Pointer<Uint8> _pointer;

  MmapGGUFLoader._({
    required this.header,
    required this.fd,
    required this.mmapedLength,
    required Pointer<Uint8> pointer,
  }) : _pointer = pointer;

  @override
  late final Map<String, String> metadata = header.metadata.map(
    (key, value) => MapEntry(key, value.toString()),
  );

  @override
  late final Map<String, SafeTensorInfo> tensorInfos = _computeTensorInfos(
    header,
    mmapedLength,
  );

  static Map<String, SafeTensorInfo> _computeTensorInfos(
    GGUFHeader header,
    int fileLength,
  ) {
    // Sort by offset to calculate sizes
    final sortedInfos = header.tensorInfos.toList()
      ..sort((a, b) => a.offset.compareTo(b.offset));

    final map = <String, SafeTensorInfo>{};
    final dataSize = fileLength - header.dataOffset;

    for (int i = 0; i < sortedInfos.length; i++) {
      final info = sortedInfos[i];
      final start = info.offset;
      final end = (i < sortedInfos.length - 1)
          ? sortedInfos[i + 1].offset
          : dataSize;

      map[info.name] = SafeTensorInfo(
        dtype: info.type.name,
        shape: info.shape,
        startOffset: start,
        endOffset: end,
      );
    }
    return map;
  }

  @override
  Map<String, dynamic> get ggufMetadata => header.metadata;

  @override
  Tensor? tryLoadByName(String name, {Device device = Device.cpu}) {
    // Find GGUF info
    final ggufInfo = header.tensorInfos.firstWhere(
      (e) => e.name == name,
      orElse: () => throw Exception('Tensor not found: $name'),
    );

    final datatype = ggufInfo.type.toDataType();
    if (datatype == null) {
      return null;
    }

    final dataPointer = _pointer + header.dataOffset + ggufInfo.offset;

    final tensor = Tensor.fromBlob(
      dataPointer.cast<Void>(),
      ggufInfo.shape,
      datatype: datatype,
      device: Device.cpu,
    );

    if (device.deviceType != DeviceType.cpu) {
      return tensor.to(device: device);
    }
    return tensor;
  }

  @override
  Tensor loadByName(String name, {Device device = Device.cpu}) {
    final t = tryLoadByName(name, device: device);
    if (t == null) throw Exception('Tensor not found $name');
    return t;
  }

  @override
  Future<void> release() async {
    MmapSafeTensorLoader.munmap(_pointer, mmapedLength);
    MmapSafeTensorLoader.close(fd);
  }

  static MmapGGUFLoader make({
    required GGUFHeader header,
    required String path,
    required int fileLength,
  }) {
    final fd = MmapSafeTensorLoader.open(path.toNativeUtf8(), 0);
    if (fd == -1) {
      throw Exception('Failed to open file: $path');
    }
    final Pointer<Uint8> result = MmapSafeTensorLoader.mmap(
      nullptr,
      fileLength,
      1, // PROT_READ
      2, // MAP_PRIVATE (or 1 for shared?) 2 is normally MAP_PRIVATE on some systems?
      // Wait, MmapSafeTensorLoader uses:
      // 1 = PROT_READ
      // 2 = MAP_PRIVATE
      fd,
      0,
    );
    if (result.address == -1) {
      throw Exception('mmap failed');
    }
    return MmapGGUFLoader._(
      header: header,
      fd: fd,
      mmapedLength: fileLength,
      pointer: result,
    );
  }
}
