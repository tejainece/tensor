import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:tensor/tensor.dart';

enum GGUFType {
  f32(0),
  f16(1),
  q4_0(2),
  q4_1(3),
  q5_0(6),
  q5_1(7),
  q8_0(8),
  q8_1(9),
  q2_k(10),
  q3_k(11),
  q4_k(12),
  q5_k(13),
  q6_k(14),
  q8_k(15),
  i8(16),
  i16(17),
  i32(18),
  i64(19),
  f64(20),
  iq2_xxs(21),
  iq2_xs(22),
  iq3_xxs(23),
  iq1_s(24),
  iq4_nl(25),
  iq3_s(26),
  iq2_s(27),
  iq4_xs(28),
  i8_ss(29),
  i8_s(30),
  i8_q8_0(31),
  i16_p0(32),
  i16_p1(33),
  i16_q8_0(34);

  final int id;
  const GGUFType(this.id);

  static GGUFType fromId(int id) {
    return GGUFType.values.firstWhere(
      (e) => e.id == id,
      orElse: () => throw Exception('Unknown GGUFType id: $id'),
    );
  }

  DataType? toDataType() {
    switch (this) {
      case GGUFType.f32:
        return DataType.float32;
      case GGUFType
          .f16: // Not strictly supported by all backends yet as native type, but usually maps to f16 or f32
        return DataType
            .float32; // Mapping to float32 for now as Dart/Tensor usage often promotes. Or maybe we should support f16 specific?
      case GGUFType.q5_k: // Map quantized types to uint8 for now
        return DataType.uint8;
      // TODO: Add support for other quantized types
      default:
        return null;
    }
  }
}

enum GGUFMetadataValueType {
  uint8(0),
  int8(1),
  uint16(2),
  int16(3),
  uint32(4),
  int32(5),
  float32(6),
  bool(7),
  string(8),
  array(9),
  uint64(10),
  int64(11),
  float64(12);

  final int id;
  const GGUFMetadataValueType(this.id);

  static GGUFMetadataValueType fromId(int id) {
    return GGUFMetadataValueType.values.firstWhere(
      (e) => e.id == id,
      orElse: () => throw Exception('Unknown GGUFMetadataValueType id: $id'),
    );
  }
}

class GGUFTensorInfo {
  final String name;
  final List<int> shape;
  final GGUFType type;
  final int offset;

  GGUFTensorInfo({
    required this.name,
    required this.shape,
    required this.type,
    required this.offset,
  });
}

class GGUFHeader {
  static const int magic = 0x46554747; // 'GGUF'

  final int version;
  final int tensorCount;
  final int metadataKvCount;
  final Map<String, dynamic> metadata;
  final List<GGUFTensorInfo> tensorInfos;
  final int dataOffset;

  GGUFHeader({
    required this.version,
    required this.tensorCount,
    required this.metadataKvCount,
    required this.metadata,
    required this.tensorInfos,
    required this.dataOffset,
  });

  static Future<GGUFHeader> read(RandomAccessFile file) async {
    // Helper to read bytes
    Future<ByteData> readBytes(int count) async {
      final buffer = Uint8List(count);
      final read = await file.readInto(buffer);
      if (read != count)
        throw Exception('Unexpected EOF during GGUF header read');
      return ByteData.sublistView(buffer);
    }

    // GGUF is typically Little Endian
    final endian = Endian.little;

    final magicBytes = await readBytes(4);
    if (magicBytes.getUint32(0, endian) != magic) {
      throw Exception('Invalid GGUF magic number');
    }

    final versionBytes = await readBytes(4);
    final version = versionBytes.getUint32(0, endian);

    // Only support version 3 and above ideally, but let's check parsing
    // v1, v2 had slightly different packing? Llama.cpp uses v3 mostly now.

    final countsBytes = await readBytes(16); // 2 * uint64
    final tensorCount = countsBytes.getUint64(0, endian);
    final metadataKvCount = countsBytes.getUint64(8, endian);

    final metadata = <String, dynamic>{};

    // Helper to read simple types
    Future<dynamic> readValue(GGUFMetadataValueType type) async {
      switch (type) {
        case GGUFMetadataValueType.uint8:
          return (await readBytes(1)).getUint8(0);
        case GGUFMetadataValueType.int8:
          return (await readBytes(1)).getInt8(0);
        case GGUFMetadataValueType.uint16:
          return (await readBytes(2)).getUint16(0, endian);
        case GGUFMetadataValueType.int16:
          return (await readBytes(2)).getInt16(0, endian);
        case GGUFMetadataValueType.uint32:
          return (await readBytes(4)).getUint32(0, endian);
        case GGUFMetadataValueType.int32:
          return (await readBytes(4)).getInt32(0, endian);
        case GGUFMetadataValueType.float32:
          return (await readBytes(4)).getFloat32(0, endian);
        case GGUFMetadataValueType.bool:
          return (await readBytes(1)).getUint8(0) != 0;
        case GGUFMetadataValueType.string:
          final len = (await readBytes(
            8,
          )).getUint64(0, endian); // String length is uint64
          final strBytes = Uint8List(len);
          await file.readInto(strBytes);
          return utf8.decode(strBytes);
        case GGUFMetadataValueType.uint64:
          return (await readBytes(8)).getUint64(0, endian);
        case GGUFMetadataValueType.int64:
          return (await readBytes(8)).getInt64(0, endian);
        case GGUFMetadataValueType.float64:
          return (await readBytes(8)).getFloat64(0, endian);
        case GGUFMetadataValueType.array:
          final typeId = (await readBytes(4)).getUint32(0, endian);
          final len = (await readBytes(8)).getUint64(0, endian);
          final itemType = GGUFMetadataValueType.fromId(typeId);
          final list = [];
          for (var i = 0; i < len; i++) {
            list.add(await readValue(itemType));
          }
          return list;
      }
    }

    for (var i = 0; i < metadataKvCount; i++) {
      // Read key
      final keyLen = (await readBytes(8)).getUint64(0, endian);
      final keyBytes = Uint8List(keyLen);
      await file.readInto(keyBytes);
      final key = utf8.decode(keyBytes);

      // Read value type
      final typeId = (await readBytes(4)).getUint32(0, endian);
      final valueType = GGUFMetadataValueType.fromId(typeId);

      // Read value
      final value = await readValue(valueType);
      metadata[key] = value;
    }

    final tensorInfos = <GGUFTensorInfo>[];
    for (var i = 0; i < tensorCount; i++) {
      // Name
      final nameLen = (await readBytes(8)).getUint64(0, endian);
      final nameBytes = Uint8List(nameLen);
      await file.readInto(nameBytes);
      final name = utf8.decode(nameBytes);

      // Dimensions
      final nDim = (await readBytes(4)).getUint32(0, endian);
      final shape = <int>[];
      final dimsBytes = await readBytes(nDim * 8);
      for (var j = 0; j < nDim; j++) {
        shape.add(dimsBytes.getUint64(j * 8, endian));
      }

      // Type
      final typeId = (await readBytes(4)).getUint32(0, endian);
      final type = GGUFType.fromId(typeId);

      // Offset
      final offset = (await readBytes(8)).getUint64(0, endian);

      tensorInfos.add(
        GGUFTensorInfo(name: name, shape: shape, type: type, offset: offset),
      );
    }

    // Calculate data offset.
    // It is aligned to a multiple of 'general.alignment' (default 32)
    var cursor = await file.position();
    final alignment = (metadata['general.alignment'] as int?) ?? 32;
    final padding = (alignment - (cursor % alignment)) % alignment;
    final dataOffset = cursor + padding;

    return GGUFHeader(
      version: version,
      tensorCount: tensorCount,
      metadataKvCount: metadataKvCount,
      metadata: metadata,
      tensorInfos: tensorInfos,
      dataOffset: dataOffset,
    );
  }
}
