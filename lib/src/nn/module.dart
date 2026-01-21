import 'package:tensor/tensor.dart';

abstract class Module {
  String name;

  Module({required this.name});

  void resetParameters();

  Map<String, dynamic> get meta;

  Iterable<Tensor> get parameters;

  // TODO force children to implement this
  Iterable<Tensor> get nonTrainableParameters => [];

  Iterable<Module> get submodules;

  void to_(Device device, {bool cascade = false}) {
    for (final parameter in parameters) {
      if (parameter.device == device) continue;
      parameter.to_(device: device);
    }
    for (final parameter in nonTrainableParameters) {
      if (parameter.device == device) continue;
      parameter.to_(device: device);
    }
    if (cascade) {
      for (final submodule in submodules) {
        submodule.to_(device, cascade: cascade);
      }
    }
  }

  int requiredMemory() {
    int ret = 0;
    for (final parameter in parameters) {
      ret += parameter.elementSize;
    }
    for (final parameter in nonTrainableParameters) {
      ret += parameter.elementSize;
    }
    return ret;
  }

  @override
  String toString() {
    return '$runtimeType(${meta.entries.map((e) => '${e.key}: ${e.value}').join(', ')})';
  }

  static String combineDirs(String prefix, String name) {
    StringBuffer sb = StringBuffer();
    if (prefix.isNotEmpty) {
      sb.write(prefix);
      sb.write('.');
    }
    if (name.isNotEmpty) {
      sb.write(name);
      sb.write('.');
    }
    return sb.toString();
  }
}

extension ModuleExtension on Module {
  Map<String, Tensor> stateDict({bool withName = true}) {
    String prefix = withName ? '$name.' : '';
    final ret = Map.fromEntries(
      parameters.map((e) {
        if (e.name == null) {
          throw Exception(
            'State of module $name of type $runtimeType has no name',
          );
        }
        return MapEntry(prefix + e.name!, e);
      }),
    );
    for (final submodule in submodules) {
      ret.addAll(
        submodule.stateDict().map(
          (key, value) => MapEntry(prefix + key, value),
        ),
      );
    }
    return ret;
  }
}

abstract class SimpleModule implements Module {
  Tensor forward(Tensor input, {required Context context});
}

abstract class EmbeddableModule implements Module {
  Tensor forward(Tensor input, {Tensor? embeds, required Context context});
}

abstract class InplaceModule implements Module {
  void forward_(Tensor x, {required Context context});
}

// TODO support loading and inference/training phases.
class Context {
  bool isTraining;

  Device device;

  final Offloader offloader = Offloader();

  Context({this.isTraining = false, required this.device});

  factory Context.best({bool isTraining = false}) {
    return Context(isTraining: isTraining, device: Device.best());
  }

  void onloadModule(Module module) {
    if (device == Device.cpu) {
      // TODO handle low RAM situations
      return;
    }
    offloader.freeAndLoadModule(module, device);
  }
}

class Offloader {
  final Set<Module> keep = {};

  final Set<Module> modules = {};

  Offloader();

  void freeAndLoadModule(Module module, Device device) {
    if (modules.contains(module)) return;

    int requiredMemory = module.requiredMemory();
    if (requiredMemory > device.freeMemory) {
      if (!freeMemory(requiredMemory, device)) {
        throw Exception('Not enough memory');
      }
    }
    module.to_(device);
    modules.add(module);
  }

  void offloadModule(Module module) {
    if (!modules.contains(module)) return;
    for (final parameter in module.parameters) {
      final device = parameter.device;
      if (device == Device.cpu) continue;
      // TODO pin tensor?
      parameter.to_(device: Device.cpu);
    }
    for (final submodule in module.nonTrainableParameters) {
      final device = submodule.device;
      if (device == Device.cpu) continue;
      // TODO pin tensor?
      submodule.to_(device: Device.cpu);
    }
    modules.remove(module);
    keep.remove(module);
  }

  bool freeMemory(int requiredMemory, Device device) {
    // TODO implement an intelligent algorithm to decide which modules to offload
    // TODO better to offload lowest memory modules first?
    for (final module in modules) {
      if (keep.contains(module)) continue;
      offloadModule(module);
      if (device.freeMemory >= requiredMemory) return true;
    }
    return false;
  }

  void offloadAll() {
    for (final module in modules) {
      offloadModule(module);
    }
    modules.clear();
  }
}
