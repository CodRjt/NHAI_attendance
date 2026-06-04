#include <jni.h>
#include "visionauthOnLoad.hpp"
#include <fbjni/fbjni.h>
#include <NitroModules/HybridObjectRegistry.hpp>
#include "VisionAuthImpl.hpp"

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  return facebook::jni::initialize(vm, []() {
    margelo::nitro::visionauth::registerAllNatives();
    
    margelo::nitro::HybridObjectRegistry::registerHybridObjectConstructor(
      "VisionAuth",
      []() -> std::shared_ptr<margelo::nitro::HybridObject> {
        return std::make_shared<margelo::nitro::visionauth::VisionAuthImpl>();
      }
    );
  });
}