#pragma once

#include <VisionAuthSpecJSI.h>

#include <memory>

namespace facebook::react {

class VisionAuthImpl
  : public NativeVisionAuthCxxSpec<VisionAuthImpl> {
public:
  VisionAuthImpl(std::shared_ptr<CallInvoker> jsInvoker);

  double multiply(jsi::Runtime& rt, double a, double b);
};

}
