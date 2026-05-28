#include "VisionAuthImpl.h"

namespace facebook::react {

VisionAuthImpl::VisionAuthImpl(
  std::shared_ptr<CallInvoker> jsInvoker
)
  : NativeVisionAuthCxxSpec(std::move(jsInvoker)) {}

double VisionAuthImpl::multiply(
  jsi::Runtime& rt,
  double a,
  double b
) {
  return a * b;
}

}
