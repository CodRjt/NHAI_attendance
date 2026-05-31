#pragma once

#include "HybridVisionAuthSpec.hpp"
#include <memory>

namespace margelo::nitro::visionauth {

class VisionAuthImpl : public HybridVisionAuthSpec {
public:
  VisionAuthImpl() : HybridObject(TAG) {}
  ~VisionAuthImpl() override = default;

  double multiply(double a, double b) override;
};

} // namespace margelo::nitro::visionauth
