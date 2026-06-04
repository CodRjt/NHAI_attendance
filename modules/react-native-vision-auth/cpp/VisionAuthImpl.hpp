#pragma once

#include "HybridVisionAuthSpec.hpp"
#include <memory>
#include <string>
#include <vector>

#include <jni.h>
#include <tensorflow/lite/c/c_api.h>

namespace margelo::nitro::visionauth {

class VisionAuthImpl final : public HybridVisionAuthSpec {
private:
  // Models
  TfLiteModel *_blazeFaceModel = nullptr;
  TfLiteModel *_faceLandmarkerModel = nullptr;
  TfLiteModel *_ghostFaceModel = nullptr;

  // Interpreters
  TfLiteInterpreter *_blazeFaceInterpreter = nullptr;
  TfLiteInterpreter *_faceLandmarkerInterpreter = nullptr;
  TfLiteInterpreter *_ghostFaceInterpreter = nullptr;

  // Flex Delegate handling
  TfLiteDelegate *_flexDelegate = nullptr;
  void *_flexLibraryHandle = nullptr;
  void (*_deleteFlexDelegate)(JNIEnv *, jclass, jlong) = nullptr;

  bool _modelsLoaded = false;

  // Blink detection state. EAR is model/camera dependent, so use an adaptive
  // open-eye baseline instead of one global threshold.
  float _earOpenBaseline = 0.0f;
  float _leftEarOpenBaseline = 0.0f;
  float _rightEarOpenBaseline = 0.0f;
  bool _blinkClosedSeen = false;
  int _closedFrameCount = 0;
  int _openFrameCount = 0;
  int _blinkCooldownFrames = 0;

  std::vector<uint8_t> _uprightBuffer;
  std::vector<std::pair<float, float>> _blazeFaceAnchors;
  int _faceLandmarksOutputIndex = -1;

  // Internal helpers
  bool runBlazeFace(const uint8_t *rgbData, int width, int height,
                    int bytesPerRow, int srcChannels, int cropX, int cropY,
                    int cropW, int cropH, float &outX, float &outY, float &outW,
                    float &outH, float &outScore);

  bool runFaceLandmarker(const uint8_t *rgbData, int width, int height,
                         int bytesPerRow, int srcChannels, int cropX, int cropY,
                         int cropW, int cropH, float &leftEAR, float &rightEAR,
                         std::vector<double> &leftEyeBoxOut,
                         std::vector<double> &rightEyeBoxOut);

  bool runGhostFace(const uint8_t *rgbData, int width, int height,
                    int bytesPerRow, int srcChannels, int cropX, int cropY,
                    int cropW, int cropH, std::vector<double> &embedding);

  void resizeBilinear(const uint8_t *src, int srcW, int srcH, int bytesPerRow,
                      int srcChannels, int cropX, int cropY, int cropW,
                      int cropH, float *dst, int dstW, int dstH,
                      int dstChannels);

public:
  VisionAuthImpl();
  ~VisionAuthImpl() override;

  bool loadModels(const std::string &blazeFacePath,
                  const std::string &faceLandmarkerPath,
                  const std::string &ghostFacePath) override;
  VisionAuthResult analyzeFrame(const std::shared_ptr<ArrayBuffer> &pixelData,
                                double width, double height,
                                double bytesPerRow) override;
};

} // namespace margelo::nitro::visionauth
