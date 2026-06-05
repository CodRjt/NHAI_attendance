#pragma once

#include "HybridVisionAuthSpec.hpp"
#include <memory>
#include <string>
#include <vector>

#include <tensorflow/lite/c/c_api.h>

namespace margelo::nitro::visionauth {

class VisionAuthImpl final : public HybridVisionAuthSpec {
private:
  // Models
  TfLiteModel *_blazeFaceModel = nullptr;
  TfLiteModel *_faceLandmarkerModel = nullptr;
  TfLiteModel *_faceRecognitionModel = nullptr;
  TfLiteModel *_antiSpoofingModel = nullptr;

  // Interpreters
  TfLiteInterpreter *_blazeFaceInterpreter = nullptr;
  TfLiteInterpreter *_faceLandmarkerInterpreter = nullptr;
  TfLiteInterpreter *_faceRecognitionInterpreter = nullptr;
  TfLiteInterpreter *_antiSpoofingInterpreter = nullptr;

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

  bool runFaceRecognition(const uint8_t *rgbData, int width, int height,
                          int bytesPerRow, int srcChannels, int cropX,
                          int cropY, int cropW, int cropH,
                          std::vector<double> &embedding);

  bool runAntiSpoofing(const uint8_t *rgbData, int width, int height,
                       int bytesPerRow, int srcChannels, int cropX, int cropY,
                       int cropW, int cropH, float &livenessScore);

  void resizeBilinear(const uint8_t *src, int srcW, int srcH, int bytesPerRow,
                      int srcChannels, int cropX, int cropY, int cropW,
                      int cropH, float *dst, int dstW, int dstH,
                      int dstChannels, int normMode = 1, bool bgr = false);

public:
  VisionAuthImpl();
  ~VisionAuthImpl() override;

  bool loadModels(const std::string &blazeFacePath,
                  const std::string &faceLandmarkerPath,
                  const std::string &ghostFacePath,
                  const std::string &antiSpoofingPath) override;
  VisionAuthResult analyzeFrame(const std::shared_ptr<ArrayBuffer> &pixelData,
                                double width, double height,
                                double bytesPerRow) override;
};

} // namespace margelo::nitro::visionauth
