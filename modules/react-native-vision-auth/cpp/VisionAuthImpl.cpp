#include "VisionAuthImpl.hpp"
#include <algorithm>
#include <android/log.h>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <fbjni/fbjni.h>

#define LOGI(...)                                                              \
  __android_log_print(ANDROID_LOG_INFO, "VisionAuth", __VA_ARGS__)
#define LOGE(...)                                                              \
  __android_log_print(ANDROID_LOG_ERROR, "VisionAuth", __VA_ARGS__)

namespace margelo::nitro::visionauth {

VisionAuthImpl::VisionAuthImpl() : HybridObject(TAG), HybridVisionAuthSpec() {}

VisionAuthImpl::~VisionAuthImpl() {
  if (_blazeFaceInterpreter)
    TfLiteInterpreterDelete(_blazeFaceInterpreter);
  if (_faceLandmarkerInterpreter)
    TfLiteInterpreterDelete(_faceLandmarkerInterpreter);
  if (_ghostFaceInterpreter)
    TfLiteInterpreterDelete(_ghostFaceInterpreter);

  if (_blazeFaceModel)
    TfLiteModelDelete(_blazeFaceModel);
  if (_faceLandmarkerModel)
    TfLiteModelDelete(_faceLandmarkerModel);
  if (_ghostFaceModel)
    TfLiteModelDelete(_ghostFaceModel);

  if (_flexDelegate && _deleteFlexDelegate) {
    JNIEnv *env = facebook::jni::Environment::current();
    if (env) {
      _deleteFlexDelegate(env, nullptr, reinterpret_cast<jlong>(_flexDelegate));
    }
  }

  if (_flexLibraryHandle) {
    dlclose(_flexLibraryHandle);
  }
}

void VisionAuthImpl::resizeBilinear(const uint8_t *src, int srcW, int srcH,
                                    int bytesPerRow, int srcChannels, int cropX,
                                    int cropY, int cropW, int cropH, float *dst,
                                    int dstW, int dstH, int dstChannels) {
  for (int y = 0; y < dstH; y++) {
    float srcYf = cropY + (y + 0.5f) * cropH / dstH - 0.5f;
    int y0 = std::max(0, std::min((int)srcYf, srcH - 1));
    int y1 = std::max(0, std::min(y0 + 1, srcH - 1));
    float yFrac = srcYf - y0;
    
    int y0_offset = y0 * bytesPerRow;
    int y1_offset = y1 * bytesPerRow;

    for (int x = 0; x < dstW; x++) {
      float srcXf = cropX + (x + 0.5f) * cropW / dstW - 0.5f;
      int x0 = std::max(0, std::min((int)srcXf, srcW - 1));
      int x1 = std::max(0, std::min(x0 + 1, srcW - 1));
      float xFrac = srcXf - x0;
      
      int x0_offset = x0 * srcChannels;
      int x1_offset = x1 * srcChannels;

      for (int c = 0; c < dstChannels; c++) {
        int srcC = std::min(c, srcChannels - 1);
        float v00 = src[y0_offset + x0_offset + srcC];
        float v01 = src[y0_offset + x1_offset + srcC];
        float v10 = src[y1_offset + x0_offset + srcC];
        float v11 = src[y1_offset + x1_offset + srcC];

        float v0 = v00 + (v01 - v00) * xFrac;
        float v1 = v10 + (v11 - v10) * xFrac;
        float val = v0 + (v1 - v0) * yFrac;

        // Normalize to [-1.0, 1.0] for TFLite
        dst[(y * dstW + x) * dstChannels + c] = val / 127.5f - 1.0f;
      }
    }
  }
}

bool VisionAuthImpl::loadModels(const std::string &blazeFacePath,
                                const std::string &faceLandmarkerPath,
                                const std::string &ghostFacePath) {
  LOGI("Loading BlazeFace model from %s", blazeFacePath.c_str());
  _blazeFaceModel = TfLiteModelCreateFromFile(blazeFacePath.c_str());
  if (!_blazeFaceModel) {
    LOGE("Failed to load BlazeFace");
    return false;
  }

  LOGI("Loading Face Landmarks model from %s", faceLandmarkerPath.c_str());
  _faceLandmarkerModel = TfLiteModelCreateFromFile(faceLandmarkerPath.c_str());
  if (!_faceLandmarkerModel) {
    LOGE("Failed to load Face Landmarker");
    return false;
  }

  LOGI("Loading GhostFace model from %s", ghostFacePath.c_str());
  _ghostFaceModel = TfLiteModelCreateFromFile(ghostFacePath.c_str());
  if (!_ghostFaceModel) {
    LOGE("Failed to load GhostFace");
    return false;
  }

  TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsSetNumThreads(options, 2);

  // GhostFace relies on Select TF Ops, so we dynamically load the Flex Delegate
  _flexLibraryHandle =
      dlopen("libtensorflowlite_flex_jni.so", RTLD_NOW | RTLD_GLOBAL);
  if (_flexLibraryHandle) {
    using CreateFlexDelegate =
        jlong (*)(JNIEnv *, jclass, jobjectArray, jobjectArray);
    auto createFlexDelegate = reinterpret_cast<CreateFlexDelegate>(dlsym(
        _flexLibraryHandle,
        "Java_org_tensorflow_lite_flex_FlexDelegate_nativeCreateDelegate"));
    _deleteFlexDelegate =
        reinterpret_cast<void (*)(JNIEnv *, jclass, jlong)>(dlsym(
            _flexLibraryHandle,
            "Java_org_tensorflow_lite_flex_FlexDelegate_nativeDeleteDelegate"));

    JNIEnv *env = facebook::jni::Environment::current();
    if (env && createFlexDelegate) {
      _flexDelegate = reinterpret_cast<TfLiteDelegate *>(
          createFlexDelegate(env, nullptr, nullptr, nullptr));
      if (_flexDelegate) {
        TfLiteInterpreterOptionsAddDelegate(options, _flexDelegate);
        LOGI("TensorFlow Lite Flex delegate enabled");
      }
    }
  } else {
    LOGE("Failed to load libtensorflowlite_flex_jni.so: %s", dlerror());
  }

  _blazeFaceInterpreter = TfLiteInterpreterCreate(_blazeFaceModel, options);
  _faceLandmarkerInterpreter =
      TfLiteInterpreterCreate(_faceLandmarkerModel, options);
  _ghostFaceInterpreter = TfLiteInterpreterCreate(_ghostFaceModel, options);

  TfLiteInterpreterOptionsDelete(options);

  if (!_blazeFaceInterpreter || !_faceLandmarkerInterpreter ||
      !_ghostFaceInterpreter) {
    LOGE("Failed to create TFLite Interpreters");
    return false;
  }

  if (TfLiteInterpreterAllocateTensors(_blazeFaceInterpreter) != kTfLiteOk ||
      TfLiteInterpreterAllocateTensors(_faceLandmarkerInterpreter) !=
          kTfLiteOk ||
      TfLiteInterpreterAllocateTensors(_ghostFaceInterpreter) != kTfLiteOk) {
    LOGE("Failed to allocate tensors");
    return false;
  }

  _modelsLoaded = true;

  // Pre-compute BlazeFace anchors
  const int NUM_ANCHORS = 896;
  _blazeFaceAnchors.clear();
  _blazeFaceAnchors.reserve(NUM_ANCHORS);
  for (int y = 0; y < 16; ++y) {
    for (int x = 0; x < 16; ++x) {
      float cx = (x + 0.5f) * 8.0f;
      float cy = (y + 0.5f) * 8.0f;
      _blazeFaceAnchors.push_back({cx, cy});
      _blazeFaceAnchors.push_back({cx, cy});
    }
  }
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      float cx = (x + 0.5f) * 16.0f;
      float cy = (y + 0.5f) * 16.0f;
      for (int i = 0; i < 6; ++i) {
        _blazeFaceAnchors.push_back({cx, cy});
      }
    }
  }

  // Find FaceLandmarker output tensor index
  int outCount = TfLiteInterpreterGetOutputTensorCount(_faceLandmarkerInterpreter);
  const TfLiteTensor *landmarksTensor = TfLiteInterpreterGetOutputTensor(_faceLandmarkerInterpreter, 0);
  _faceLandmarksOutputIndex = 0;
  for (int i = 1; i < outCount; i++) {
    const TfLiteTensor *t = TfLiteInterpreterGetOutputTensor(_faceLandmarkerInterpreter, i);
    if (TfLiteTensorByteSize(t) > TfLiteTensorByteSize(landmarksTensor)) {
      landmarksTensor = t;
      _faceLandmarksOutputIndex = i;
    }
  }

  LOGI("All models successfully loaded!");
  return true;
}

bool VisionAuthImpl::runBlazeFace(const uint8_t *rgbData, int width, int height,
                                  int bytesPerRow, int srcChannels, int cropX,
                                  int cropY, int cropW, int cropH, float &outX,
                                  float &outY, float &outW, float &outH,
                                  float &outScore) {
  const int NUM_ANCHORS = 896;
  TfLiteTensor *inputTensor =
      TfLiteInterpreterGetInputTensor(_blazeFaceInterpreter, 0);
  int INPUT_W = TfLiteTensorDim(inputTensor, 2);
  int INPUT_H = TfLiteTensorDim(inputTensor, 1);
  int INPUT_C = TfLiteTensorDim(inputTensor, 3);

  // Zero-copy: Write directly to the interpreter's tensor memory
  float *inputData = (float *)TfLiteTensorData(inputTensor);
  resizeBilinear(rgbData, width, height, bytesPerRow, srcChannels, cropX, cropY,
                 cropW, cropH, inputData, INPUT_W, INPUT_H, INPUT_C);

  if (TfLiteInterpreterInvoke(_blazeFaceInterpreter) != kTfLiteOk) {
    LOGE("BlazeFace inference failed");
    return false;
  }

  const TfLiteTensor *regressorTensor =
      TfLiteInterpreterGetOutputTensor(_blazeFaceInterpreter, 0);
  const TfLiteTensor *classifierTensor =
      TfLiteInterpreterGetOutputTensor(_blazeFaceInterpreter, 1);

  const float *regressors = (const float *)TfLiteTensorData(regressorTensor);
  const float *classifiers = (const float *)TfLiteTensorData(classifierTensor);

  float bestScore = -1.0f;
  int bestIdx = -1;

  for (int i = 0; i < NUM_ANCHORS; i++) {
    float rawScore = classifiers[i];
    float score = 1.0f / (1.0f + expf(-rawScore)); // sigmoid
    if (score > bestScore) {
      bestScore = score;
      bestIdx = i;
    }
  }

  if (bestScore < 0.5f || bestIdx < 0) {
    return false;
  }

  const float *box = regressors + bestIdx * 16;
  float dx = box[0];
  float dy = box[1];
  float w = box[2];
  float h = box[3];

  float anchorX = _blazeFaceAnchors[bestIdx].first;
  float anchorY = _blazeFaceAnchors[bestIdx].second;

  float centerX = dx + anchorX;
  float centerY = dy + anchorY;

  // Convert normalized [0, INPUT_W] coordinates back to original pixel scale
  outX = (centerX / INPUT_W) * cropW + cropX;
  outY = (centerY / INPUT_H) * cropH + cropY;
  outW = (w / INPUT_W) * cropW;
  outH = (h / INPUT_H) * cropH;

  // Convert center coordinates to top-left
  outX = outX - outW / 2.0f;
  outY = outY - outH / 2.0f;

  outX = std::max(0.0f, outX);
  outY = std::max(0.0f, outY);
  outW = std::min(outW, (float)width - outX);
  outH = std::min(outH, (float)height - outY);

  outScore = bestScore;
  return true;
}

static float computeEAR(const float *landmarks, const int *eyeIndices) {
  auto dist = [&](int a, int b) -> float {
    float dx = landmarks[a * 3 + 0] - landmarks[b * 3 + 0];
    float dy = landmarks[a * 3 + 1] - landmarks[b * 3 + 1];
    return sqrtf(dx * dx + dy * dy);
  };

  float vertical1 = dist(eyeIndices[1], eyeIndices[5]);
  float vertical2 = dist(eyeIndices[2], eyeIndices[4]);
  float horizontal = dist(eyeIndices[0], eyeIndices[3]);

  if (horizontal < 1e-6f)
    return 0.0f;
  return (vertical1 + vertical2) / (2.0f * horizontal);
}

bool VisionAuthImpl::runFaceLandmarker(const uint8_t *rgbData, int width,
                                       int height, int bytesPerRow,
                                       int srcChannels, int cropX, int cropY,
                                       int cropW, int cropH, float &leftEAR,
                                       float &rightEAR,
                                       std::vector<double> &leftEyeBoxOut,
                                       std::vector<double> &rightEyeBoxOut) {
  TfLiteTensor *inputTensor =
      TfLiteInterpreterGetInputTensor(_faceLandmarkerInterpreter, 0);
  int INPUT_W = TfLiteTensorDim(inputTensor, 2);
  int INPUT_H = TfLiteTensorDim(inputTensor, 1);
  int INPUT_C = TfLiteTensorDim(inputTensor, 3);

  // Zero-copy: Write directly to the interpreter's tensor memory
  float *inputData = (float *)TfLiteTensorData(inputTensor);
  resizeBilinear(rgbData, width, height, bytesPerRow, srcChannels, cropX, cropY,
                 cropW, cropH, inputData, INPUT_W, INPUT_H, INPUT_C);

  if (TfLiteInterpreterInvoke(_faceLandmarkerInterpreter) != kTfLiteOk) {
    LOGE("Face Landmarker inference failed");
    return false;
  }

  const TfLiteTensor *landmarksTensor =
      TfLiteInterpreterGetOutputTensor(_faceLandmarkerInterpreter, _faceLandmarksOutputIndex);

  const float *landmarks = (const float *)TfLiteTensorData(landmarksTensor);

  static const int leftEyeIdx[6] = {33, 160, 158, 133, 153, 144};
  static const int rightEyeIdx[6] = {362, 385, 387, 263, 373, 380};

  leftEAR = computeEAR(landmarks, leftEyeIdx);
  rightEAR = computeEAR(landmarks, rightEyeIdx);

  auto getEyeBox = [&](const int *indices, std::vector<double> &outBox) {
    float minX = 1e9f, maxX = -1e9f;
    float minY = 1e9f, maxY = -1e9f;
    for (int i = 0; i < 6; i++) {
      float lx = landmarks[indices[i] * 3 + 0];
      float ly = landmarks[indices[i] * 3 + 1];
      minX = std::min(minX, lx);
      maxX = std::max(maxX, lx);
      minY = std::min(minY, ly);
      maxY = std::max(maxY, ly);
    }
    float scaleX = (maxX > 2.0f) ? INPUT_W : 1.0f;
    float scaleY = (maxY > 2.0f) ? INPUT_H : 1.0f;

    float absMinX = cropX + (minX / scaleX) * cropW;
    float absMaxX = cropX + (maxX / scaleX) * cropW;
    float absMinY = cropY + (minY / scaleY) * cropH;
    float absMaxY = cropY + (maxY / scaleY) * cropH;

    // Add a tiny bit of padding to the eye box for better visualization
    float ew = absMaxX - absMinX;
    float eh = absMaxY - absMinY;
    absMinX -= ew * 0.2f;
    absMaxX += ew * 0.2f;
    absMinY -= eh * 0.2f;
    absMaxY += eh * 0.2f;

    outBox = {(double)absMinX, (double)absMinY, (double)(absMaxX - absMinX),
              (double)(absMaxY - absMinY)};
  };

  getEyeBox(leftEyeIdx, leftEyeBoxOut);
  getEyeBox(rightEyeIdx, rightEyeBoxOut);

  return true;
}

bool VisionAuthImpl::runGhostFace(const uint8_t *rgbData, int width, int height,
                                  int bytesPerRow, int srcChannels, int cropX,
                                  int cropY, int cropW, int cropH,
                                  std::vector<double> &embedding) {
  TfLiteTensor *inputTensor =
      TfLiteInterpreterGetInputTensor(_ghostFaceInterpreter, 0);
  int INPUT_W = TfLiteTensorDim(inputTensor, 2);
  int INPUT_H = TfLiteTensorDim(inputTensor, 1);
  int INPUT_C = TfLiteTensorDim(inputTensor, 3);

  // Zero-copy: Write directly to the interpreter's tensor memory
  float *inputData = (float *)TfLiteTensorData(inputTensor);
  resizeBilinear(rgbData, width, height, bytesPerRow, srcChannels, cropX, cropY,
                 cropW, cropH, inputData, INPUT_W, INPUT_H, INPUT_C);

  if (TfLiteInterpreterInvoke(_ghostFaceInterpreter) != kTfLiteOk) {
    LOGE("GhostFace inference failed");
    return false;
  }

  const int EMBEDDING_SIZE = 512;
  const TfLiteTensor *outputTensor =
      TfLiteInterpreterGetOutputTensor(_ghostFaceInterpreter, 0);
  const float *outputData = (const float *)TfLiteTensorData(outputTensor);

  embedding.assign(outputData, outputData + EMBEDDING_SIZE);
  return true;
}

VisionAuthResult
VisionAuthImpl::analyzeFrame(const std::shared_ptr<ArrayBuffer> &pixelData,
                             double width, double height, double bytesPerRow) {
  VisionAuthResult result;

  if (!_modelsLoaded || !pixelData) {
    result.faceDetected = false;
    return result;
  }

  int rawW = static_cast<int>(width);
  int rawH = static_cast<int>(height);
  int bpr = static_cast<int>(bytesPerRow);
  int srcChannels = (bpr >= rawW * 4) ? 4 : 3;

  // The raw Vision Camera buffer on Android is typically landscape (1280x720).
  // In portrait mode, the front camera is rotated 270 degrees and mirrored.
  // We rotate the entire image into a perfect upright 720x1280 buffer!
  int w = rawH; // 720
  int h = rawW; // 1280
  
  size_t requiredSize = w * h * srcChannels;
  if (_uprightBuffer.size() < requiredSize) {
    _uprightBuffer.resize(requiredSize);
  }
  
  const uint8_t *rawData = pixelData->data();

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      // 90 degrees counter-clockwise rotation (sensor is rotated 90 deg
      // clockwise in portrait)
      int rawX = h - 1 - y;
      int rawY = x;
      int dstIdx = (y * w + x) * srcChannels;
      int srcIdx = rawY * bpr + rawX * srcChannels;
      for (int c = 0; c < srcChannels; c++) {
        _uprightBuffer[dstIdx + c] = rawData[srcIdx + c];
      }
    }
  }

  const uint8_t *data = _uprightBuffer.data();
  // We now pass the 'w' (720) and 'h' (1280) and standard bytes per row to the
  // models!
  int uprightBpr = w * srcChannels;

  static int frameCount = 0;
  if (frameCount++ % 10 == 0) {
    LOGI("analyzeFrame: rawW=%d, rawH=%d -> rotated to upright w=%d, h=%d",
         rawW, rawH, w, h);
  }

  // 1. Detect Face (BlazeFace)
  // We feed a perfectly square, aspect-ratio preserving crop (letterboxed) to
  // BlazeFace For a 720x1280 image, maxDim is 1280. cropX = (720-1280)/2 =
  // -280. resizeBilinear will automatically clamp the out-of-bounds pixels to
  // the edge, preserving aspect ratio!
  int maxDim = std::max(w, h);
  int bfCropX = (w - maxDim) / 2;
  int bfCropY = (h - maxDim) / 2;

  float faceX, faceY, faceW, faceH, faceScore;
  bool faceDetected =
      runBlazeFace(data, w, h, uprightBpr, srcChannels, bfCropX, bfCropY,
                   maxDim, maxDim, faceX, faceY, faceW, faceH, faceScore);

  if (!faceDetected) {
    // Only log occasionally to prevent log spam
    static int missCount = 0;
    if (missCount++ % 10 == 0)
      LOGI("No face detected");
    _blinkClosedSeen = false;
    _closedFrameCount = 0;
    _openFrameCount = 0;
    result.faceDetected = false;
    return result;
  }

  LOGI("Face detected! Score: %.2f | Box: x=%.1f, y=%.1f, w=%.1f, h=%.1f",
       faceScore, faceX, faceY, faceW, faceH);

  result.faceDetected = true;
  result.faceScore = faceScore;
  // UI gets the raw, tight bounding box directly from BlazeFace (no padding)
  result.faceBox = std::vector<double>{
      static_cast<double>(faceX), static_cast<double>(faceY),
      static_cast<double>(faceW), static_cast<double>(faceH)};

  // Face Landmarker needs a perfectly square crop, padded by 25% (as per
  // MediaPipe docs)
  float centerX = faceX + faceW / 2.0f;
  float centerY = faceY + faceH / 2.0f;
  float squareSide = std::max(faceW, faceH) * 1.25f;

  int cropX = static_cast<int>(centerX - squareSide / 2.0f);
  int cropY = static_cast<int>(centerY - squareSide / 2.0f);
  int cropW = static_cast<int>(squareSide);
  int cropH = static_cast<int>(squareSide);

  // 2. Liveness Detection (Blink / EAR)
  float leftEAR = 0.0f, rightEAR = 0.0f;
  std::vector<double> leftEyeBox, rightEyeBox;
  bool landmarksOk = runFaceLandmarker(data, w, h, uprightBpr, srcChannels,
                                       cropX, cropY, cropW, cropH, leftEAR,
                                       rightEAR, leftEyeBox, rightEyeBox);

  bool eyesCurrentlyClosed = false;
  bool blinkDetected = false;

  if (landmarksOk) {
    float avgEAR = (leftEAR + rightEAR) / 2.0f;

    result.leftEAR = leftEAR;
    result.rightEAR = rightEAR;
    result.avgEAR = avgEAR;
    result.baseline = _earOpenBaseline;
    result.leftEyeBox = leftEyeBox;
    result.rightEyeBox = rightEyeBox;

    if (std::isfinite(leftEAR) && std::isfinite(rightEAR) && leftEAR > 0.02f &&
        leftEAR < 1.0f && rightEAR > 0.02f && rightEAR < 1.0f) {
      if (_earOpenBaseline <= 0.0f) {
        _earOpenBaseline = avgEAR;
      }
      if (_leftEarOpenBaseline <= 0.0f) {
        _leftEarOpenBaseline = leftEAR;
      }
      if (_rightEarOpenBaseline <= 0.0f) {
        _rightEarOpenBaseline = rightEAR;
      }

      const float leftCloseThreshold =
          std::max(0.10f, _leftEarOpenBaseline * 0.75f);
      const float rightCloseThreshold =
          std::max(0.10f, _rightEarOpenBaseline * 0.75f);
      const float leftOpenThreshold =
          std::max(leftCloseThreshold + 0.02f, _leftEarOpenBaseline * 0.85f);
      const float rightOpenThreshold =
          std::max(rightCloseThreshold + 0.02f, _rightEarOpenBaseline * 0.85f);

      // Use OR instead of AND. Often one eye's landmarks track poorly during a
      // fast blink. If either eye drops below 75% of its open baseline, we
      // count it as a blink/wink.
      eyesCurrentlyClosed =
          leftEAR < leftCloseThreshold || rightEAR < rightCloseThreshold;
      const bool eyesCurrentlyOpen =
          leftEAR > leftOpenThreshold && rightEAR > rightOpenThreshold;

      if (!eyesCurrentlyClosed && eyesCurrentlyOpen && !_blinkClosedSeen) {
        _earOpenBaseline = (_earOpenBaseline * 0.98f) + (avgEAR * 0.02f);
        _leftEarOpenBaseline =
            (_leftEarOpenBaseline * 0.98f) + (leftEAR * 0.02f);
        _rightEarOpenBaseline =
            (_rightEarOpenBaseline * 0.98f) + (rightEAR * 0.02f);
      }

      if (_blinkCooldownFrames > 0) {
        _blinkCooldownFrames--;
      }

      if (eyesCurrentlyClosed) {
        _closedFrameCount++;
        _openFrameCount = 0;
        if (_closedFrameCount >= 2) {
          _blinkClosedSeen = true;
        }
      } else if (eyesCurrentlyOpen) {
        _openFrameCount++;
        if (_blinkClosedSeen && _openFrameCount >= 1 &&
            _blinkCooldownFrames == 0) {
          blinkDetected = true;
          _blinkCooldownFrames = 8;
        }
        _blinkClosedSeen = false;
        _closedFrameCount = 0;
      }

      LOGI("Landmarks OK! leftEAR: %.3f | rightEAR: %.3f | avgEAR: %.3f | "
           "baseline: %.3f | L<%.3f R<%.3f | L>%.3f R>%.3f | closedFrames=%d | "
           "closed=%d | blink=%d",
           leftEAR, rightEAR, avgEAR, _earOpenBaseline, leftCloseThreshold,
           rightCloseThreshold, leftOpenThreshold, rightOpenThreshold,
           _closedFrameCount, eyesCurrentlyClosed, blinkDetected);
    } else {
      LOGE("Invalid EAR values: leftEAR=%.3f rightEAR=%.3f avgEAR=%.3f",
           leftEAR, rightEAR, avgEAR);
    }

    result.eyesClosed = blinkDetected;
  } else {
    LOGE("Face Landmarker failed to run or found no landmarks");
  }

  // 3. Facial Recognition (GhostFace)
  static int ghostFaceCounter = 0;
  bool runGhostFaceModel =
      (!eyesCurrentlyClosed) && (ghostFaceCounter++ % 5 == 0);

  std::vector<double> embedding;
  bool embeddingOk = false;

  if (runGhostFaceModel) {
    embeddingOk = runGhostFace(data, w, h, uprightBpr, srcChannels, cropX,
                               cropY, cropW, cropH, embedding);
  }

  if (embeddingOk) {
    result.embeddingOk = true;
    result.embedding = embedding;
  }

  return result;
}

} // namespace margelo::nitro::visionauth
