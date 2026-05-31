#include "VisionAuthImpl.hpp"
#include <android/log.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <dlfcn.h>
#include <fbjni/fbjni.h>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "VisionAuth", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "VisionAuth", __VA_ARGS__)

namespace margelo::nitro::visionauth {

VisionAuthImpl::VisionAuthImpl() : HybridObject(TAG), HybridVisionAuthSpec() {}

VisionAuthImpl::~VisionAuthImpl() {
  if (_blazeFaceInterpreter) TfLiteInterpreterDelete(_blazeFaceInterpreter);
  if (_faceLandmarkerInterpreter) TfLiteInterpreterDelete(_faceLandmarkerInterpreter);
  if (_ghostFaceInterpreter) TfLiteInterpreterDelete(_ghostFaceInterpreter);

  if (_blazeFaceModel) TfLiteModelDelete(_blazeFaceModel);
  if (_faceLandmarkerModel) TfLiteModelDelete(_faceLandmarkerModel);
  if (_ghostFaceModel) TfLiteModelDelete(_ghostFaceModel);

  if (_flexDelegate && _deleteFlexDelegate) {
    JNIEnv* env = facebook::jni::Environment::current();
    if (env) {
      _deleteFlexDelegate(env, nullptr, reinterpret_cast<jlong>(_flexDelegate));
    }
  }

  if (_flexLibraryHandle) {
    dlclose(_flexLibraryHandle);
  }
}

void VisionAuthImpl::resizeBilinear(
  const uint8_t* src, int srcW, int srcH, int bytesPerRow, int srcChannels,
  int cropX, int cropY, int cropW, int cropH,
  float* dst, int dstW, int dstH, int dstChannels
) {
  for (int y = 0; y < dstH; y++) {
    for (int x = 0; x < dstW; x++) {
      float srcXf = cropX + (x + 0.5f) * cropW / dstW - 0.5f;
      float srcYf = cropY + (y + 0.5f) * cropH / dstH - 0.5f;

      int x0 = std::max(0, std::min((int)srcXf, srcW - 1));
      int y0 = std::max(0, std::min((int)srcYf, srcH - 1));
      int x1 = std::max(0, std::min(x0 + 1, srcW - 1));
      int y1 = std::max(0, std::min(y0 + 1, srcH - 1));

      float xFrac = srcXf - x0;
      float yFrac = srcYf - y0;

      for (int c = 0; c < dstChannels; c++) {
        int srcC = std::min(c, srcChannels - 1);
        float v00 = src[y0 * bytesPerRow + x0 * srcChannels + srcC];
        float v01 = src[y0 * bytesPerRow + x1 * srcChannels + srcC];
        float v10 = src[y1 * bytesPerRow + x0 * srcChannels + srcC];
        float v11 = src[y1 * bytesPerRow + x1 * srcChannels + srcC];

        float v0 = v00 + (v01 - v00) * xFrac;
        float v1 = v10 + (v11 - v10) * xFrac;
        float val = v0 + (v1 - v0) * yFrac;

        // Normalize to [-1.0, 1.0] for TFLite
        dst[(y * dstW + x) * dstChannels + c] = val / 127.5f - 1.0f;
      }
    }
  }
}

bool VisionAuthImpl::loadModels(
  const std::string& blazeFacePath,
  const std::string& faceLandmarkerPath,
  const std::string& ghostFacePath
) {
  LOGI("Loading BlazeFace model from %s", blazeFacePath.c_str());
  _blazeFaceModel = TfLiteModelCreateFromFile(blazeFacePath.c_str());
  if (!_blazeFaceModel) { LOGE("Failed to load BlazeFace"); return false; }

  LOGI("Loading Face Landmarks model from %s", faceLandmarkerPath.c_str());
  _faceLandmarkerModel = TfLiteModelCreateFromFile(faceLandmarkerPath.c_str());
  if (!_faceLandmarkerModel) { LOGE("Failed to load Face Landmarker"); return false; }

  LOGI("Loading GhostFace model from %s", ghostFacePath.c_str());
  _ghostFaceModel = TfLiteModelCreateFromFile(ghostFacePath.c_str());
  if (!_ghostFaceModel) { LOGE("Failed to load GhostFace"); return false; }

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsSetNumThreads(options, 2);

  // GhostFace relies on Select TF Ops, so we dynamically load the Flex Delegate
  _flexLibraryHandle = dlopen("libtensorflowlite_flex_jni.so", RTLD_NOW | RTLD_GLOBAL);
  if (_flexLibraryHandle) {
    using CreateFlexDelegate = jlong (*)(JNIEnv*, jclass, jobjectArray, jobjectArray);
    auto createFlexDelegate = reinterpret_cast<CreateFlexDelegate>(
      dlsym(_flexLibraryHandle, "Java_org_tensorflow_lite_flex_FlexDelegate_nativeCreateDelegate"));
    _deleteFlexDelegate = reinterpret_cast<void (*)(JNIEnv*, jclass, jlong)>(
      dlsym(_flexLibraryHandle, "Java_org_tensorflow_lite_flex_FlexDelegate_nativeDeleteDelegate"));

    JNIEnv* env = facebook::jni::Environment::current();
    if (env && createFlexDelegate) {
      _flexDelegate = reinterpret_cast<TfLiteDelegate*>(createFlexDelegate(env, nullptr, nullptr, nullptr));
      if (_flexDelegate) {
        TfLiteInterpreterOptionsAddDelegate(options, _flexDelegate);
        LOGI("TensorFlow Lite Flex delegate enabled");
      }
    }
  } else {
    LOGE("Failed to load libtensorflowlite_flex_jni.so: %s", dlerror());
  }

  _blazeFaceInterpreter = TfLiteInterpreterCreate(_blazeFaceModel, options);
  _faceLandmarkerInterpreter = TfLiteInterpreterCreate(_faceLandmarkerModel, options);
  _ghostFaceInterpreter = TfLiteInterpreterCreate(_ghostFaceModel, options);

  TfLiteInterpreterOptionsDelete(options);

  if (!_blazeFaceInterpreter || !_faceLandmarkerInterpreter || !_ghostFaceInterpreter) {
    LOGE("Failed to create TFLite Interpreters");
    return false;
  }

  if (TfLiteInterpreterAllocateTensors(_blazeFaceInterpreter) != kTfLiteOk ||
      TfLiteInterpreterAllocateTensors(_faceLandmarkerInterpreter) != kTfLiteOk ||
      TfLiteInterpreterAllocateTensors(_ghostFaceInterpreter) != kTfLiteOk) {
    LOGE("Failed to allocate tensors");
    return false;
  }

  _modelsLoaded = true;
  LOGI("All models successfully loaded!");
  return true;
}

bool VisionAuthImpl::runBlazeFace(
  const uint8_t* rgbData, int width, int height, int bytesPerRow, int srcChannels,
  float& outX, float& outY, float& outW, float& outH, float& outScore
) {
  const int NUM_ANCHORS = 896;
  TfLiteTensor* inputTensor = TfLiteInterpreterGetInputTensor(_blazeFaceInterpreter, 0);
  int INPUT_W = TfLiteTensorDim(inputTensor, 2);
  int INPUT_H = TfLiteTensorDim(inputTensor, 1);
  int INPUT_C = TfLiteTensorDim(inputTensor, 3);

  std::vector<float> inputBuf(INPUT_W * INPUT_H * INPUT_C);
  resizeBilinear(rgbData, width, height, bytesPerRow, srcChannels, 0, 0, width, height,
                 inputBuf.data(), INPUT_W, INPUT_H, INPUT_C);

  if (TfLiteTensorCopyFromBuffer(inputTensor, inputBuf.data(), inputBuf.size() * sizeof(float)) != kTfLiteOk) {
    LOGE("BlazeFace: TfLiteTensorCopyFromBuffer failed!");
    return false;
  }

  if (TfLiteInterpreterInvoke(_blazeFaceInterpreter) != kTfLiteOk) {
    LOGE("BlazeFace inference failed");
    return false;
  }

  const TfLiteTensor* regressorTensor = TfLiteInterpreterGetOutputTensor(_blazeFaceInterpreter, 0);
  const TfLiteTensor* classifierTensor = TfLiteInterpreterGetOutputTensor(_blazeFaceInterpreter, 1);

  const float* regressors = (const float*)TfLiteTensorData(regressorTensor);
  const float* classifiers = (const float*)TfLiteTensorData(classifierTensor);

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

  // Generate anchors exactly as MediaPipe SSD Anchor Calculator for BlazeFace
  std::vector<std::pair<float, float>> anchors;
  anchors.reserve(NUM_ANCHORS);
  // Layer 1: 16x16 grid, stride 8, 2 anchors per cell
  for (int y = 0; y < 16; ++y) {
    for (int x = 0; x < 16; ++x) {
      float cx = (x + 0.5f) * 8.0f;
      float cy = (y + 0.5f) * 8.0f;
      anchors.push_back({cx, cy});
      anchors.push_back({cx, cy});
    }
  }
  // Layer 2: 8x8 grid, stride 16, 6 anchors per cell
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      float cx = (x + 0.5f) * 16.0f;
      float cy = (y + 0.5f) * 16.0f;
      for (int i = 0; i < 6; ++i) {
        anchors.push_back({cx, cy});
      }
    }
  }

  const float* box = regressors + bestIdx * 16;
  float dx = box[0];
  float dy = box[1];
  float w  = box[2];
  float h  = box[3];

  float anchorX = anchors[bestIdx].first;
  float anchorY = anchors[bestIdx].second;

  float centerX = dx + anchorX;
  float centerY = dy + anchorY;

  // Convert normalized [0, INPUT_W] coordinates back to original pixel scale
  outX = (centerX / INPUT_W) * width;
  outY = (centerY / INPUT_H) * height;
  outW = (w / INPUT_W) * width;
  outH = (h / INPUT_H) * height;

  // Convert center coordinates to top-left
  outX = outX - outW / 2.0f;
  outY = outY - outH / 2.0f;

  // Add 20% padding to the bounding box so FaceLandmarker gets the whole head!
  float padW = outW * 0.20f;
  float padH = outH * 0.20f;
  outX -= padW / 2.0f;
  outY -= padH / 2.0f;
  outW += padW;
  outH += padH;

  outX = std::max(0.0f, outX);
  outY = std::max(0.0f, outY);
  outW = std::min(outW, (float)width - outX);
  outH = std::min(outH, (float)height - outY);

  outScore = bestScore;
  return true;
}

static float computeEAR(const float* landmarks, const int* eyeIndices) {
  auto dist = [&](int a, int b) -> float {
    float dx = landmarks[a * 3 + 0] - landmarks[b * 3 + 0];
    float dy = landmarks[a * 3 + 1] - landmarks[b * 3 + 1];
    return sqrtf(dx * dx + dy * dy);
  };

  float vertical1 = dist(eyeIndices[1], eyeIndices[5]);
  float vertical2 = dist(eyeIndices[2], eyeIndices[4]);
  float horizontal = dist(eyeIndices[0], eyeIndices[3]);

  if (horizontal < 1e-6f) return 0.0f;
  return (vertical1 + vertical2) / (2.0f * horizontal);
}

bool VisionAuthImpl::runFaceLandmarker(
  const uint8_t* rgbData, int width, int height, int bytesPerRow, int srcChannels,
  int cropX, int cropY, int cropW, int cropH,
  float& leftEAR, float& rightEAR
) {
  TfLiteTensor* inputTensor = TfLiteInterpreterGetInputTensor(_faceLandmarkerInterpreter, 0);
  int INPUT_W = TfLiteTensorDim(inputTensor, 2);
  int INPUT_H = TfLiteTensorDim(inputTensor, 1);
  int INPUT_C = TfLiteTensorDim(inputTensor, 3);

  std::vector<float> inputBuf(INPUT_W * INPUT_H * INPUT_C);
  resizeBilinear(rgbData, width, height, bytesPerRow, srcChannels, cropX, cropY, cropW, cropH,
                 inputBuf.data(), INPUT_W, INPUT_H, INPUT_C);

  if (TfLiteTensorCopyFromBuffer(inputTensor, inputBuf.data(), inputBuf.size() * sizeof(float)) != kTfLiteOk) {
    LOGE("FaceLandmarker: TfLiteTensorCopyFromBuffer failed! Check model input size.");
    return false;
  }

  if (TfLiteInterpreterInvoke(_faceLandmarkerInterpreter) != kTfLiteOk) {
    LOGE("Face Landmarker inference failed");
    return false;
  }

  // The model may have multiple outputs (e.g., face score and landmarks). 
  // We want the landmarks tensor, which is the largest one (1404 floats).
  int outCount = TfLiteInterpreterGetOutputTensorCount(_faceLandmarkerInterpreter);
  const TfLiteTensor* landmarksTensor = TfLiteInterpreterGetOutputTensor(_faceLandmarkerInterpreter, 0);
  for (int i = 1; i < outCount; i++) {
    const TfLiteTensor* t = TfLiteInterpreterGetOutputTensor(_faceLandmarkerInterpreter, i);
    if (TfLiteTensorByteSize(t) > TfLiteTensorByteSize(landmarksTensor)) {
      landmarksTensor = t;
    }
  }

  const float* landmarks = (const float*)TfLiteTensorData(landmarksTensor);

  static const int leftEyeIdx[6]  = {33, 160, 158, 133, 153, 144};
  static const int rightEyeIdx[6] = {362, 385, 387, 263, 373, 380};

  leftEAR = computeEAR(landmarks, leftEyeIdx);
  rightEAR = computeEAR(landmarks, rightEyeIdx);

  return true;
}

bool VisionAuthImpl::runGhostFace(
  const uint8_t* rgbData, int width, int height, int bytesPerRow, int srcChannels,
  int cropX, int cropY, int cropW, int cropH,
  std::vector<double>& embedding
) {
  TfLiteTensor* inputTensor = TfLiteInterpreterGetInputTensor(_ghostFaceInterpreter, 0);
  int INPUT_W = TfLiteTensorDim(inputTensor, 2);
  int INPUT_H = TfLiteTensorDim(inputTensor, 1);
  int INPUT_C = TfLiteTensorDim(inputTensor, 3);

  std::vector<float> inputBuf(INPUT_W * INPUT_H * INPUT_C);
  resizeBilinear(rgbData, width, height, bytesPerRow, srcChannels, cropX, cropY, cropW, cropH,
                 inputBuf.data(), INPUT_W, INPUT_H, INPUT_C);

  if (TfLiteTensorCopyFromBuffer(inputTensor, inputBuf.data(), inputBuf.size() * sizeof(float)) != kTfLiteOk) {
    LOGE("GhostFace: TfLiteTensorCopyFromBuffer failed!");
    return false;
  }

  if (TfLiteInterpreterInvoke(_ghostFaceInterpreter) != kTfLiteOk) {
    LOGE("GhostFace inference failed");
    return false;
  }

  const int EMBEDDING_SIZE = 512;
  const TfLiteTensor* outputTensor = TfLiteInterpreterGetOutputTensor(_ghostFaceInterpreter, 0);
  const float* outputData = (const float*)TfLiteTensorData(outputTensor);

  embedding.assign(outputData, outputData + EMBEDDING_SIZE);
  return true;
}

VisionAuthResult VisionAuthImpl::analyzeFrame(
  const std::shared_ptr<ArrayBuffer>& pixelData,
  double width,
  double height,
  double bytesPerRow
) {
  VisionAuthResult result;

  if (!_modelsLoaded || !pixelData) {
    result.faceDetected = false;
    return result;
  }

  int w = static_cast<int>(width);
  int h = static_cast<int>(height);
  int bpr = static_cast<int>(bytesPerRow);
  int srcChannels = (bpr >= w * 4) ? 4 : 3;
  
  static int frameCount = 0;
  if (frameCount++ % 10 == 0) {
    LOGI("analyzeFrame: w=%d, h=%d, bpr=%d, computedChannels=%d", w, h, bpr, srcChannels);
  }

  const uint8_t* data = pixelData->data();

  // 1. Detect Face (BlazeFace)
  float faceX, faceY, faceW, faceH, faceScore;
  bool faceDetected = runBlazeFace(data, w, h, bpr, srcChannels, faceX, faceY, faceW, faceH, faceScore);

  if (!faceDetected) {
    // Only log occasionally to prevent log spam
    static int missCount = 0;
    if (missCount++ % 10 == 0) LOGI("No face detected");
    result.faceDetected = false;
    return result;
  }

  LOGI("Face detected! Score: %.2f | Box: x=%.1f, y=%.1f, w=%.1f, h=%.1f", faceScore, faceX, faceY, faceW, faceH);

  result.faceDetected = true;
  result.faceScore = faceScore;
  result.faceBox = std::vector<double>{
    static_cast<double>(faceX),
    static_cast<double>(faceY),
    static_cast<double>(faceW),
    static_cast<double>(faceH)
  };

  int cropX = static_cast<int>(faceX);
  int cropY = static_cast<int>(faceY);
  int cropW = static_cast<int>(faceW);
  int cropH = static_cast<int>(faceH);

  // 2. Liveness Detection (Blink / EAR)
  float leftEAR = 0.0f, rightEAR = 0.0f;
  bool landmarksOk = runFaceLandmarker(data, w, h, bpr, srcChannels, cropX, cropY, cropW, cropH, leftEAR, rightEAR);
  
  bool eyesClosed = false;

  if (landmarksOk) {
    float avgEAR = (leftEAR + rightEAR) / 2.0f;
    LOGI("Landmarks OK! leftEAR: %.3f | rightEAR: %.3f | avgEAR: %.3f", leftEAR, rightEAR, avgEAR);
    
    result.leftEAR = leftEAR;
    result.rightEAR = rightEAR;
    result.avgEAR = avgEAR;
    // Standard Eye Aspect Ratio (EAR) threshold for a blink is < 0.21.
    // Now that the camera runs at 60 FPS, it will easily capture the eye fully shut!
    eyesClosed = (avgEAR < 0.21f);
    result.eyesClosed = eyesClosed;
  } else {
    LOGE("Face Landmarker failed to run or found no landmarks");
  }

  // 3. Facial Recognition (GhostFace)
  static int ghostFaceCounter = 0;
  bool runGhostFaceModel = (!eyesClosed) && (ghostFaceCounter++ % 5 == 0);

  std::vector<double> embedding;
  bool embeddingOk = false;
  
  if (runGhostFaceModel) {
    embeddingOk = runGhostFace(data, w, h, bpr, srcChannels, cropX, cropY, cropW, cropH, embedding);
  }

  if (embeddingOk) {
    result.embeddingOk = true;
    result.embedding = embedding;
  }

  return result;
}

} // namespace margelo::nitro::visionauth
