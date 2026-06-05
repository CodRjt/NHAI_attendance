import { type HybridObject, NitroModules } from 'react-native-nitro-modules';

export interface VisionAuthResult {
  faceDetected: boolean;
  faceScore?: number;
  faceBox?: number[]; // [x, y, w, h]
  leftEyeBox?: number[]; // [x, y, w, h]
  rightEyeBox?: number[]; // [x, y, w, h]
  leftEAR?: number;
  rightEAR?: number;
  avgEAR?: number;
  baseline?: number;
  eyesClosed?: boolean;
  livenessScore?: number;
  embeddingOk?: boolean;
  embedding?: number[];
  yawRatio?: number;
}

export interface VisionAuth extends HybridObject<{ ios: 'c++'; android: 'c++' }> {
  loadModels(blazeFacePath: string, faceLandmarkerPath: string, ghostFacePath: string, antiSpoofingPath: string): boolean;
  analyzeFrame(pixelData: ArrayBuffer, width: number, height: number, bytesPerRow: number): VisionAuthResult;
}

export const VisionAuthModule = NitroModules.createHybridObject<VisionAuth>('VisionAuth');
