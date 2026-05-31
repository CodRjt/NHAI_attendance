import { type HybridObject, NitroModules } from 'react-native-nitro-modules';

export interface VisionAuthResult {
  faceDetected: boolean;
  faceScore?: number;
  faceBox?: number[]; // [x, y, w, h]
  leftEAR?: number;
  rightEAR?: number;
  avgEAR?: number;
  eyesClosed?: boolean;
  embeddingOk?: boolean;
  embedding?: number[];
}

export interface VisionAuth extends HybridObject<{ ios: 'c++'; android: 'c++' }> {
  loadModels(blazeFacePath: string, faceLandmarkerPath: string, ghostFacePath: string): boolean;
  analyzeFrame(pixelData: ArrayBuffer, width: number, height: number, bytesPerRow: number): VisionAuthResult;
}

export const VisionAuthModule = NitroModules.createHybridObject<VisionAuth>('VisionAuth');
