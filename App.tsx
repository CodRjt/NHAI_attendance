import React, { useEffect } from 'react';
import { StyleSheet, Text, View, ActivityIndicator } from 'react-native';
import { SafeAreaView, SafeAreaProvider } from 'react-native-safe-area-context';
import {
  useCameraDevice,
  useCameraPermission,
  useFrameOutput,
  Camera,
} from 'react-native-vision-camera';
import { VisionAuthModule, type VisionAuthResult } from 'react-native-vision-auth';
import { runOnJS } from 'react-native-worklets';
import * as RNFS from 'react-native-fs';
import { createMMKV } from 'react-native-mmkv';

const storage = createMMKV();

function AppContent() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('front');
  const [modelsReady, setModelsReady] = React.useState(false);

  useEffect(() => {
    async function setup() {
      if (!hasPermission) {
        await requestPermission();
      }

      try {
        const models = [
          'blaze_face_short_range.tflite',
          'face_landmarks_detector.tflite',
          'ghostface.tflite',
        ];
        const paths = [];
        for (const model of models) {
          const destPath = `${RNFS.DocumentDirectoryPath}/${model}`;
          const exists = await RNFS.exists(destPath);
          if (!exists) {
            console.log(`Copying ${model} from assets...`);
            await RNFS.copyFileAssets(`models/${model}`, destPath);
          }
          paths.push(destPath);
        }

        const success = VisionAuthModule.loadModels(
          paths[0],
          paths[1],
          paths[2],
        );
        if (success) {
          setModelsReady(true);
        } else {
          console.error(
            'Failed to load models in C++ (loadModels returned false)',
          );
        }
      } catch (e) {
        console.error('Error setting up models:', e);
      }
    }
    setup();
  }, [hasPermission, requestPermission]);

  const [livenessStatus, setLivenessStatus] = React.useState(
    'Please look at the camera',
  );

  const handleFrameResult = React.useCallback((result: VisionAuthResult) => {
    try {
      if (!result.faceDetected) {
        setLivenessStatus('No face detected');
      } else if (result.eyesClosed) {
        setLivenessStatus('Blink detected! Liveness confirmed.');
        if (result.embeddingOk && result.embedding) {
          // Save embedding for later use
          console.log(
            'Got valid face embedding:',
            result.embedding.length,
            'dims',
          );
          storage.set('user_face_embedding', JSON.stringify(result.embedding));
        }
      } else {
        setLivenessStatus('Please Blink');
      }
    } catch (e) {
      console.error('Frame parsing error', e);
    }
  }, []);

  const frameOutput = useFrameOutput({
    pixelFormat: 'rgb',
    onFrame: frame => {
      'worklet';
      if (!(globalThis as any).lastProcessTime)
        (globalThis as any).lastProcessTime = 0;
      const now = Date.now();

      // Throttle to ~30 FPS (33ms) so we don't miss the 100ms blink window!
      if (now - (globalThis as any).lastProcessTime >= 33) {
        (globalThis as any).lastProcessTime = now;

        if (frame.hasPixelBuffer) {
          const buffer = frame.getPixelBuffer();
          // Call Nitro Module directly on the Worklet thread!
          try {
            const result = VisionAuthModule.analyzeFrame(
              buffer,
              frame.width,
              frame.height,
              frame.bytesPerRow,
            );
            // Run state update on JS thread with the typed object!
            runOnJS(handleFrameResult)(result);
          } catch (e) {
            console.error('Error calling Nitro Module:', e);
          }
        }
      }
      frame.dispose();
    },
  });

  if (!hasPermission) {
    return (
      <SafeAreaView style={styles.container}>
        <ActivityIndicator size="large" color="#00ff00" />
        <Text style={styles.text}>Requesting Camera Permission...</Text>
      </SafeAreaView>
    );
  }

  if (device == null) {
    return (
      <SafeAreaView style={styles.container}>
        <Text style={styles.text}>No Front Camera Found</Text>
      </SafeAreaView>
    );
  }

  return (
    <View style={styles.container}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={modelsReady}
        outputs={[frameOutput]}
      />
      {/* Overlay UI */}
      <SafeAreaView style={styles.overlay}>
        <View style={styles.header}>
          <Text style={styles.title}>Liveness Detection</Text>
          <Text style={styles.subtitle}>
            Position your face within the frame
          </Text>
        </View>

        {/* Face Outline Guide */}
        <View style={styles.faceOutline} />

        <View style={styles.footerContainer}>
          <View style={styles.footer}>
            <Text style={styles.instruction}>{livenessStatus}</Text>
          </View>
          <Text style={styles.devText}>
            Models loaded: {modelsReady ? 'Yes' : 'No'}
          </Text>
        </View>
      </SafeAreaView>
    </View>
  );
}

export default function App() {
  return (
    <SafeAreaProvider>
      <AppContent />
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
  },
  text: {
    color: '#fff',
    fontSize: 18,
    marginTop: 10,
  },
  overlay: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    left: 0,
    right: 0,
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 40,
  },
  header: {
    alignItems: 'center',
    marginTop: 20,
  },
  title: {
    color: '#fff',
    fontSize: 24,
    fontWeight: 'bold',
  },
  subtitle: {
    color: '#ccc',
    fontSize: 16,
    marginTop: 5,
  },
  faceOutline: {
    width: 250,
    height: 350,
    borderWidth: 3,
    borderColor: 'rgba(0, 255, 0, 0.7)',
    borderRadius: 125,
    borderStyle: 'dashed',
  },
  footerContainer: {
    alignItems: 'center',
    marginBottom: 20,
  },
  footer: {
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 30,
    marginBottom: 10,
  },
  instruction: {
    color: '#00ff00',
    fontSize: 20,
    fontWeight: 'bold',
  },
  devText: {
    color: 'rgba(255,255,255,0.5)',
    fontSize: 12,
  },
});
