import React, {useEffect, useCallback, useRef} from 'react';
import {
  StyleSheet,
  Text,
  View,
  ActivityIndicator,
  Dimensions,
  TouchableOpacity,
  TextInput,
  Alert,
  StatusBar,
} from 'react-native';
import {SafeAreaView, SafeAreaProvider} from 'react-native-safe-area-context';
import {
  useCameraDevice,
  useCameraPermission,
  useFrameOutput,
  Camera,
} from 'react-native-vision-camera';
import {
  VisionAuthModule,
  type VisionAuthResult,
} from 'react-native-vision-auth';
import {runOnJS} from 'react-native-worklets';
import * as RNFS from 'react-native-fs';
import {createMMKV} from 'react-native-mmkv';

const storage = createMMKV();
const MATCH_THRESHOLD = 0.830;
const EMBEDDINGS_KEY = 'stored_face_embeddings';

type StoredFace = {
  label: string;
  embedding: number[];
};

type AppMode = 'menu' | 'store' | 'match';

function cosineSimilarity(a: number[], b: number[]): number {
  // Both embeddings are already L2-normalized in C++, so dot product = cosine similarity
  let dot = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
  }
  return dot;
}

function getStoredFaces(): StoredFace[] {
  const raw = storage.getString(EMBEDDINGS_KEY);
  if (!raw) {
    return [];
  }
  try {
    return JSON.parse(raw) as StoredFace[];
  } catch {
    return [];
  }
}

function saveStoredFaces(faces: StoredFace[]) {
  storage.set(EMBEDDINGS_KEY, JSON.stringify(faces));
}

// ─── Home Menu Screen ────────────────────────────────────────────────
function HomeMenu({
  onSelect,
  storedCount,
}: {
  onSelect: (mode: AppMode) => void;
  storedCount: number;
}) {
  return (
    <SafeAreaView style={menuStyles.container}>
      <StatusBar barStyle="light-content" />
      <View style={menuStyles.hero}>
        <Text style={menuStyles.icon}>🔐</Text>
        <Text style={menuStyles.title}>Face Auth</Text>
        <Text style={menuStyles.subtitle}>
          Secure face recognition powered by on-device AI
        </Text>
      </View>

      <View style={menuStyles.cardContainer}>
        <TouchableOpacity
          style={menuStyles.card}
          activeOpacity={0.85}
          onPress={() => onSelect('store')}>
          <View style={menuStyles.cardIconWrap}>
            <Text style={menuStyles.cardIcon}>📸</Text>
          </View>
          <Text style={menuStyles.cardTitle}>Store Face</Text>
          <Text style={menuStyles.cardDesc}>
            Register a new face with a label
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[menuStyles.card, menuStyles.cardAlt]}
          activeOpacity={0.85}
          onPress={() => onSelect('match')}>
          <View style={[menuStyles.cardIconWrap, menuStyles.cardIconWrapAlt]}>
            <Text style={menuStyles.cardIcon}>🔍</Text>
          </View>
          <Text style={menuStyles.cardTitle}>Match Face</Text>
          <Text style={menuStyles.cardDesc}>
            Check if a face matches stored records
          </Text>
        </TouchableOpacity>
      </View>

      <View style={menuStyles.badge}>
        <Text style={menuStyles.badgeText}>
          {storedCount} face{storedCount !== 1 ? 's' : ''} stored
        </Text>
      </View>
    </SafeAreaView>
  );
}

// ─── Label Input Modal ───────────────────────────────────────────────
function LabelInput({
  onSubmit,
  onCancel,
}: {
  onSubmit: (label: string) => void;
  onCancel: () => void;
}) {
  const [label, setLabel] = React.useState('');

  return (
    <View style={labelStyles.overlay}>
      <View style={labelStyles.modal}>
        <Text style={labelStyles.modalTitle}>✅ Face Captured</Text>
        <Text style={labelStyles.modalSubtitle}>
          Enter a name or label for this face
        </Text>
        <TextInput
          style={labelStyles.input}
          placeholder="e.g. John Doe"
          placeholderTextColor="#666"
          value={label}
          onChangeText={setLabel}
          autoFocus
          returnKeyType="done"
          onSubmitEditing={() => {
            if (label.trim()) {
              onSubmit(label.trim());
            }
          }}
        />
        <View style={labelStyles.buttonRow}>
          <TouchableOpacity style={labelStyles.cancelBtn} onPress={onCancel}>
            <Text style={labelStyles.cancelText}>Cancel</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[
              labelStyles.saveBtn,
              !label.trim() && labelStyles.saveBtnDisabled,
            ]}
            onPress={() => {
              if (label.trim()) {
                onSubmit(label.trim());
              }
            }}
            disabled={!label.trim()}>
            <Text style={labelStyles.saveText}>Save</Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
}

// ─── Match Result Overlay ────────────────────────────────────────────
function MatchResult({
  matchLabel,
  similarity,
  onDone,
}: {
  matchLabel: string | null;
  similarity: number;
  onDone: () => void;
}) {
  const matched = matchLabel !== null;
  return (
    <View style={matchStyles.overlay}>
      <View style={matchStyles.modal}>
        <Text style={matchStyles.icon}>{matched ? '✅' : '❌'}</Text>
        <Text style={matchStyles.title}>
          {matched ? 'Match Found!' : 'No Match'}
        </Text>
        {matched ? (
          <>
            <Text style={matchStyles.label}>{matchLabel}</Text>
            <Text style={matchStyles.score}>
              Similarity: {(similarity * 100).toFixed(1)}%
            </Text>
          </>
        ) : (
          <Text style={matchStyles.noMatch}>
            Face does not match any stored record.
            {'\n'}Best similarity: {(similarity * 100).toFixed(1)}%
          </Text>
        )}
        <TouchableOpacity style={matchStyles.doneBtn} onPress={onDone}>
          <Text style={matchStyles.doneText}>Done</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

// ─── Camera Screen ───────────────────────────────────────────────────
function CameraScreen({
  mode,
  onBack,
  onStoreDone,
  onMatchDone,
}: {
  mode: 'store' | 'match';
  onBack: () => void;
  onStoreDone: (embedding: number[]) => void;
  onMatchDone: (embedding: number[]) => void;
}) {
  const device = useCameraDevice('front');
  const [modelsReady, setModelsReady] = React.useState(false);
  const capturedRef = useRef(false);
  const latestEmbeddingRef = useRef<number[] | null>(null);
  const spoofCooldownRef = useRef<number>(0);
  const [livenessStatus, setLivenessStatus] = React.useState(
    'Position your face and blink',
  );

  const [debugMetrics, setDebugMetrics] = React.useState<{
    faceBox?: number[];
    leftEyeBox?: number[];
    rightEyeBox?: number[];
    leftEAR?: number;
    rightEAR?: number;
    baseline?: number;
  }>({});

  useEffect(() => {
    async function setup() {
      try {
        const models = [
          'blaze_face_short_range.tflite',
          'face_landmarks_detector.tflite',
          'student_qat_int8.tflite',
          'minifasnet.tflite',
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
          paths[3],
        );
        if (success) {
          setModelsReady(true);
        } else {
          console.error('Failed to load models');
        }
      } catch (e) {
        console.error('Error setting up models:', e);
      }
    }
    setup();
  }, []);

  const handleFrameResult = useCallback(
    (result: VisionAuthResult) => {
      try {
        setDebugMetrics({
          faceBox: result.faceBox,
          leftEyeBox: result.leftEyeBox,
          rightEyeBox: result.rightEyeBox,
          leftEAR: result.leftEAR,
          rightEAR: result.rightEAR,
          baseline: result.baseline,
        });

        // Cache the latest valid embedding as it comes in
        // (runs every 5 frames in C++)
        if (result.embeddingOk && result.embedding) {
          latestEmbeddingRef.current = Array.from(result.embedding);
        }

        if (!result.faceDetected) {
          setLivenessStatus('No face detected');
        } else if (result.eyesClosed) {
          if (!capturedRef.current) {
            if (result.livenessScore !== undefined && result.livenessScore < 0.90) {
              spoofCooldownRef.current = Date.now() + 2000;
              setLivenessStatus(`Spoof Detected! Score: ${(result.livenessScore * 100).toFixed(0)}%`);
            } else if (latestEmbeddingRef.current) {
              capturedRef.current = true;
              setLivenessStatus('Liveness Confirmed: Real Face!');
              const emb = latestEmbeddingRef.current;
              if (mode === 'store') {
                onStoreDone(emb);
              } else {
                onMatchDone(emb);
              }
            } else {
              setLivenessStatus('Blink detected, analyzing liveness...');
            }
          }
        } else {
          if (!capturedRef.current && Date.now() > spoofCooldownRef.current) {
            setLivenessStatus('Please blink to verify liveness');
          }
        }
      } catch (e) {
        console.error('Frame parsing error', e);
      }
    },
    [mode, onStoreDone, onMatchDone],
  );

  const frameOutput = useFrameOutput({
    pixelFormat: 'rgb',
    onFrame: frame => {
      'worklet';
      if (frame.hasPixelBuffer) {
        const buffer = frame.getPixelBuffer();
        try {
          const result = VisionAuthModule.analyzeFrame(
            buffer,
            frame.width,
            frame.height,
            frame.bytesPerRow,
          );
          runOnJS(handleFrameResult)(result);
        } catch (e) {
          console.error('Error calling Nitro Module:', e);
        }
      }
      frame.dispose();
    },
  });

  if (device == null) {
    return (
      <SafeAreaView style={styles.container}>
        <Text style={styles.text}>No Front Camera Found</Text>
      </SafeAreaView>
    );
  }

  const screenW = Dimensions.get('window').width;
  const screenH = Dimensions.get('window').height;

  return (
    <View style={styles.container}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={modelsReady}
        outputs={[frameOutput]}
      />

      {/* Face bounding box */}
      {debugMetrics.faceBox && (
        <View
          style={[
            styles.dynamicFaceBox,
            {
              left:
                (1 -
                  (debugMetrics.faceBox[0] + debugMetrics.faceBox[2]) / 720) *
                screenW,
              top: (debugMetrics.faceBox[1] / 1280) * screenH,
              width: (debugMetrics.faceBox[2] / 720) * screenW,
              height: (debugMetrics.faceBox[3] / 1280) * screenH,
            },
          ]}
        />
      )}

      {/* Eye boxes */}
      {debugMetrics.leftEyeBox && (
        <View
          style={[
            styles.dynamicFaceBox,
            {
              borderColor: '#00ffff',
              backgroundColor: 'rgba(0, 255, 255, 0.1)',
              left:
                (1 -
                  (debugMetrics.leftEyeBox[0] + debugMetrics.leftEyeBox[2]) /
                    720) *
                screenW,
              top: (debugMetrics.leftEyeBox[1] / 1280) * screenH,
              width: (debugMetrics.leftEyeBox[2] / 720) * screenW,
              height: (debugMetrics.leftEyeBox[3] / 1280) * screenH,
            },
          ]}
        />
      )}
      {debugMetrics.rightEyeBox && (
        <View
          style={[
            styles.dynamicFaceBox,
            {
              borderColor: '#00ffff',
              backgroundColor: 'rgba(0, 255, 255, 0.1)',
              left:
                (1 -
                  (debugMetrics.rightEyeBox[0] + debugMetrics.rightEyeBox[2]) /
                    720) *
                screenW,
              top: (debugMetrics.rightEyeBox[1] / 1280) * screenH,
              width: (debugMetrics.rightEyeBox[2] / 720) * screenW,
              height: (debugMetrics.rightEyeBox[3] / 1280) * screenH,
            },
          ]}
        />
      )}

      {/* Camera overlay UI */}
      <SafeAreaView style={styles.overlay}>
        <View style={styles.header}>
          <TouchableOpacity style={camStyles.backBtn} onPress={onBack}>
            <Text style={camStyles.backText}>← Back</Text>
          </TouchableOpacity>
          <Text style={styles.title}>
            {mode === 'store' ? 'Store Face' : 'Match Face'}
          </Text>
          <Text style={styles.subtitle}>
            Position your face and blink to capture
          </Text>
        </View>

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

      {/* Debug HUD */}
      <View style={styles.debugHud}>
        <Text style={styles.debugText}>
          Left EAR: {debugMetrics.leftEAR?.toFixed(3) || '0.000'}
        </Text>
        <Text style={styles.debugText}>
          Right EAR: {debugMetrics.rightEAR?.toFixed(3) || '0.000'}
        </Text>
        <Text style={styles.debugText}>
          Baseline: {debugMetrics.baseline?.toFixed(3) || '0.000'}
        </Text>
      </View>
    </View>
  );
}

// ─── Main App ────────────────────────────────────────────────────────
function AppContent() {
  const {hasPermission, requestPermission} = useCameraPermission();
  const [mode, setMode] = React.useState<AppMode>('menu');
  const [storedCount, setStoredCount] = React.useState(0);
  const [pendingEmbedding, setPendingEmbedding] = React.useState<
    number[] | null
  >(null);
  const [matchResult, setMatchResult] = React.useState<{
    label: string | null;
    similarity: number;
  } | null>(null);

  // On app open: clear all stored embeddings and request camera permission
  useEffect(() => {
    storage.remove(EMBEDDINGS_KEY);
    setStoredCount(0);
    console.log('Cleared all stored face embeddings on app launch');

    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  const goToMenu = useCallback(() => {
    setMode('menu');
    setPendingEmbedding(null);
    setMatchResult(null);
    setStoredCount(getStoredFaces().length);
  }, []);

  // Store face flow
  const handleStoreDone = useCallback((embedding: number[]) => {
    setPendingEmbedding(embedding);
  }, []);

  const handleLabelSubmit = useCallback(
    (label: string) => {
      if (!pendingEmbedding) {
        return;
      }
      const faces = getStoredFaces();
      faces.push({label, embedding: pendingEmbedding});
      saveStoredFaces(faces);
      console.log(`Stored face for "${label}" (${pendingEmbedding.length} dims)`);
      Alert.alert('Success', `Face stored for "${label}"`, [
        {text: 'OK', onPress: goToMenu},
      ]);
      setPendingEmbedding(null);
    },
    [pendingEmbedding, goToMenu],
  );

  // Match face flow
  const handleMatchDone = useCallback((embedding: number[]) => {
    const faces = getStoredFaces();
    if (faces.length === 0) {
      setMatchResult({label: null, similarity: 0});
      return;
    }

    let bestLabel: string | null = null;
    let bestSim = -1;

    for (const face of faces) {
      const sim = cosineSimilarity(embedding, face.embedding);
      if (sim > bestSim) {
        bestSim = sim;
        bestLabel = face.label;
      }
    }

    if (bestSim >= MATCH_THRESHOLD) {
      setMatchResult({label: bestLabel, similarity: bestSim});
    } else {
      setMatchResult({label: null, similarity: bestSim});
    }
  }, []);

  if (!hasPermission) {
    return (
      <SafeAreaView style={styles.container}>
        <ActivityIndicator size="large" color="#7C3AED" />
        <Text style={styles.text}>Requesting Camera Permission...</Text>
      </SafeAreaView>
    );
  }

  if (mode === 'menu') {
    return <HomeMenu onSelect={setMode} storedCount={storedCount} />;
  }

  return (
    <View style={{flex: 1}}>
      <CameraScreen
        mode={mode}
        onBack={goToMenu}
        onStoreDone={handleStoreDone}
        onMatchDone={handleMatchDone}
      />

      {/* Label input overlay (store mode) */}
      {mode === 'store' && pendingEmbedding && (
        <LabelInput onSubmit={handleLabelSubmit} onCancel={goToMenu} />
      )}

      {/* Match result overlay */}
      {mode === 'match' && matchResult && (
        <MatchResult
          matchLabel={matchResult.label}
          similarity={matchResult.similarity}
          onDone={goToMenu}
        />
      )}
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

// ─── Styles ──────────────────────────────────────────────────────────
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
    borderColor: 'rgba(255, 255, 255, 0.7)',
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
  dynamicFaceBox: {
    position: 'absolute',
    borderWidth: 2,
    borderColor: '#00ff00',
    backgroundColor: 'rgba(0, 255, 0, 0.1)',
  },
  debugHud: {
    position: 'absolute',
    top: 100,
    left: 20,
    backgroundColor: 'rgba(0,0,0,0.7)',
    padding: 10,
    borderRadius: 8,
  },
  debugText: {
    color: '#00ffff',
    fontFamily: 'monospace',
    fontSize: 14,
    marginBottom: 4,
  },
});

const menuStyles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0F0F1A',
    paddingHorizontal: 24,
    justifyContent: 'center',
  },
  hero: {
    alignItems: 'center',
    marginBottom: 48,
  },
  icon: {
    fontSize: 56,
    marginBottom: 16,
  },
  title: {
    color: '#FFFFFF',
    fontSize: 32,
    fontWeight: '800',
    letterSpacing: 1,
  },
  subtitle: {
    color: '#8B8BA3',
    fontSize: 16,
    marginTop: 8,
    textAlign: 'center',
  },
  cardContainer: {
    gap: 16,
  },
  card: {
    backgroundColor: '#1A1A2E',
    borderRadius: 20,
    padding: 24,
    borderWidth: 1,
    borderColor: 'rgba(124, 58, 237, 0.3)',
  },
  cardAlt: {
    borderColor: 'rgba(6, 182, 212, 0.3)',
  },
  cardIconWrap: {
    width: 52,
    height: 52,
    borderRadius: 16,
    backgroundColor: 'rgba(124, 58, 237, 0.15)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 16,
  },
  cardIconWrapAlt: {
    backgroundColor: 'rgba(6, 182, 212, 0.15)',
  },
  cardIcon: {
    fontSize: 24,
  },
  cardTitle: {
    color: '#FFFFFF',
    fontSize: 20,
    fontWeight: '700',
    marginBottom: 6,
  },
  cardDesc: {
    color: '#8B8BA3',
    fontSize: 14,
    lineHeight: 20,
  },
  badge: {
    alignSelf: 'center',
    marginTop: 32,
    backgroundColor: 'rgba(255,255,255,0.08)',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
  },
  badgeText: {
    color: '#8B8BA3',
    fontSize: 13,
    fontWeight: '600',
  },
});

const camStyles = StyleSheet.create({
  backBtn: {
    backgroundColor: 'rgba(0,0,0,0.5)',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    marginBottom: 12,
  },
  backText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});

const labelStyles = StyleSheet.create({
  overlay: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(0,0,0,0.75)',
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 32,
  },
  modal: {
    backgroundColor: '#1A1A2E',
    borderRadius: 24,
    padding: 32,
    width: '100%',
    maxWidth: 360,
    borderWidth: 1,
    borderColor: 'rgba(124, 58, 237, 0.3)',
  },
  modalTitle: {
    color: '#FFFFFF',
    fontSize: 24,
    fontWeight: '700',
    textAlign: 'center',
    marginBottom: 8,
  },
  modalSubtitle: {
    color: '#8B8BA3',
    fontSize: 15,
    textAlign: 'center',
    marginBottom: 24,
  },
  input: {
    backgroundColor: '#0F0F1A',
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 14,
    color: '#FFFFFF',
    fontSize: 16,
    borderWidth: 1,
    borderColor: 'rgba(124, 58, 237, 0.4)',
    marginBottom: 20,
  },
  buttonRow: {
    flexDirection: 'row',
    gap: 12,
  },
  cancelBtn: {
    flex: 1,
    paddingVertical: 14,
    borderRadius: 12,
    backgroundColor: 'rgba(255,255,255,0.08)',
    alignItems: 'center',
  },
  cancelText: {
    color: '#8B8BA3',
    fontSize: 16,
    fontWeight: '600',
  },
  saveBtn: {
    flex: 1,
    paddingVertical: 14,
    borderRadius: 12,
    backgroundColor: '#7C3AED',
    alignItems: 'center',
  },
  saveBtnDisabled: {
    opacity: 0.4,
  },
  saveText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '700',
  },
});

const matchStyles = StyleSheet.create({
  overlay: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(0,0,0,0.75)',
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 32,
  },
  modal: {
    backgroundColor: '#1A1A2E',
    borderRadius: 24,
    padding: 32,
    width: '100%',
    maxWidth: 360,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(6, 182, 212, 0.3)',
  },
  icon: {
    fontSize: 56,
    marginBottom: 12,
  },
  title: {
    color: '#FFFFFF',
    fontSize: 24,
    fontWeight: '700',
    marginBottom: 8,
  },
  label: {
    color: '#7C3AED',
    fontSize: 28,
    fontWeight: '800',
    marginBottom: 8,
  },
  score: {
    color: '#06B6D4',
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 24,
  },
  noMatch: {
    color: '#8B8BA3',
    fontSize: 15,
    textAlign: 'center',
    lineHeight: 22,
    marginBottom: 24,
  },
  doneBtn: {
    backgroundColor: '#7C3AED',
    paddingHorizontal: 40,
    paddingVertical: 14,
    borderRadius: 12,
    marginTop: 8,
  },
  doneText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '700',
  },
});
