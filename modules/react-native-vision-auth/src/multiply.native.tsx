import { NitroModules } from 'react-native-nitro-modules';
import type { VisionAuth } from './VisionAuth.nitro';

const VisionAuthHybridObject =
  NitroModules.createHybridObject<VisionAuth>('VisionAuth');

export function multiply(a: number, b: number): number {
  return VisionAuthHybridObject.multiply(a, b);
}
