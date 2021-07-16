import { Tensor, InferenceSession } from 'onnxjs';

export const session: InferenceSession = new InferenceSession({ backendHint: 'webgl' });
const url = 'model.onnx';

export const loadModel = () => {
  session.loadModel(url);
};

export const HIDDEN_SIZE = 100;
export class OutputParams {
  /** Optional: title */
  title: string;

  /** Key as a number between 1-12 */
  key: number;

  /**
   * Musical mode
   * 1: Ionian (Major)
   * 2: Dorian
   * 3: Phrygia
   * 4: Lydian
   * 5: Mixolydian
   * 6: Aeolian (Minor)
   * 7: Locrian
   */
  mode: number;

  /** Beats per minute */
  bpm: number;

  /** How energetic the track should be, 0 (less energetic) to 1 (very energetic) */
  energy: number;

  /** How positive the music should be, 0 (sad) to 1 (cheerful) */
  valence: number;

  chords: number[];

  melodies: number[][];

  public constructor(init?: Partial<OutputParams>) {
    Object.assign(this, init);
  }
}

const argmax = (arr: Float32Array) => arr.indexOf(Math.max(...arr));

export const decode2 = async (input: number[]) => {
  const vec = Float32Array.of(...input);
  const inputs = [
    new Tensor(vec, 'float32', [1, 100])
  ];
  const outputMap = await session.run(inputs);

  const chords = [];
  const melodies = [];

  const outputTensors = [...outputMap.values()].map((t) => t.data as Float32Array);

  const maxNumChords = 16;

  for (let i = 0; i < maxNumChords; i += 1) {
    const chord = argmax(outputTensors[i]);
    if (chord === 8) break;
    chords.push(chord);
    melodies.push(outputTensors.slice(16 + i * 8, 16 + (i + 1) * 8).map((m) => argmax(m)));
  }
  const key = argmax(outputTensors[outputTensors.length - 5]) + 1;
  const mode = argmax(outputTensors[outputTensors.length - 4]) + 1;
  const bpm = Math.round(outputTensors[outputTensors.length - 3][0]);
  const energy = parseFloat(outputTensors[outputTensors.length - 2][0].toFixed(3));
  const valence = parseFloat(outputTensors[outputTensors.length - 1][0].toFixed(3));

  const outputParams = new OutputParams({
    title: '',
    key,
    mode,
    bpm,
    valence,
    energy,
    chords,
    melodies
  });

  return outputParams;
};
