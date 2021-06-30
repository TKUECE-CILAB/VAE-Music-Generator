import * as Tone from 'tone';
import sampleConfig from './samples.json';

export const SAMPLES_BASE_URL = './samples';
export const DRUM_LOOP_DEFAULT_VOLUME = -6;

class SampleGroup {
  name: string;

  category: string;

  urls: string[];

  volume: number;

  size: number;

  energyRanges: number[][];

  public constructor(
    name: string,
    category: string,
    urls: string[],
    volume: number,
    energyRanges: number[][]
  ) {
    this.name = name;
    this.category = category;
    this.urls = urls;
    this.volume = DRUM_LOOP_DEFAULT_VOLUME + volume;
    this.size = urls.length;
    this.energyRanges = energyRanges;
  }

  getSampleUrl(index: number) {
    return `${SAMPLES_BASE_URL}/loops/${this.category}/${this.name}/${this.urls[index]}`;
  }

  getFilters() {
    if (this.category === 'drums') {
      return [
        new Tone.Filter({
          type: 'lowpass',
          frequency: 2400,
          Q: 1.0
        })
      ];
    }
    return [];
  }
}

export const SAMPLEGROUPS: Map<string, SampleGroup> = sampleConfig.reduce((map, sampleGroup) => {
  map.set(
    sampleGroup.name,
    new SampleGroup(
      sampleGroup.name,
      sampleGroup.category,
      sampleGroup.urls,
      sampleGroup.volume,
      sampleGroup.energyRanges
    )
  );
  return map;
}, new Map());

export const selectDrumbeat = (bpm: number, energy: number): [string, number] => {
  const sampleGroup = `drumloop${bpm}`;

  const index = SAMPLEGROUPS.get(sampleGroup).energyRanges.findIndex(
    (range) => range[0] <= energy && range[1] >= energy
  );

  return [sampleGroup, index];
};
