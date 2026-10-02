export const MOUTH_NAMES = ['aa', 'ih', 'ee', 'oh', 'ou'];
export const damp = (from, to, speed, dt) => from + (to - from) * (1 - Math.exp(-speed * Math.max(0, dt)));

export function sanitizeCues(cues) {
  if (!Array.isArray(cues)) return [];
  let previous = 0;
  const result = [];
  for (const cue of cues) {
    if (!cue || !Number.isFinite(cue.start) || !Number.isFinite(cue.end)) continue;
    const start = Math.max(0, previous, cue.start), end = cue.end;
    if (end <= start) continue;
    const weights = Object.fromEntries(MOUTH_NAMES.map(name => [name,
      Number.isFinite(cue.weights?.[name]) ? Math.max(0, Math.min(1, cue.weights[name])) : 0]));
    const total = Object.values(weights).reduce((a, b) => a + b, 0);
    if (total > 1) for (const name of MOUTH_NAMES) weights[name] /= total;
    result.push({ start, end, weights });
    previous = end;
  }
  return result;
}

export function sampleCues(cues, time) {
  const zero = Object.fromEntries(MOUTH_NAMES.map(name => [name, 0]));
  if (!Number.isFinite(time) || time < 0) return zero;
  let lo = 0, hi = cues.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const cue = cues[mid];
    if (time < cue.start) hi = mid - 1;
    else if (time >= cue.end) lo = mid + 1;
    else {
      // Short ease-in/out is based on audio time, never wall-clock sine waves.
      const edge = Math.min(.035, (cue.end - cue.start) / 3);
      const amount = Math.min(1, (time - cue.start) / edge, (cue.end - time) / edge);
      for (const name of MOUTH_NAMES) zero[name] = (cue.weights[name] || 0) * amount;
      return zero;
    }
  }
  return zero;
}

export class SpeechPlayer {
  constructor(onState) {
    this.onState = onState;
    this.context = null;
    this.source = null;
    this.analyser = null;
    this.gain = null;
    this.cues = [];
    this.generation = 0;
    this.volume = .8;
    this.muted = false;
  }
  async unlock() {
    if (!this.context) {
      const AudioContext = globalThis.AudioContext || globalThis.webkitAudioContext;
      if (!AudioContext) throw new Error('Web Audio is not supported by this browser.');
      this.context = new AudioContext();
    }
    if (this.context.state !== 'running') await this.context.resume();
    if (this.context.state !== 'running') throw new Error('Click Replay voice to enable audio.');
  }
  setVolume(volume, muted) {
    this.volume = volume;
    this.muted = muted;
    if (this.gain) this.gain.gain.setTargetAtTime(muted ? 0 : volume, this.context.currentTime, .015);
  }
  stop() {
    this.generation++;
    if (this.source) {
      this.source.onended = null;
      try { this.source.stop(); } catch { /* already ended */ }
      this.source.disconnect();
    }
    this.analyser?.disconnect();
    this.gain?.disconnect();
    this.source = this.analyser = this.gain = null;
    this.cues = [];
    this.onState(false);
  }
  async play(dataUri, cues) {
    this.stop();
    const generation = this.generation;
    await this.unlock();
    if (!/^data:audio\/(wav|mpeg);base64,/.test(dataUri)) throw new Error('Unsupported audio response.');
    const raw = atob(dataUri.slice(dataUri.indexOf(',') + 1));
    const bytes = Uint8Array.from(raw, char => char.charCodeAt(0));
    const buffer = await this.context.decodeAudioData(bytes.buffer);
    if (generation !== this.generation) return;
    this.source = this.context.createBufferSource();
    this.source.buffer = buffer;
    this.analyser = this.context.createAnalyser();
    this.analyser.fftSize = 1024;
    this.samples = new Float32Array(this.analyser.fftSize);
    this.gain = this.context.createGain();
    this.source.connect(this.analyser);
    this.analyser.connect(this.gain);
    this.gain.connect(this.context.destination);
    this.setVolume(this.volume, this.muted);
    this.cues = sanitizeCues(cues);
    this.startedAt = this.context.currentTime;
    this.duration = buffer.duration;
    this.source.onended = () => { if (generation === this.generation) this.stop(); };
    this.source.start();
    this.onState(true);
  }
  get playing() { return !!this.source && this.context?.state === 'running'; }
  get time() { return this.playing ? Math.max(0, Math.min(this.duration, this.context.currentTime - this.startedAt)) : 0; }
  energy() {
    if (!this.playing || !this.analyser) return 0;
    this.analyser.getFloatTimeDomainData(this.samples);
    let sum = 0;
    for (const sample of this.samples) sum += sample * sample;
    return Math.max(0, Math.min(1, (Math.sqrt(sum / this.samples.length) - .008) * 12));
  }
  weights() {
    if (!this.playing) return Object.fromEntries(MOUTH_NAMES.map(name => [name, 0]));
    if (this.cues.length) return sampleCues(this.cues, this.time);
    return { aa: this.energy() * .75, ih: 0, ee: 0, oh: 0, ou: 0 };
  }
}
