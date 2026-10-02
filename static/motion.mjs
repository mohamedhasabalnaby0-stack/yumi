import * as THREE from 'three';

// Original authored poses, not third-party mocap. Every clip owns the same rig
// tracks so body layers never fight; the mixer crossfades complete poses.
const REST = {
  hips: [0, 0, 0], spine: [0, 0, 0], chest: [0, 0, 0], upperChest: [0, 0, 0],
  neck: [0, 0, 0], head: [0, 0, 0],
  leftShoulder: [0, 0, 0], rightShoulder: [0, 0, 0],
  leftUpperArm: [0, 0, 1.15], rightUpperArm: [0, 0, -1.15],
  leftLowerArm: [0, 0, .18], rightLowerArm: [0, 0, -.18],
  leftHand: [0, 0, .05], rightHand: [0, 0, -.05],
  leftUpperLeg: [0, 0, 0], rightUpperLeg: [0, 0, 0],
  leftLowerLeg: [0, 0, 0], rightLowerLeg: [0, 0, 0],
  leftFoot: [0, 0, 0], rightFoot: [0, 0, 0],
};
for (const side of ['left', 'right']) {
  const sign = side === 'left' ? 1 : -1;
  for (const finger of ['Index', 'Middle', 'Ring', 'Little']) {
    for (const segment of ['Proximal', 'Intermediate', 'Distal']) {
      REST[`${side}${finger}${segment}`] = [0, 0, sign * .15];
    }
  }
}
const frame = (time, pose = {}, lift = 0) => ({ time, pose, lift });
export const CLIPS = {
  idle: [frame(0), frame(1.8, { chest: [.015, .015, .008], head: [-.015, .04, -.02], leftHand: [.04, 0, .07] }, .0015), frame(4, { chest: [-.008, -.015, -.008], head: [.01, -.045, .015], rightHand: [-.03, 0, -.07] }), frame(6.4)],
  thinking: [frame(0, { head: [.04, -.12, -.08], rightUpperArm: [-.55, -.35, -.8], rightLowerArm: [0, -.9, -1.25], rightHand: [.05, .25, .1] }), frame(1.8, { head: [.02, -.07, -.06], rightUpperArm: [-.58, -.35, -.8], rightLowerArm: [0, -.9, -1.25], rightHand: [.08, .2, .1], chest: [.015, -.025, 0] }), frame(3.6, { head: [.04, -.12, -.08], rightUpperArm: [-.55, -.35, -.8], rightLowerArm: [0, -.9, -1.25], rightHand: [.05, .25, .1] })],
  talking: [frame(0), frame(.65, { chest: [.025, -.05, 0], head: [.03, .03, -.02], rightUpperArm: [-.3, -.2, -.85], rightLowerArm: [0, -.65, -.7], rightHand: [.12, .3, -.1] }), frame(1.4, { head: [-.02, .015, 0], rightUpperArm: [-.4, -.2, -.75], rightLowerArm: [0, -.45, -.8], rightHand: [.15, .5, -.15] }), frame(2.2), frame(3.2, { chest: [.02, .035, .015], head: [.02, -.035, .015], leftUpperArm: [-.25, .15, .9], leftLowerArm: [0, .55, .65], leftHand: [.1, -.35, .12] }), frame(4.1, { head: [-.025, -.015, 0], leftUpperArm: [-.2, .18, .92], leftLowerArm: [0, .35, .6] }), frame(5.2)],
  wave: [frame(0), frame(.35, { chest: [0, -.06, .02], head: [0, .06, -.06], rightUpperArm: [-.15, 0, -.25], rightLowerArm: [0, 0, 1.65] }), frame(.7, { head: [0, .06, -.06], rightUpperArm: [-.15, 0, -.2], rightLowerArm: [0, 0, 1.65], rightHand: [.1, .25, .25] }), frame(.95, { rightUpperArm: [-.15, 0, -.2], rightLowerArm: [0, 0, 1.65], rightHand: [-.1, -.25, -.25] }), frame(1.2, { rightUpperArm: [-.15, 0, -.2], rightLowerArm: [0, 0, 1.65], rightHand: [.1, .25, .25] }), frame(1.45, { rightUpperArm: [-.15, 0, -.2], rightLowerArm: [0, 0, 1.65], rightHand: [-.1, -.25, -.25] }), frame(1.75, { rightUpperArm: [-.1, 0, -.65], rightLowerArm: [0, 0, .7] }), frame(2.2)],
  nod: [frame(0), frame(.25, { head: [.16, 0, 0], neck: [.04, 0, 0], chest: [.02, 0, 0] }), frame(.5, { head: [-.035, 0, 0] }), frame(.8, { head: [.12, 0, 0], neck: [.02, 0, 0] }), frame(1.2)],
  think: [frame(0), frame(.65, { head: [.03, -.1, -.08], rightUpperArm: [-.55, -.35, -.8], rightLowerArm: [0, -.9, -1.25], rightHand: [.05, .25, .1] }), frame(1.7, { head: [.02, -.04, -.06], rightUpperArm: [-.55, -.35, -.8], rightLowerArm: [0, -.9, -1.25], rightHand: [.05, .25, .1] }), frame(2.5)],
  laugh: [frame(0), frame(.3, { chest: [-.05, 0, 0], head: [-.12, .02, 0], leftShoulder: [0, 0, .04], rightShoulder: [0, 0, -.04] }, .003), frame(.52, { chest: [.06, 0, 0], head: [.025, 0, .02] }), frame(.8, { chest: [-.04, 0, 0], head: [-.08, 0, -.02] }, .002), frame(1.02, { chest: [.05, 0, 0], head: [.015, 0, .02] }), frame(1.3, { chest: [-.02, 0, 0], head: [-.06, 0, 0] }), frame(1.9)],
  shrug: [frame(0), frame(.5, { leftShoulder: [0, 0, -.09], rightShoulder: [0, 0, .09], leftUpperArm: [-.1, .15, .85], rightUpperArm: [-.1, -.15, -.85], leftLowerArm: [0, .75, .8], rightLowerArm: [0, -.75, -.8], leftHand: [0, -.7, .1], rightHand: [0, .7, -.1], head: [0, .04, -.09] }), frame(1.25, { leftUpperArm: [-.1, .15, .85], rightUpperArm: [-.1, -.15, -.85], leftLowerArm: [0, .75, .8], rightLowerArm: [0, -.75, -.8], leftHand: [0, -.7, .1], rightHand: [0, .7, -.1], head: [.02, -.03, -.06] }), frame(1.95)],
};

export function buildClip(vrm, name, frames) {
  const tracks = [];
  // Ease each authored interval with quaternion slerp, preserving rotations.
  const samples = [];
  for (let i = 0; i < frames.length - 1; i++) {
    const a = frames[i], b = frames[i + 1];
    const count = Math.max(2, Math.ceil((b.time - a.time) * 30));
    for (let n = 0; n < count; n++) {
      const t = n / count;
      samples.push({ a, b, mix: t * t * (3 - 2 * t), time: a.time + (b.time - a.time) * t });
    }
  }
  const last = frames.at(-1);
  samples.push({ a: last, b: last, mix: 0, time: last.time });
  const times = samples.map(s => s.time);
  for (const [name, rest] of Object.entries(REST)) {
    const bone = vrm.humanoid.getNormalizedBoneNode(name);
    if (!bone) continue;
    const values = [];
    for (const s of samples) {
      const a = new THREE.Quaternion().setFromEuler(new THREE.Euler(...(s.a.pose[name] ?? rest)));
      const b = new THREE.Quaternion().setFromEuler(new THREE.Euler(...(s.b.pose[name] ?? rest)));
      a.slerp(b, s.mix).toArray(values, values.length);
    }
    tracks.push(new THREE.QuaternionKeyframeTrack(`${bone.uuid}.quaternion`, times, values));
  }
  const hips = vrm.humanoid.getNormalizedBoneNode('hips');
  if (hips) {
    const p = hips.position;
    const values = samples.flatMap(s => [p.x, p.y + THREE.MathUtils.lerp(s.a.lift, s.b.lift, s.mix), p.z]);
    tracks.push(new THREE.VectorKeyframeTrack(`${hips.uuid}.position`, times, values));
  }
  return new THREE.AnimationClip(name, last.time, tracks);
}

export class MotionController {
  constructor(vrm) {
    this.mixer = new THREE.AnimationMixer(vrm.scene);
    this.actions = new Map(Object.entries(CLIPS).map(([name, frames]) => [name, this.mixer.clipAction(buildClip(vrm, name, frames))]));
    this.state = 'idle';
    this.enabled = true;
    this.active = null;
    this.gesture = false;
    this.retiring = new Map();
    this.mixer.addEventListener('finished', event => {
      if (event.action === this.active) { this.gesture = false; this.transition(this.state); }
    });
    this.transition('idle');
  }
  transition(name, once = false) {
    const action = this.actions.get(name);
    if (!action || (action === this.active && !once)) return;
    this.retiring.delete(action);
    action.reset().setEffectiveTimeScale(1).setEffectiveWeight(1);
    action.setLoop(once ? THREE.LoopOnce : THREE.LoopRepeat, once ? 1 : Infinity);
    action.clampWhenFinished = once;
    action.play();
    if (this.active && this.active !== action) {
      this.active.fadeOut(.3);
      action.fadeIn(.3);
      this.retiring.set(this.active, .35);
    }
    this.active = action;
  }
  setState(state) {
    this.state = ['idle', 'thinking', 'talking'].includes(state) ? state : 'idle';
    if (!this.gesture && this.enabled) this.transition(this.state);
  }
  play(name) {
    if (!this.enabled || !this.actions.has(name) || ['idle', 'thinking', 'talking'].includes(name)) return;
    this.gesture = true;
    this.transition(name, true);
  }
  setEnabled(enabled) {
    this.enabled = enabled;
    this.gesture = false;
    this.mixer.stopAllAction();
    this.retiring.clear();
    this.active = null;
    this.transition(enabled ? this.state : 'idle');
    this.mixer.update(0);
  }
  update(dt) {
    if (!this.enabled) return;
    for (const [action, remaining] of this.retiring) {
      if (remaining <= dt) { action.stop(); this.retiring.delete(action); }
      else this.retiring.set(action, remaining - dt);
    }
    this.mixer.update(dt);
  }
}
