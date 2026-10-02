import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { CLIPS, buildClip, MotionController } from '../static/motion.mjs';
import { damp, sanitizeCues, sampleCues, SpeechPlayer } from '../static/lips.mjs';

function rig() {
  const scene = new THREE.Group();
  const bones = new Map();
  return { scene, humanoid: { getNormalizedBoneNode(name) {
    if (!bones.has(name)) { const bone = new THREE.Bone(); scene.add(bone); bones.set(name, bone); }
    return bones.get(name);
  } } };
}

test('all authored clips contain valid normalized quaternion tracks', () => {
  const vrm = rig();
  for (const [name, frames] of Object.entries(CLIPS)) {
    const clip = buildClip(vrm, name, frames);
    assert.ok(clip.validate(), name);
    assert.ok(clip.duration > 0);
    assert.ok(clip.tracks.length >= 20);
    for (const track of clip.tracks.filter(track => track.ValueTypeName === 'quaternion')) {
      for (let i = 0; i < track.values.length; i += 4) {
        assert.ok(Math.abs(Math.hypot(...track.values.slice(i, i + 4)) - 1) < 1e-5);
      }
    }
  }
});

test('gestures return to the requested base state', () => {
  const motion = new MotionController(rig());
  motion.play('wave');
  motion.setState('talking');
  for (let i = 0; i < 180; i++) motion.update(1 / 60);
  assert.equal(motion.active.getClip().name, 'talking');
  assert.equal(motion.gesture, false);
  assert.equal(motion.retiring.size, 0);
});

test('reduced motion freezes the rig but preserves state for resuming', () => {
  const motion = new MotionController(rig());
  motion.setEnabled(false);
  motion.setState('thinking');
  motion.play('wave');
  motion.update(1);
  assert.equal(motion.active.getClip().name, 'idle');
  assert.equal(motion.active.time, 0);
  motion.setEnabled(true);
  assert.equal(motion.active.getClip().name, 'thinking');
});

test('mouth sampling respects actual audio time, closed consonants, and gaps', () => {
  const cues = sanitizeCues([{ start: 0, end: .3, weights: { aa: .8 } }, { start: .3, end: .5, weights: {} }, { start: .8, end: 1, weights: { ou: .7 } }]);
  assert.equal(sampleCues(cues, .15).aa, .8);
  assert.equal(sampleCues(cues, .4).aa, 0);
  assert.equal(sampleCues(cues, .6).aa, 0);
  assert.equal(sampleCues(cues, .9).ou, .7);
  assert.equal(sampleCues(cues, 1.2).ou, 0);
  assert.equal(sampleCues(cues, -1).aa, 0);
});

test('malformed cues cannot produce unbounded mouth expressions', () => {
  const cues = sanitizeCues([null, { start: NaN, end: 1 }, { start: 0, end: 1, weights: { aa: 3, ee: 2, ih: -1, ou: Infinity } }]);
  assert.equal(cues.length, 1);
  assert.ok(Object.values(cues[0].weights).reduce((a, b) => a + b, 0) <= 1);
  assert.equal(cues[0].weights.ih, 0);
});

test('damping is frame-rate independent', () => {
  let sixty = 0, thirty = 0;
  for (let i = 0; i < 60; i++) sixty = damp(sixty, 1, 7, 1 / 60);
  for (let i = 0; i < 30; i++) thirty = damp(thirty, 1, 7, 1 / 30);
  assert.ok(Math.abs(sixty - thirty) < 1e-10);
});

test('stop disconnects audio nodes and invalidates pending playback', () => {
  let disconnected = 0, stopped = 0, state = true;
  const player = new SpeechPlayer(value => { state = value; });
  player.source = { stop() { stopped++; }, disconnect() { disconnected++; } };
  player.analyser = { disconnect() { disconnected++; } };
  player.gain = { disconnect() { disconnected++; } };
  player.stop();
  assert.equal(disconnected, 3);
  assert.equal(stopped, 1);
  assert.equal(state, false);
  assert.equal(player.playing, false);
  assert.equal(player.generation, 1);
  assert.equal(player.weights().aa, 0);
});
