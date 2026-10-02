import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMUtils, VRMLoaderPlugin } from 'https://cdn.jsdelivr.net/npm/@pixiv/three-vrm@2.1.1/lib/three-vrm.module.js';
import { MotionController } from './motion.mjs';
import { SpeechPlayer, MOUTH_NAMES, damp } from './lips.mjs';

const $ = id => document.getElementById(id);
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
$('motion').checked = !reducedMotion.matches;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x090912);
scene.fog = new THREE.Fog(0x090912, 5, 12);
const camera = new THREE.PerspectiveCamera(30, innerWidth / innerHeight, .05, 30);
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
$('stage').append(renderer.domElement);
renderer.domElement.setAttribute('aria-label', 'Yumi avatar. Drag to orbit or scroll to zoom.');
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.enablePan = false;
controls.minPolarAngle = .65;
controls.maxPolarAngle = 1.8;
scene.add(new THREE.HemisphereLight(0xfff4ff, 0x211729, 2.4));
for (const [color, intensity, position] of [[0xffe9e5, 2.7, [2, 4, 4]], [0x8d91ff, 1.8, [-3, 2, 1]], [0xf496d8, 2, [1, 3, -3]]]) {
  const light = new THREE.DirectionalLight(color, intensity);
  light.position.set(...position);
  scene.add(light);
}
const floor = new THREE.Mesh(new THREE.CircleGeometry(5, 80), new THREE.MeshStandardMaterial({ color: 0x14101e, roughness: .95 }));
floor.rotation.x = -Math.PI / 2;
floor.position.y = -.006;
scene.add(floor);
const ring = new THREE.Mesh(new THREE.RingGeometry(.7, .705, 96), new THREE.MeshBasicMaterial({ color: 0xa274c4, transparent: true, opacity: .3, side: THREE.DoubleSide }));
ring.rotation.x = -Math.PI / 2;
ring.position.y = -.003;
scene.add(ring);

let vrm = null, motion = null, busy = false, emotion = 'neutral', lastSpeech = null;
let modelHeight = 1.6, frameId = 0, disposed = false;
let blinkWait = 2.5, blinkTime = -1;
const gazeTarget = new THREE.Object3D();
scene.add(gazeTarget);
const pointer = new THREE.Vector2();
const mouths = Object.fromEntries(MOUTH_NAMES.map(name => [name, 0]));
const speech = new SpeechPlayer(playing => {
  $('stop').disabled = !playing;
  setState(playing ? 'talking' : busy ? 'thinking' : 'idle');
});

function setState(state) {
  motion?.setState(state);
  $('status').textContent = state === 'thinking' ? 'Yumi is thinking…' : state === 'talking' ? 'Yumi is speaking' : 'Here with you';
}
function addMessage(text, type = 'yumi') {
  const element = document.createElement('div');
  element.className = `message ${type}`;
  element.textContent = text;
  $('messages').append(element);
  // Bound DOM growth in long-running local sessions.
  while ($('messages').children.length > 100) $('messages').firstElementChild.remove();
  $('messages').scrollTop = $('messages').scrollHeight;
}
function fitCamera() {
  const full = $('camera-view').value === 'full';
  const targetY = modelHeight * (full ? .5 : .76);
  // Framing accounts for both portrait screens and the bottom chat overlay.
  const visibleHeight = modelHeight * (full ? 1.65 : .82);
  const distance = visibleHeight / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) / Math.min(1, camera.aspect);
  camera.position.set(0, targetY + modelHeight * .03, distance);
  controls.target.set(0, targetY, 0);
  controls.minDistance = modelHeight * .5;
  controls.maxDistance = Math.max(distance * 1.5, 5);
  controls.update();
}
function hasExpression(name) { return !!vrm?.expressionManager?.getExpression(name); }
function expression(name, value) {
  if (hasExpression(name)) vrm.expressionManager.setValue(name, THREE.MathUtils.clamp(value, 0, 1));
}
function updateFace(dt) {
  const em = vrm.expressionManager;
  if (!em) return;
  for (const name of ['happy', 'sad', 'angry', 'surprised', 'relaxed', 'neutral']) {
    // Reduce emotional mouth overrides while speaking; speech always wins.
    const target = name === emotion ? (speech.playing ? .22 : .65) : 0;
    expression(name, damp(em.getValue(name) || 0, target, 7, dt));
  }
  blinkWait -= dt;
  if (blinkWait <= 0 && blinkTime < 0) blinkTime = 0;
  let blink = 0;
  if (blinkTime >= 0) {
    blinkTime += dt;
    blink = Math.max(0, 1 - Math.abs(blinkTime - .08) / .08);
    if (blinkTime >= .16) { blinkTime = -1; blinkWait = 2 + Math.random() * 4; }
  }
  if (hasExpression('blink')) expression('blink', blink);
  else { expression('blinkLeft', blink); expression('blinkRight', blink); }
  const targets = speech.weights();
  const energy = speech.energy();
  const supported = MOUTH_NAMES.filter(hasExpression);
  if (supported.length === 1 && supported[0] === 'aa') {
    targets.aa = Math.max(...Object.values(targets));
  }
  for (const name of MOUTH_NAMES) {
    const target = speech.playing ? (targets[name] || 0) * (speech.cues.length ? Math.min(1, energy * 4) : 1) : 0;
    mouths[name] = damp(mouths[name], target, target > mouths[name] ? 35 : 28, dt);
    expression(name, mouths[name]);
  }
}
async function playReply(data) {
  $('sync-status').textContent = data.lip_sync_mode === 'rhubarb' ? 'Audio-recognized lip sync' : data.lip_sync_mode === 'aligned-text' ? 'Timed mouth shapes · approximate pronunciation' : 'Audio-energy mouth movement';
  try {
    await speech.play(data.audio, data.mouth_cues);
  } catch (error) {
    speech.stop();
    $('sync-status').textContent = 'Audio paused or unavailable · try Replay voice';
    console.warn('Voice playback:', error);
  }
}

$('composer').addEventListener('submit', async event => {
  event.preventDefault();
  const input = $('message-input');
  const text = input.value.trim();
  if (busy || !text) return;
  busy = true;
  $('send').disabled = true;
  $('replay').disabled = true;
  lastSpeech = null;
  speech.stop();
  // Initiate resume synchronously inside the user gesture, before network work.
  void speech.unlock().catch(error => console.warn('Audio unlock:', error));
  addMessage(text, 'user');
  input.value = '';
  setState('thinking');
  const abort = new AbortController();
  const timeout = setTimeout(() => abort.abort(), 180000);
  try {
    const response = await fetch('/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }), signal: abort.signal,
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(typeof data.detail === 'string' ? data.detail : `Server returned ${response.status}.`);
    }
    const data = await response.json();
    if (typeof data.reply !== 'string') throw new Error('The server returned an invalid reply.');
    addMessage(data.reply);
    emotion = typeof data.emotion === 'string' ? data.emotion : 'neutral';
    if (data.audio) {
      lastSpeech = data;
      await playReply(data);
    } else {
      $('sync-status').textContent = data.voice_error || 'Text-only reply';
    }
    motion?.play(data.gesture);
  } catch (error) {
    addMessage(error.name === 'AbortError' ? 'The request timed out. The server may still be finishing this turn; wait before retrying.' : error.message, 'error');
    if (!input.value) input.value = text;
  } finally {
    clearTimeout(timeout);
    busy = false;
    $('send').disabled = false;
    $('replay').disabled = !lastSpeech;
    setState(speech.playing ? 'talking' : 'idle');
    input.focus();
  }
});
$('replay').addEventListener('click', () => { if (!busy && lastSpeech) void playReply(lastSpeech); });
$('stop').addEventListener('click', () => speech.stop());
for (const id of ['volume', 'mute']) $(id).addEventListener('input', () => speech.setVolume(Number($('volume').value), $('mute').checked));
$('motion').addEventListener('change', () => motion?.setEnabled($('motion').checked));
reducedMotion.addEventListener('change', event => { $('motion').checked = !event.matches; motion?.setEnabled(!event.matches); });
$('preview').addEventListener('click', () => motion?.play($('gesture').value));
$('camera-view').addEventListener('change', fitCamera);
$('settings-toggle').addEventListener('click', () => {
  $('settings').hidden = !$('settings').hidden;
  $('settings-toggle').setAttribute('aria-expanded', String(!$('settings').hidden));
});
addEventListener('keydown', event => {
  if (event.key === 'Escape') { $('settings').hidden = true; $('settings-toggle').setAttribute('aria-expanded', 'false'); }
});
addEventListener('pointermove', event => pointer.set(event.clientX / innerWidth * 2 - 1, 1 - event.clientY / innerHeight * 2));
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  fitCamera();
});
document.addEventListener('visibilitychange', () => { if (document.hidden) speech.stop(); });
renderer.domElement.addEventListener('webglcontextlost', event => {
  event.preventDefault();
  speech.stop();
  showBootError('The graphics context was lost. Reload to restore Yumi.');
});
function showBootError(message) {
  $('boot').hidden = false;
  $('boot-error').textContent = message;
  $('boot-error').hidden = false;
  $('retry').hidden = false;
}

const loader = new GLTFLoader();
loader.register(parser => new VRMLoaderPlugin(parser));
fitCamera();
loader.load('/static/yumi.vrm', gltf => {
  try {
    vrm = gltf.userData.vrm;
    if (!vrm?.humanoid) throw new Error('The model does not contain a VRM humanoid rig.');
    VRMUtils.removeUnnecessaryVertices(gltf.scene);
    VRMUtils.removeUnnecessaryJoints(gltf.scene);
    // Correct VRM 0 orientation without turning VRM 1 avatars backwards.
    VRMUtils.rotateVRM0(vrm);
    vrm.scene.traverse(object => { object.frustumCulled = false; });
    scene.add(vrm.scene);
    const bounds = new THREE.Box3().setFromObject(vrm.scene);
    modelHeight = Math.max(.5, bounds.max.y - bounds.min.y);
    vrm.scene.position.y -= bounds.min.y;
    motion = new MotionController(vrm);
    motion.setEnabled($('motion').checked);
    if (vrm.lookAt) vrm.lookAt.target = gazeTarget;
    const available = MOUTH_NAMES.filter(hasExpression);
    $('rig-status').textContent = available.length ? `Available mouth shapes: ${available.join(', ')}` : 'This VRM has no standard mouth shapes; visible lip sync is unavailable.';
    $('preview').disabled = false;
    fitCamera();
    $('boot').hidden = true;
    setState(busy ? 'thinking' : speech.playing ? 'talking' : 'idle');
    motion.play('wave');
  } catch (error) {
    console.error('VRM setup:', error);
    showBootError(error.message);
  }
}, progress => {
  if (progress.total) {
    $('progress').value = progress.loaded / progress.total * 100;
    $('boot-text').textContent = `Loading Yumi… ${Math.round($('progress').value)}%`;
  }
}, error => {
  console.error('VRM load:', error);
  showBootError('Could not load /static/yumi.vrm. Check the model file and server, then retry.');
});

void (async () => {
  try {
    const response = await fetch('/health');
    if (!response.ok) throw new Error('Health check failed');
    const data = await response.json();
    const notes = [data.chat_configured ? 'Chat key configured (connection not tested).' : 'Set GEMINI_API_KEY on the server to enable chat.', data.voice_configured ? 'Voice key configured (connection not tested).' : 'Voice needs ELEVENLABS_API_KEY.', data.rhubarb_available ? 'Rhubarb recognition available.' : 'Rhubarb not installed: using approximate, timestamp-aligned mouth shapes.'];
    $('diagnostics').textContent = notes.join(' ');
  } catch {
    $('diagnostics').textContent = 'Server health could not be checked.';
  }
})();

let lastTime = performance.now();
function animate(now) {
  if (disposed) return;
  frameId = requestAnimationFrame(animate);
  const dt = Math.min((now - lastTime) / 1000, .05);
  lastTime = now;
  if (document.hidden) return;
  if (vrm) {
    motion?.update(dt);
    updateFace(dt);
    const follow = $('gaze').checked && !reducedMotion.matches;
    gazeTarget.position.set(follow ? pointer.x * .45 : 0, modelHeight * .87 + (follow ? pointer.y * .2 : 0), 2.5);
    vrm.update(dt);
  }
  controls.update();
  renderer.render(scene, camera);
}
frameId = requestAnimationFrame(animate);
addEventListener('pagehide', event => {
  speech.stop();
  if (event.persisted) return;
  disposed = true;
  cancelAnimationFrame(frameId);
  controls.dispose();
  motion?.mixer.stopAllAction();
  if (vrm) VRMUtils.deepDispose(vrm.scene);
  floor.geometry.dispose(); floor.material.dispose();
  ring.geometry.dispose(); ring.material.dispose();
  renderer.dispose();
  void speech.context?.close();
});
addMessage("Hi! I'm Yumi. Make yourself comfortable.");
