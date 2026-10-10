// CPU Three.js/Rapier + actual control methods; no WebGL, browser or model.
// Node 20: node --experimental-default-type=module --test tests/test_physics_runtime.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import test from 'node:test';
import * as THREE from 'three';
import { Physics, initRapier } from '../kernel/physics.js';
import { buildScene } from '../kernel/scene.js';
await initRapier();
const DT = 1 / 60;
const source = fs.readFileSync(new URL('../kernel/main.js', import.meta.url), 'utf8');
assert(source.includes('  const hasSimulated') && source.includes('  updateMotions(0); game.mode'));
const control = source.slice(source.indexOf('  const hasSimulated'), source.indexOf('  updateMotions(0); game.mode'));
function object(motion = {type: 'dynamic'}, geom = {kind: 'primitive', shape: 'sphere', radius: 0.25}) {
  return {id: 'synthetic', class: 'generic', pose: {pos: [0, 4, 0], quat: [0, 0, 0, 1]}, geom, motion};
}
function harness(objects) {
  const {scene, registry} = buildScene({objects});
  const physics = new Physics(DT);
  registry.forEach(e => physics.addEntry(e));
  physics.createPlayer(new THREE.Vector3(100, 100, 100));
  const template = {camera: {}, playerMesh: {}, goalMesh: {}, spawn: new THREE.Vector3(100, 100, 100), state: () => ({}), update: () => {}};
  const replay = {camera: {}, update: () => {}};
  const game = {t: 0, frame: 0, modeName: 'replay', input: {}};
  vm.runInNewContext(control, {registry, physics, template, replay, game, DT, scene, renderer: {}, far: 100});
  return {game, physics, registry};
}
test('composite total mass is volume-weighted and independent of part count', () => {
  const geom = {kind: 'composite', parts: [
    {shape: 'sphere', radius: 0.25, offset: [-1, 0, 0]},
    {shape: 'sphere', radius: 0.5, offset: [1, 0, 0]},
  ]};
  const h = harness([object({type: 'dynamic', mass: 9}, geom)]);
  try {
    const e = h.registry[0];
    assert(Math.abs(e.body.mass() - 9) < 1e-5);
    const masses = e.colliderHandles.map(handle => h.physics.world.getCollider(handle).mass());
    assert(Math.abs(masses[0] - 1) < 1e-5 && Math.abs(masses[1] - 8) < 1e-5);
    h.game.reset();
    assert(Math.abs(e.body.mass() - 9) < 1e-5);
  } finally { h.physics.world.free(); }
});
test('single collider mass and default-density mass remain usable', () => {
  for (const mass of [undefined, 2]) {
    const h = harness([object({type: 'dynamic', mass})]);
    try { assert(mass == null ? h.registry[0].body.mass() > 0 : Math.abs(h.registry[0].body.mass() - mass) < 1e-5); }
    finally { h.physics.world.free(); }
  }
});
test('dynamic step rejects unsupported dt without advancing state', () => {
  const h = harness([object()]);
  try {
    for (const dt of [0.5, DT / 2, NaN, Infinity, 0, -1]) assert.throws(() => h.game.step(dt));
    assert.equal(h.game.t, 0); assert.equal(h.game.frame, 0);
    h.game.step();
    assert.equal(h.game.t, DT); assert(h.registry[0].group.position.y < 4);
  } finally { h.physics.world.free(); }
});
test('seek quantization is explicit, stable and reset/replay agrees', () => {
  const h = harness([object()]);
  try {
    const requested = 1 / 24;
    assert.equal(h.game.seek(requested), 3 * DT);
    assert.equal(h.game.frame, 3);
    const first = h.registry[0].group.position.clone();
    h.game.seek(requested);
    assert.equal(h.game.frame, 3); assert(first.distanceTo(h.registry[0].group.position) < 1e-7);
    for (const invalid of [-1, NaN, Infinity]) assert.throws(() => h.game.seek(invalid));
    assert.equal(h.game.frame, 3);
    h.game.seek(0.5);
    const final = h.registry[0].group.position.clone();
    h.game.seek(0); h.game.seek(0.5);
    assert(final.distanceTo(h.registry[0].group.position) < 1e-7);
  } finally { h.physics.world.free(); }
});
test('legacy scripted seek remains exact and fixed-step callers work', () => {
  const h = harness([object({type: 'prismatic', axis: [1, 0, 0], rate: 1})]);
  try { assert.equal(h.game.seek(1 / 24), 1 / 24); h.game.step(DT); }
  finally { h.physics.world.free(); }
});
test('existing collision scene stays finite and replays identically on CPU', () => {
  const program = JSON.parse(fs.readFileSync(new URL('../examples/collision_chain/program.json', import.meta.url), 'utf8'));
  const {scene, registry} = buildScene(program);
  const physics = new Physics(DT);
  registry.forEach(e => physics.addEntry(e));
  physics.createPlayer(new THREE.Vector3(100, 100, 100));
  const template = {camera: {}, playerMesh: {}, goalMesh: {}, spawn: new THREE.Vector3(100, 100, 100), state: () => ({}), update: () => {}};
  const game = {t: 0, frame: 0, modeName: 'replay', input: {}};
  vm.runInNewContext(control, {registry, physics, template, replay: {camera: {}, update: () => {}}, game, DT, scene, renderer: {}, far: 100});
  try {
    for (let i = 0; i < 180; i++) game.seek(i / 30);
    const final = registry.filter(e => e.simulated).map(e => e.group.position.clone());
    assert(final.every(p => p.toArray().every(Number.isFinite)));
    assert(final.at(-1).distanceTo(new THREE.Vector3(...registry.filter(e => e.simulated).at(-1).base.pos)) > 1);
    game.reset(); game.seek(179 / 30);
    registry.filter(e => e.simulated).forEach((e, i) => assert(e.group.position.distanceTo(final[i]) < 1e-6));
  } finally { physics.world.free(); }
});
test('runtime guard rejects nonfinite values and unsupported trigger before creating body', () => {
  const physics = new Physics(DT);
  try {
    for (const motion of [{type:'dynamic',mass:Infinity}, {type:'dynamic',mass:0},
      {type:'dynamic',linear_velocity:[NaN,0,0]}, {type:'dynamic',linear_velocity:[1,2]},
      {type:'dynamic',linear_velocity:1}, {type:'dynamic',restitution:2}, {type:'dynamic',trigger:true}]) {
      const {registry} = buildScene({objects:[object(motion)]});
      assert.throws(() => physics.addEntry(registry[0]));
      assert.equal(physics.bodyToEntry.size, 0);
    }
  } finally { physics.world.free(); }
});

async function runHarnessScript(relative, game) {
  const files = new Map();
  const mockFs = {mkdirSync: () => {}, createWriteStream: file => ({
    write: value => files.set(file, (files.get(file) ?? '') + value), end: () => {},
  }), writeFileSync: (file, value) => files.set(file, value)};
  let closed = false;
  const g = {logs: [], close: async () => {closed = true;}, page: {evaluate: async (fn, value) => fn(value)}};
  const text = fs.readFileSync(new URL(relative, import.meta.url), 'utf8').replace(/^import .*;\r?\n/gm, '');
  const context = {fs: mockFs, path, process: {argv: []}, parseArgs: () => ({game: 'synthetic', out: 'synthetic-out', fps: 24, duration: 0.125, times: '0,0.041666666666666664', passes: 'rgb'}),
    openGame: async () => g, savePng: () => {}, window: {__game: game}, console: {log: () => {}}, Date, Number};
  await vm.runInNewContext(`(async () => {${text}\n})()`, context);
  assert(closed);
  return files;
}
test('record script labels actual state time and retains requested encoded time', async () => {
  const h = harness([object()]);
  h.game.render = () => ''; h.game.cameraInfo = () => ({});
  try {
    const files = await runHarnessScript('../harness/record.mjs', h.game);
    const rows = [...files.values()].find(value => value.includes('"requested_t"')).trim().split('\n').map(line => JSON.parse(line));
    assert.equal(rows[1].requested_t, 1 / 24);
    assert.equal(rows[1].t, 3 * DT); assert.equal(rows[1].actual_t, rows[1].t);
    assert.equal(rows.length, 3);
  } finally { h.physics.world.free(); }
});
test('render index retains requested time while labeling actual frame time', async () => {
  const h = harness([object()]);
  h.game.render = () => ''; h.game.cameraInfo = () => ({});
  try {
    const files = await runHarnessScript('../harness/render.mjs', h.game);
    const index = JSON.parse([...files.values()].find(value => value.includes('"requested_t"')));
    assert.equal(index.time_sampling.mode, 'nearest_physics_frame');
    assert.equal(index.time_sampling.dt, DT);
    assert.equal(index.frames[1].requested_t, 1 / 24);
    assert.equal(index.frames[1].t, 3 * DT);
    assert.equal(index.frames[1].state.t, index.frames[1].actual_t);
  } finally { h.physics.world.free(); }
});
