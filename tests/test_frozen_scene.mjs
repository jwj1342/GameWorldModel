import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { buildScene } from '../kernel/scene.js';
import { makeReplayCamera } from '../kernel/replay_camera.js';
import { freezeFrame } from '../harness/frozen_scene.mjs';

const source = JSON.parse(fs.readFileSync(new URL('../examples/cross_engine/program.json', import.meta.url)));
function fixture(t = 0) {
  const program = structuredClone(source);
  const { scene, registry } = buildScene(program);
  const replay = makeReplayCamera(program.camera, 4 / 3);
  replay.update(t);
  for (const e of registry) { const p = e.pose(t); e.group.position.copy(p.pos); e.group.quaternion.copy(p.quat); }
  global.window = { __game: { t, sceneOnly: true, scene, registry, template: null,
    state: () => ({ t }), cameraInfo: () => ({ pos: replay.camera.position.toArray(),
      quat: replay.camera.quaternion.toArray(), fov: replay.camera.fov, aspect: replay.camera.aspect,
      near: replay.camera.near, far: replay.camera.far }) } };
  return window.__game;
}
const freeze = () => freezeFrame({ width: 320, height: 240 });
test('two instances keep distinct identity and the live state is not mutated', () => {
  const g = fixture();
  const before = g.registry.map(e => e.group.position.toArray());
  const packet = freeze();
  assert.equal(new Set(packet.meshes.map(m => m.key)).size, 5);
  assert.ok(packet.meshes.some(m => m.key === 'markers#2/0'));
  assert.deepEqual(g.registry.map(e => e.group.position.toArray()), before);
});
test('actual production motion reaches expected platform height and door rotation', () => {
  fixture(2);
  const packet = freeze();
  const platform = packet.meshes.find(m => m.object_id === 'platform');
  const door = packet.meshes.find(m => m.object_id === 'door');
  assert.ok(Math.abs(platform.matrix_world[13] - 1.3) < 1e-12);
  assert.ok(Math.abs(door.matrix_world[12] - 1) < 1e-12);
  assert.ok(Math.abs(door.matrix_world[14] + 0.5) < 1e-12);
});
test('non-rendered ancestry stays invisible', () => {
  const g = fixture(); g.registry[0].group.visible = false;
  assert.equal(freeze().meshes[0].visible, false);
});
for (const mode of ['gameplay', 'dynamic', 'asset', 'event', 'texture', 'aspect']) {
  test(`reject unsupported ${mode}`, () => {
    const g = fixture(), e = g.registry[0];
    if (mode === 'gameplay') g.sceneOnly = false;
    if (mode === 'dynamic') e.simulated = true;
    if (mode === 'asset') e.spec.geom.kind = 'asset';
    if (mode === 'event') e.events.push({ type: 'despawn_on_contact' });
    if (mode === 'texture') e.group.children[0].material.map = {};
    if (mode === 'aspect') g.cameraInfo = () => ({ aspect: 1 });
    assert.throws(freeze);
  });
}
