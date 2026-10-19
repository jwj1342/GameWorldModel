// 直接操纵：抓住、拖动、松手，物理一路都在。不开浏览器、不用 WebGL。
// Node 20: node --experimental-default-type=module --test tests/test_manipulate.mjs
import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import { Physics, initRapier } from '../kernel/physics.js';
import { buildScene } from '../kernel/scene.js';
import { Manipulator, canGrab } from '../kernel/manipulate.js';

await initRapier();
const DT = 1 / 60;

function world({ motion = { type: 'static' }, interaction } = {}) {
  const program = {
    static: [{ id: 'ground', class: 'ground', geom: { kind: 'primitive', shape: 'box', extent: [20, 0.4, 20] },
               pose: { pos: [0, -0.2, 0] } }],
    objects: [{ id: 'cup', class: 'cup', geom: { kind: 'primitive', shape: 'box', extent: [0.4, 0.4, 0.4] },
                pose: { pos: [0, 0.2, 0] }, motion, ...(interaction ? { interaction } : {}) }],
  };
  const { scene, registry } = buildScene(program);
  const physics = new Physics(DT);
  registry.forEach(e => physics.addEntry(e));
  physics.createPlayer(new THREE.Vector3(100, 100, 100));
  const camera = new THREE.PerspectiveCamera(50, 16 / 9, 0.1, 100);
  camera.position.set(0, 2, 5); camera.lookAt(0, 0.2, 0); camera.updateMatrixWorld(true);
  const manipulator = new Manipulator({ camera, registry, physics, scene, dt: DT });
  const step = (n = 1) => {
    for (let i = 0; i < n; i++) {
      manipulator.update();
      physics.step();
      for (const e of registry) physics.syncSimulated(e);
    }
  };
  return { scene, registry, physics, camera, manipulator, step,
           cup: registry.find(e => e.id === 'cup'), ground: registry.find(e => e.id === 'ground') };
}

test('静态结构抓不动，物体可以抓', () => {
  const w = world();
  assert.equal(canGrab(w.ground), false, '地面不该能被拖走');
  assert.equal(canGrab(w.cup), true);
});

test('程序里显式关掉的物体抓不动', () => {
  const w = world({ interaction: { grab: false } });
  assert.equal(canGrab(w.cup), false);
});

test('抓住拖到别处，物体确实跟过去了', () => {
  const w = world();
  assert.equal(w.manipulator.grab(w.cup, w.cup.group.position.clone()), true);
  w.manipulator.dragToPoint(2.0, 1.5, 0);
  w.step(60);
  const p = w.cup.group.position;
  assert.ok(p.distanceTo(new THREE.Vector3(2.0, 1.5, 0)) < 0.25,
    `拖过去之后应该停在目标附近，实际在 ${p.toArray().map(v => v.toFixed(2))}`);
});

test('抓住的时候重力不起作用，松手之后落回地面', () => {
  const w = world();
  w.manipulator.grab(w.cup, w.cup.group.position.clone());
  w.manipulator.dragToPoint(0, 3.0, 0);
  w.step(60);
  assert.ok(w.cup.group.position.y > 2.5, '拖到空中应该停得住');
  w.manipulator.release();
  w.step(180);
  assert.ok(w.cup.group.position.y < 0.5,
    `松手之后应该落回地面，实际 y=${w.cup.group.position.y.toFixed(2)}`);
  assert.ok(w.cup.group.position.y > -0.1, '不应该穿过地面');
});

test('拖动是推过去的，会把挡路的物体撞开', () => {
  const w = world();
  // 再放一个物体挡在路上，拖过去应该把它推走，而不是穿过去
  const blocker = { id: 'blocker', class: 'crate',
    geom: { kind: 'primitive', shape: 'box', extent: [0.4, 0.4, 0.4] },
    pose: { pos: [1.0, 0.2, 0] }, motion: { type: 'dynamic', mass: 1 } };
  const { scene, registry } = buildScene({ static: [{ id: 'ground', class: 'ground',
    geom: { kind: 'primitive', shape: 'box', extent: [20, 0.4, 20] }, pose: { pos: [0, -0.2, 0] } }],
    objects: [{ id: 'cup', class: 'cup', geom: { kind: 'primitive', shape: 'box', extent: [0.4, 0.4, 0.4] },
                pose: { pos: [0, 0.2, 0] }, motion: { type: 'static' } }, blocker] });
  const physics = new Physics(DT);
  registry.forEach(e => physics.addEntry(e));
  physics.createPlayer(new THREE.Vector3(100, 100, 100));
  const camera = new THREE.PerspectiveCamera(50, 16 / 9, 0.1, 100);
  camera.position.set(0, 2, 5); camera.lookAt(0, 0, 0); camera.updateMatrixWorld(true);
  const manipulator = new Manipulator({ camera, registry, physics, scene, dt: DT });
  const cup = registry.find(e => e.id === 'cup'), other = registry.find(e => e.id === 'blocker');
  const before = other.group.position.x;
  manipulator.grab(cup, cup.group.position.clone());
  manipulator.dragToPoint(1.8, 0.2, 0);
  for (let i = 0; i < 90; i++) { manipulator.update(); physics.step(); registry.forEach(e => physics.syncSimulated(e)); }
  assert.ok(other.group.position.x > before + 0.1,
    `挡路的物体应该被推开，x 从 ${before.toFixed(2)} 变成 ${other.group.position.x.toFixed(2)}`);
});

test('挡在前面的东西要真的挡住，看不见的不算', () => {
  // 相机在 +z 看向原点，挡板横在物体前面
  const { scene, registry } = buildScene({
    static: [{ id: 'ground', class: 'ground',
               geom: { kind: 'primitive', shape: 'box', extent: [20, 0.4, 20] }, pose: { pos: [0, -0.2, 0] } }],
    objects: [{ id: 'cup', class: 'cup', geom: { kind: 'primitive', shape: 'box', extent: [0.4, 0.4, 0.4] },
                pose: { pos: [0, 0.2, 0] }, motion: { type: 'static' } },
              { id: 'panel', class: 'panel', geom: { kind: 'primitive', shape: 'box', extent: [4, 4, 0.1] },
                pose: { pos: [0, 0.2, 2.0] }, motion: { type: 'static' }, interaction: { grab: false } }] });
  const physics = new Physics(DT);
  registry.forEach(e => physics.addEntry(e));
  physics.createPlayer(new THREE.Vector3(100, 100, 100));
  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0.2, 5); camera.lookAt(0, 0.2, 0); camera.updateMatrixWorld(true);
  const manipulator = new Manipulator({ camera, registry, physics, scene, dt: DT });
  const panel = registry.find(e => e.id === 'panel');

  assert.equal(manipulator.pick(0, 0), null, '隔着挡板不该能抓到后面的杯子');
  panel.group.visible = false;
  const hit = manipulator.pick(0, 0);
  assert.equal(hit?.entry?.id, 'cup', '挡板看不见了就不该再挡着');
});

test('抓取记录写下了谁被抓、放到哪', () => {
  const w = world();
  w.manipulator.grab(w.cup, w.cup.group.position.clone());
  w.manipulator.dragToPoint(1, 1, 0); w.step(30);
  w.manipulator.release();
  const events = w.manipulator.log.map(e => e.event);
  assert.deepEqual(events, ['grab', 'release']);
  assert.equal(w.manipulator.log[0].id, 'cup');
});
