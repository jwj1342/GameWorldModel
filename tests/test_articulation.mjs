// 关节交互：抽屉沿一根轴滑，柜门绕一根轴转，都有限位。
// Node 20: node --experimental-default-type=module --test tests/test_articulation.mjs
import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import { Physics, initRapier } from '../kernel/physics.js';
import { buildScene } from '../kernel/scene.js';
import { Manipulator, canGrab, interactionOf } from '../kernel/manipulate.js';

await initRapier();
const DT = 1 / 60;

function world(interaction, pos = [0, 0.5, 0]) {
  const program = {
    static: [{ id: 'ground', class: 'ground', geom: { kind: 'primitive', shape: 'box', extent: [20, 0.4, 20] },
               pose: { pos: [0, -0.2, 0] } }],
    objects: [{ id: 'part', class: 'drawer', geom: { kind: 'primitive', shape: 'box', extent: [0.6, 0.3, 0.5] },
                pose: { pos: [...pos] }, motion: { type: 'static' }, interaction }],
  };
  const { scene, registry } = buildScene(program);
  const physics = new Physics(DT);
  registry.forEach(e => physics.addEntry(e));
  physics.createPlayer(new THREE.Vector3(100, 100, 100));
  const camera = new THREE.PerspectiveCamera(50, 16 / 9, 0.1, 100);
  camera.position.set(0, 2, 5); camera.lookAt(0, 0.5, 0); camera.updateMatrixWorld(true);
  const manipulator = new Manipulator({ camera, registry, physics, scene, dt: DT });
  const part = registry.find(e => e.id === 'part');
  const step = (n) => { for (let i = 0; i < n; i++) { manipulator.update(); physics.step(); registry.forEach(e => physics.syncSimulated(e)); } };
  return { manipulator, part, step };
}

test('没写 interaction 的物体默认自由拖动', () => {
  assert.deepEqual(interactionOf({ spec: {} }), { mode: 'free' });
  assert.equal(interactionOf({ spec: { interaction: { grab: 'none' } } }), null);
  assert.equal(interactionOf({ spec: { interaction: { grab: false } } }), null);
});

test('轴写成零向量时退回自由拖动，而不是算出 NaN', () => {
  assert.deepEqual(interactionOf({ spec: { interaction: { grab: 'slide', axis: [0, 0, 0] } } }), { mode: 'free' });
});

test('抽屉只沿轴走，垂直方向的拖动被吃掉', () => {
  const w = world({ grab: 'slide', axis: [0, 0, 1], limit: [0, 0.4] });
  w.manipulator.grab(w.part, w.part.group.position.clone());
  w.manipulator.dragToPoint(2.0, 3.0, 0.25);          // 大幅往上、往右拖
  w.step(90);
  const p = w.part.group.position;
  assert.ok(Math.abs(p.x) < 0.05, `x 不该动，实际 ${p.x.toFixed(3)}`);
  assert.ok(Math.abs(p.y - 0.5) < 0.05, `y 不该动，实际 ${p.y.toFixed(3)}`);
  assert.ok(Math.abs(p.z - 0.25) < 0.05, `z 该跟到 0.25，实际 ${p.z.toFixed(3)}`);
});

test('抽屉拉不出限位', () => {
  const w = world({ grab: 'slide', axis: [0, 0, 1], limit: [0, 0.4] });
  w.manipulator.grab(w.part, w.part.group.position.clone());
  w.manipulator.dragToPoint(0, 0.5, 5.0);             // 想拉到五米外
  w.step(120);
  assert.ok(w.part.group.position.z <= 0.42, `最多只能到 0.4，实际 ${w.part.group.position.z.toFixed(3)}`);
  w.manipulator.dragToPoint(0, 0.5, -5.0);            // 反向推过头
  w.step(120);
  assert.ok(w.part.group.position.z >= -0.02, `反向最多回到 0，实际 ${w.part.group.position.z.toFixed(3)}`);
});

test('柜门绕铰链转，而且真的转了朝向', () => {
  // 门板中心在铰链右侧 0.3 m，绕 y 轴开
  const w = world({ grab: 'hinge', axis: [0, 1, 0], pivot: [-0.3, 0.5, 0], limit: [0, Math.PI / 2] },
                  [0, 0.5, 0]);
  w.manipulator.grab(w.part, w.part.group.position.clone());
  // 注意方向：绕 +y 轴、门板在铰链 +x 侧时，要往 -z 扳才是正角度。
  // 往 +z 扳得到的是负角度，会被 [0, π/2] 这个限位正确地卡在 0——这本身也是对的行为。
  w.manipulator.dragToPoint(-0.3, 0.5, -0.3);         // 往外扳九十度
  w.step(150);
  const yaw = new THREE.Euler().setFromQuaternion(w.part.group.quaternion, 'YXZ').y;
  assert.ok(Math.abs(yaw) > 0.5, `门该转起来，实际 yaw=${yaw.toFixed(3)}`);
  const r = w.part.group.position.distanceTo(new THREE.Vector3(-0.3, 0.5, 0));
  assert.ok(Math.abs(r - 0.3) < 0.08, `中心该留在铰链半径上，实际 ${r.toFixed(3)}`);
});

test('柜门开不过限位', () => {
  const w = world({ grab: 'hinge', axis: [0, 1, 0], pivot: [-0.3, 0.5, 0], limit: [0, 0.4] });
  w.manipulator.grab(w.part, w.part.group.position.clone());
  w.manipulator.dragToPoint(-0.3, 0.5, -3.0);         // 使劲往开的方向扳
  w.step(150);
  const yaw = Math.abs(new THREE.Euler().setFromQuaternion(w.part.group.quaternion, 'YXZ').y);
  assert.ok(yaw > 0.2, `该转到限位附近，实际 ${yaw.toFixed(3)}`);
  assert.ok(yaw <= 0.5, `限位是 0.4 弧度，不该超过，实际 ${yaw.toFixed(3)}`);
});

test('关掉抓取的关节件抓不动', () => {
  const w = world({ grab: 'none', axis: [0, 0, 1] });
  assert.equal(canGrab(w.part), false);
});

test('往关门方向扳被限位卡住，不会转成负角', () => {
  const w = world({ grab: 'hinge', axis: [0, 1, 0], pivot: [-0.3, 0.5, 0], limit: [0, Math.PI / 2] });
  w.manipulator.grab(w.part, w.part.group.position.clone());
  w.manipulator.dragToPoint(-0.3, 0.5, 2.0);          // 反方向
  w.step(120);
  const yaw = new THREE.Euler().setFromQuaternion(w.part.group.quaternion, 'YXZ').y;
  assert.ok(Math.abs(yaw) < 0.15, `应该停在关着的位置，实际 ${yaw.toFixed(3)}`);
});
