import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const root = process.env.GWM_TEST_CODE_GAME;
if (!root) throw new Error('Run through test_code_scene_host.py with an isolated bundle');
const fromGame = rel => pathToFileURL(path.join(root, rel)).href;
const THREE = await import(fromGame('vendor/three/build/three.module.js'));
const { buildScene: buildSceneFromCode } = await import(fromGame('kernel/scene.js'));
const { initRapier, Physics } = await import(fromGame('kernel/physics.js'));
await initRapier();

function near(actual, expected, tolerance = 1e-5) {
  assert.equal(actual.length, expected.length);
  actual.forEach((v, i) => assert.ok(Math.abs(v - expected[i]) <= tolerance, `${actual} != ${expected}`));
}
function mesh(size = [2, 1, 3]) {
  return new THREE.Mesh(new THREE.BoxGeometry(...size), new THREE.MeshStandardMaterial());
}
function build(described) {
  globalThis.__hostFixture = () => described;
  return buildSceneFromCode({}, {});
}
function withPhysics(registry, run) {
  const physics = new Physics(1 / 60);
  try { registry.forEach(e => physics.addEntry(e)); run(physics); }
  finally { physics.world.free(); }
}
function xyz(value) { return [value.x, value.y, value.z]; }

test('large translated ground uses full dimensions and player stands away from origin', () => {
  const ground = mesh([24, .5, 16]); ground.position.set(0, -.25, 0);
  const { registry } = build([{ id: 'floor', kind: 'static', object3D: ground }]);
  const entry = registry[0];
  near(entry.base.pos, [0, -.25, 0]);
  near(entry.colliders[0].extent, [24, .5, 16]);
  near(entry.group.children[0].getWorldPosition(new THREE.Vector3()).toArray(), entry.base.pos);
  near(ground.position.toArray(), [0, -.25, 0]); // Input is untouched.
  withPhysics(registry, physics => {
    const collider = physics.world.getCollider(entry.colliderHandles[0]);
    near(xyz(collider.halfExtents()), [12, .25, 8]);
    near(xyz(collider.translation()), [0, -.25, 0]);
    physics.createPlayer(new THREE.Vector3(5, 2, 0));
    for (let i = 0; i < 180; i++) { physics.movePlayer(new THREE.Vector3(), false); physics.step(); }
    assert.ok(physics.player.grounded);
    assert.ok(physics.playerPosition().y > .8 && physics.playerPosition().y < 1.1);
  });
});

test('absolute pose is applied once and mesh, registry and body agree over time', () => {
  const object = mesh(); object.position.set(4, 2, -5);
  const pose = t => ({ pos: [4 + t, 2, -5], quat: [0, Math.sin(t / 2), 0, Math.cos(t / 2)] });
  const { scene, registry } = build([{ id: 'moving', object3D: object, pose }]);
  const entry = registry[0];
  withPhysics(registry, physics => {
    for (const t of [0, .25, 1]) {
      const p = entry.pose(t); entry.group.position.copy(p.pos); entry.group.quaternion.copy(p.quat);
      physics.syncKinematic(entry); physics.step(); scene.updateMatrixWorld(true);
      near(entry.group.children[0].getWorldPosition(new THREE.Vector3()).toArray(), pose(t).pos);
      near(xyz(entry.body.translation()), pose(t).pos);
      assert.ok(entry.group.children[0].getWorldQuaternion(new THREE.Quaternion()).angleTo(p.quat) < 1e-6);
      const q = entry.body.rotation();
      assert.ok(new THREE.Quaternion(q.x, q.y, q.z, q.w).angleTo(p.quat) < 1e-3);
    }
  });
  near(object.position.toArray(), [4, 2, -5]);
});

test('nested offset, root scale and rotation produce local collider coordinates', () => {
  const object = new THREE.Group(); const child = mesh([2, 1, 1]); child.position.set(2, 0, 0);
  object.add(child); object.scale.set(2, 1, 1); object.position.set(3, 2, 1); object.rotation.y = Math.PI / 2;
  const { scene, registry } = build([{ id: 'compound', object3D: object }]);
  scene.updateMatrixWorld(true);
  const entry = registry[0]; near(entry.colliders[0].extent, [4, 1, 1]); near(entry.colliders[0].offset, [4, 0, 0]);
  near(entry.group.children[0].children[0].getWorldPosition(new THREE.Vector3()).toArray(), [3, 2, -3]);
  withPhysics(registry, physics => {
    const collider = physics.world.getCollider(entry.colliderHandles[0]);
    near(xyz(collider.halfExtents()), [2, .5, .5]); near(xyz(collider.translation()), [3, 2, -3]);
  });
});

test('pose orientation replaces initial root rotation rather than composing twice', () => {
  const object = mesh(); object.rotation.x = Math.PI / 2;
  const { registry } = build([{ id: 'oriented', object3D: object,
    pose: () => ({ pos: [1, 2, 3], quat: [0, 0, 0, 1] }) }]);
  assert.ok(registry[0].group.children[0].getWorldQuaternion(new THREE.Quaternion()).angleTo(new THREE.Quaternion()) < 1e-6);
});

test('checked-in handwritten example preserves its initial geometry orientation', async () => {
  const { describe } = await import(fromGame('kernel/handwritten_fixture.js'));
  const { scene, registry } = build(describe(THREE));
  scene.updateMatrixWorld(true);
  for (const entry of registry) {
    near(entry.group.children[0].getWorldPosition(new THREE.Vector3()).toArray(), entry.group.position.toArray());
  }
  const disc = registry.find(entry => entry.class === 'coin');
  const orientation = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2);
  assert.ok(disc.group.children[0].getWorldQuaternion(new THREE.Quaternion()).angleTo(orientation) < 1e-6);
});

test('duplicate IDs, missing geometry, static motion and hidden parent transforms are rejected', () => {
  assert.throws(() => build([{ id: 'same', object3D: mesh() }, { id: 'same', object3D: mesh() }]), /ID/);
  assert.throws(() => build([{ id: 'empty', object3D: new THREE.Group() }]), /包围盒/);
  assert.throws(() => build([{ object3D: mesh(), kind: 'static', pose: () => ({}) }]), /static/);
  const parent = new THREE.Group(), child = mesh(); parent.add(child);
  assert.throws(() => build([{ object3D: child }]), /独立根/);
});

test('invalid initial and sampled poses cannot become apparently valid transforms', () => {
  const object = mesh(); object.position.x = Infinity;
  assert.throws(() => build([{ object3D: object }]), /有限/);
  for (const value of [null, {pos: [NaN, 0, 0]}, {quat: [0, 0, 0, 0]}, {quat: [0, 0, 1]}]) {
    assert.throws(() => build([{ object3D: mesh(), pose: () => value }]), /pose/);
  }
  const { registry } = build([{ object3D: mesh(), pose: t => t ? {pos: [0, Infinity, 0]} : {} }]);
  assert.throws(() => registry[0].pose(1), /有限/);
});

for (const scope of ['root', 'child']) {
  for (const flag of ['matrixAutoUpdate', 'matrixWorldAutoUpdate']) {
    test(`manual ${flag} on ${scope} is rejected without changing the input`, () => {
      const object = scope === 'root' ? mesh() : new THREE.Group();
      const controlled = scope === 'root' ? object : mesh();
      if (scope === 'child') object.add(controlled);
      controlled.position.set(4, 2, -5); controlled.updateMatrix();
      controlled[flag] = false;
      const originalMatrix = controlled.matrix.toArray();
      const description = { object3D: object,
        pose: () => ({ pos: [4, 2, -5], quat: [0, 0, 0, 1] }) };
      assert.throws(() => build([description]), /自动矩阵更新/);
      near(controlled.position.toArray(), [4, 2, -5]);
      assert.deepEqual(controlled.matrix.toArray(), originalMatrix);
      assert.equal(controlled[flag], false);
      assert.equal(controlled.parent, scope === 'root' ? null : object);
    });
  }
}
