// physics.js — Rapier：静态碰撞体、运动学刚体（跟随 pose(t)）、玩家胶囊 + KinematicCharacterController、平台携带
import * as THREE from 'three';
import RAPIER from '@dimforge/rapier3d-compat';

const DEG = Math.PI / 180;
let rapierReady = null;
export function initRapier() { if (!rapierReady) rapierReady = RAPIER.init(); return rapierReady; }
export { RAPIER };

function colliderDescFor(c) {
  // c: {shape, extent|size, radius, height, axis, offset}
  const off = [...(c.offset ?? [0, 0, 0])];   // 复制：plane 分支会改它，而 colliders 在 reset 后会被重用
  let desc;
  switch (c.shape) {
    case 'sphere': desc = RAPIER.ColliderDesc.ball(c.radius ?? 0.5); break;
    case 'cylinder': case 'cone': {
      const h = c.height ?? 1, r = c.radius ?? 0.5;
      desc = c.shape === 'cone' ? RAPIER.ColliderDesc.cone(h / 2, r) : RAPIER.ColliderDesc.cylinder(h / 2, r);
      if (c.axis === 'z') desc.setRotation(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2));
      else if (c.axis === 'x') desc.setRotation(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 2));
      break;
    }
    case 'plane': { const s = c.size ?? [10, 10]; desc = RAPIER.ColliderDesc.cuboid(s[0] / 2, 0.05, s[1] / 2); off[1] -= 0.05; break; }
    default: { const e = c.extent ?? [1, 1, 1]; desc = RAPIER.ColliderDesc.cuboid(e[0] / 2, e[1] / 2, e[2] / 2); }
  }
  desc.setTranslation(off[0], off[1], off[2]);
  desc.setFriction(0.8);
  return desc;
}

export class Physics {
  constructor(dt) {
    this.dt = dt;
    this.world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
    this.world.timestep = dt;
    this.bodyToEntry = new Map(); // body handle -> entry
    this.colliderToEntry = new Map();
    this.player = null;
  }
  addEntry(entry) {
    const p = entry.group.position, q = entry.group.quaternion;
    const desc = (entry.kind === 'static' || !entry.dynamic ? RAPIER.RigidBodyDesc.fixed() : RAPIER.RigidBodyDesc.kinematicPositionBased())
      .setTranslation(p.x, p.y, p.z).setRotation({ x: q.x, y: q.y, z: q.z, w: q.w });
    const body = this.world.createRigidBody(desc);
    entry.body = body; entry.colliderHandles = [];
    for (const c of entry.colliders) {
      const col = this.world.createCollider(colliderDescFor(c), body);
      entry.colliderHandles.push(col.handle); this.colliderToEntry.set(col.handle, entry);
    }
    this.bodyToEntry.set(body.handle, entry);
  }
  removeEntry(entry) {
    if (!entry.body) return;
    this.world.removeRigidBody(entry.body); // removes attached colliders too
    for (const h of entry.colliderHandles) this.colliderToEntry.delete(h);
    this.bodyToEntry.delete(entry.body.handle); entry.body = null;
  }
  syncKinematic(entry) {
    if (!entry.body || !entry.dynamic) return;
    const p = entry.group.position, q = entry.group.quaternion;
    entry.body.setNextKinematicTranslation({ x: p.x, y: p.y, z: p.z });
    entry.body.setNextKinematicRotation({ x: q.x, y: q.y, z: q.z, w: q.w });
  }
  createPlayer(pos, { radius = 0.35, halfHeight = 0.55 } = {}) {
    const body = this.world.createRigidBody(RAPIER.RigidBodyDesc.kinematicPositionBased().setTranslation(pos.x, pos.y, pos.z));
    const collider = this.world.createCollider(RAPIER.ColliderDesc.capsule(halfHeight, radius).setFriction(0.0), body);
    const controller = this.world.createCharacterController(0.02);
    controller.enableAutostep(0.45, 0.25, true);
    controller.enableSnapToGround(0.35);
    controller.setMaxSlopeClimbAngle(55 * DEG);
    controller.setMinSlopeSlideAngle(65 * DEG);
    controller.setUp({ x: 0, y: 1, z: 0 });
    this.player = { body, collider, controller, radius, halfHeight, vy: 0, grounded: false, groundEntry: null };
    return this.player;
  }
  teleportPlayer(pos) { this.player.body.setNextKinematicTranslation({ x: pos.x, y: pos.y, z: pos.z }); this.player.body.setTranslation({ x: pos.x, y: pos.y, z: pos.z }, true); this.player.vy = 0; }
  groundEntryBelow() {
    const pl = this.player; const t = pl.body.translation();
    const origin = { x: t.x, y: t.y - pl.halfHeight - pl.radius + 0.05, z: t.z };
    const ray = new RAPIER.Ray(origin, { x: 0, y: -1, z: 0 });
    const hit = this.world.castRay(ray, 0.6, true, undefined, undefined, pl.collider);
    if (!hit) return null;
    return this.colliderToEntry.get(hit.collider.handle) ?? null;
  }
  movePlayer(move, jump, { speed = 4.0, gravity = -20, jumpSpeed = 7.5 } = {}) {
    const pl = this.player, dt = this.dt;
    const carry = new THREE.Vector3();
    if (pl.grounded && pl.groundEntry?.dynamic && pl.groundEntry.delta) carry.copy(pl.groundEntry.delta);
    if (pl.grounded && jump) pl.vy = jumpSpeed;
    pl.vy += gravity * dt;
    const desired = { x: move.x * speed * dt + carry.x, y: pl.vy * dt + carry.y, z: move.z * speed * dt + carry.z };
    pl.controller.computeColliderMovement(pl.collider, desired);
    const mv = pl.controller.computedMovement();
    const t = pl.body.translation();
    pl.body.setNextKinematicTranslation({ x: t.x + mv.x, y: t.y + mv.y, z: t.z + mv.z });
    pl.grounded = pl.controller.computedGrounded();
    if (pl.grounded && pl.vy < 0) pl.vy = 0;
    if (!pl.grounded && desired.y > 0 && mv.y < desired.y * 0.5) pl.vy = Math.min(pl.vy, 0); // ceiling
    pl.groundEntry = pl.grounded ? this.groundEntryBelow() : null;
  }
  step() { this.world.step(); }
  playerPosition() { const t = this.player.body.translation(); return new THREE.Vector3(t.x, t.y, t.z); }
}
