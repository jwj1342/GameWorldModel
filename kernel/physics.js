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
    // 三种刚体，对应三种位姿来源：
    //   fixed                  静态几何，位姿不变
    //   kinematicPositionBased 脚本运动，pose(t) 算出来推给物理
    //   dynamic                物理驱动，Rapier 算出来读回 group
    const m = entry.spec?.motion ?? {};
    if (entry.simulated) {
      if (m.trigger) throw new Error('dynamic trigger is not supported');
      const velocity = m.linear_velocity ?? [0, 0, 0];
      if (!Array.isArray(velocity) || velocity.length !== 3 || !velocity.every(Number.isFinite)) throw new Error('dynamic velocity must be a finite vec3');
      if (m.mass != null && (!Number.isFinite(m.mass) || m.mass <= 0)) throw new Error('dynamic mass must be finite and positive');
      if (m.restitution != null && (!Number.isFinite(m.restitution) || m.restitution < 0 || m.restitution > 1)) throw new Error('dynamic restitution must be within [0, 1]');
    }
    const desc = (entry.simulated ? RAPIER.RigidBodyDesc.dynamic()
                  : entry.kind === 'static' || !entry.dynamic ? RAPIER.RigidBodyDesc.fixed()
                  : RAPIER.RigidBodyDesc.kinematicPositionBased())
      .setTranslation(p.x, p.y, p.z).setRotation({ x: q.x, y: q.y, z: q.z, w: q.w });
    if (entry.simulated) {
      const lv = m.linear_velocity ?? [0, 0, 0];
      desc.setLinvel(lv[0], lv[1], lv[2]);
    }
    const body = this.world.createRigidBody(desc);
    entry.body = body; entry.colliderHandles = [];
    for (const c of entry.colliders) {
      const cd = colliderDescFor(c);
      if (entry.simulated) {
        if (m.restitution != null) cd.setRestitution(m.restitution);
        if (m.friction != null) cd.setFriction(m.friction);
      }
      const col = this.world.createCollider(cd, body);
      entry.colliderHandles.push(col.handle); this.colliderToEntry.set(col.handle, entry);
    }
    if (entry.simulated && m.mass != null) {
      // mass is the object's total, not a separate mass for every part.
      const colliders = entry.colliderHandles.map(handle => this.world.getCollider(handle));
      const volume = colliders.reduce((sum, col) => sum + col.volume(), 0);
      if (!Number.isFinite(volume) || volume <= 0) throw new Error('dynamic colliders need positive finite volume');
      for (const col of colliders) col.setMass(m.mass * col.volume() / volume);
      body.recomputeMassPropertiesFromColliders();
    }
    this.bodyToEntry.set(body.handle, entry);
  }

  // 物理驱动的物体：仿真之后把位姿读回 three.js，方向和 syncKinematic 相反
  syncSimulated(entry) {
    if (!entry.body || !entry.simulated) return;
    const t = entry.body.translation(), r = entry.body.rotation();
    entry.delta = new THREE.Vector3(t.x, t.y, t.z).sub(entry.group.position);
    entry.group.position.set(t.x, t.y, t.z);
    entry.group.quaternion.set(r.x, r.y, r.z, r.w);
  }

  // 把物理驱动的物体放回初始位姿与初速度，供 reset 使用
  respawnSimulated(entry) {
    if (!entry.body || !entry.simulated) return;
    const m = entry.spec?.motion ?? {};
    const b = entry.base, lv = m.linear_velocity ?? [0, 0, 0];
    entry.body.setTranslation({ x: b.pos[0], y: b.pos[1], z: b.pos[2] }, true);
    entry.body.setRotation({ x: b.quat[0], y: b.quat[1], z: b.quat[2], w: b.quat[3] }, true);
    entry.body.setLinvel({ x: lv[0], y: lv[1], z: lv[2] }, true);
    entry.body.setAngvel({ x: 0, y: 0, z: 0 }, true);   // 清掉碰撞攒下来的自转
    entry.delta = new THREE.Vector3();
    entry.group.position.set(b.pos[0], b.pos[1], b.pos[2]);
    entry.group.quaternion.set(b.quat[0], b.quat[1], b.quat[2], b.quat[3]);
  }
  removeEntry(entry) {
    if (!entry.body) return;
    this.world.removeRigidBody(entry.body); // removes attached colliders too
    for (const h of entry.colliderHandles) this.colliderToEntry.delete(h);
    this.bodyToEntry.delete(entry.body.handle); entry.body = null;
  }
  syncKinematic(entry) {
    if (!entry.body || !entry.dynamic || entry.simulated) return;   // 物理驱动的不往回推
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
