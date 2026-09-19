// platformer_3p.js — 第三人称平台跳跃模板：出生点、目标体、收集物、危险物、跟随相机、胜负
import * as THREE from 'three';

function aabbOf(entry) { entry.group.updateMatrixWorld(true); return new THREE.Box3().setFromObject(entry.group); }

export function inferSlots(program, registry) {
  // 默认槽位推断（binder.py 未填时兜底）：可行走面 = 静态几何顶面
  const slots = { ...(program.binding?.slots ?? {}) };
  const statics = registry.filter(e => e.kind === 'static');
  const boxes = statics.map(aabbOf);
  const world = boxes.length ? boxes.reduce((a, b) => a.union(b), boxes[0].clone()) : new THREE.Box3(new THREE.Vector3(-5, 0, -5), new THREE.Vector3(5, 0, 5));
  const ground = boxes.length ? boxes.reduce((a, b) => (b.max.y - b.min.y) < (a.max.y - a.min.y) ? b : a) : world; // flattest static
  const topY = ground.max.y;
  if (!slots.player_spawn) slots.player_spawn = [ground.min.x + (ground.max.x - ground.min.x) * 0.2, topY + 1.2, ground.max.z - (ground.max.z - ground.min.z) * 0.2];
  if (!slots.goal_volume) slots.goal_volume = { pos: [ground.max.x - (ground.max.x - ground.min.x) * 0.15, topY + 1.0, ground.min.z + (ground.max.z - ground.min.z) * 0.15], extent: [1.5, 2, 1.5] };
  if (!slots.collectibles) slots.collectibles = registry.filter(e => e.kind === 'object' && /coin|gem|key|star|pickup/i.test(e.class)).map(e => e.id).filter((v, i, a) => a.indexOf(v) === i);
  if (!slots.hazards) slots.hazards = registry.filter(e => /lava|spike|water|fire|acid/i.test(e.class) || /lava|spike/i.test(e.spec.material ?? '')).map(e => e.id).filter((v, i, a) => a.indexOf(v) === i);
  return slots;
}

export class PlatformerTemplate {
  constructor({ program, scene, registry, physics }) {
    this.program = program; this.scene = scene; this.registry = registry; this.physics = physics;
    this.slots = inferSlots(program, registry);
    this.spawn = new THREE.Vector3(...this.slots.player_spawn);
    this.goal = { pos: new THREE.Vector3(...this.slots.goal_volume.pos), extent: new THREE.Vector3(...(this.slots.goal_volume.extent ?? [1.5, 2, 1.5])) };
    this.goalBox = new THREE.Box3().setFromCenterAndSize(this.goal.pos, this.goal.extent);
    this.collectibleIds = new Set(this.slots.collectibles ?? []);
    this.hazardIds = new Set(this.slots.hazards ?? []);
    this.collected = 0; this.deaths = 0; this.won = false; this.wonAt = null;
    this.collectiblesTotal = registry.filter(e => this.collectibleIds.has(e.id) || e.events.some(ev => ev.type === 'despawn_on_contact')).length;
    // visuals
    this.playerMesh = new THREE.Mesh(new THREE.CapsuleGeometry(0.35, 1.1, 6, 12), new THREE.MeshStandardMaterial({ color: 0x2f6fed, roughness: 0.5 }));
    this.playerMesh.castShadow = true; this.playerMesh.name = 'player'; scene.add(this.playerMesh);
    const goalMesh = new THREE.Mesh(new THREE.BoxGeometry(this.goal.extent.x, this.goal.extent.y, this.goal.extent.z),
      new THREE.MeshStandardMaterial({ color: 0x33ff88, transparent: true, opacity: 0.35, emissive: 0x11aa44, emissiveIntensity: 0.6 }));
    goalMesh.position.copy(this.goal.pos); goalMesh.name = 'goal'; scene.add(goalMesh); this.goalMesh = goalMesh;
    // physics player
    physics.createPlayer(this.spawn);
    // camera
    this.camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.05, 300);
    this.camYaw = 0; this.camDist = 6.5; this.camHeight = 3.0;
    this.updateCamera(true);
  }
  respawn() { this.physics.teleportPlayer(this.spawn); this.deaths++; }
  update(input, t) {
    const ph = this.physics;
    // move: world-space [x,z] if given, else from keys relative to camera yaw
    let mv = new THREE.Vector3();
    if (input.move) mv.set(input.move[0], 0, input.move[1]);
    else if (input.keys?.size) {
      const f = new THREE.Vector3(Math.sin(this.camYaw), 0, Math.cos(this.camYaw)).multiplyScalar(-1); // camera forward on ground
      const r = new THREE.Vector3(-f.z, 0, f.x);
      if (input.keys.has('w') || input.keys.has('arrowup')) mv.add(f);
      if (input.keys.has('s') || input.keys.has('arrowdown')) mv.sub(f);
      if (input.keys.has('a') || input.keys.has('arrowleft')) mv.sub(r);
      if (input.keys.has('d') || input.keys.has('arrowright')) mv.add(r);
    }
    if (mv.lengthSq() > 1) mv.normalize();
    const jump = !!input.jump || !!(input.keys?.has(' ') || input.keys?.has('space'));
    ph.movePlayer(mv, jump);
    const p = ph.playerPosition();
    this.playerMesh.position.copy(p);
    if (mv.lengthSq() > 1e-4) this.playerMesh.rotation.y = Math.atan2(mv.x, mv.z);
    // events
    const feet = p.y - ph.player.halfHeight - ph.player.radius;
    if (p.y < -25) this.respawn();
    for (const e of this.registry) {
      if (!e.visible || e.kind !== 'object') continue;
      const box = aabbOf(e).expandByScalar(ph.player.radius * 0.6);
      const inside = box.containsPoint(p) || box.containsPoint(new THREE.Vector3(p.x, feet + 0.1, p.z));
      if (!inside) continue;
      if (this.collectibleIds.has(e.id) || e.events.some(ev => ev.type === 'despawn_on_contact' && (ev.with ?? 'player') === 'player')) {
        e.visible = false; e.group.visible = false; ph.removeEntry(e); this.collected++; continue;
      }
      if (this.hazardIds.has(e.id)) { this.respawn(); break; }
      for (const ev of e.events) if (ev.type === 'trigger_on_enter' && ev.target) {
        const target = this.registry.find(x => x.id === ev.target && x.triggerTime == null); if (target) target.triggerTime = t;
      }
    }
    for (const e of this.registry) if (e.kind === 'static' && this.hazardIds.has(e.id)) { if (aabbOf(e).containsPoint(new THREE.Vector3(p.x, feet + 0.05, p.z))) { this.respawn(); break; } }
    if (!this.won && this.goalBox.containsPoint(p)) { this.won = true; this.wonAt = t; }
    this.updateCamera(false);
  }
  updateCamera(snap) {
    const p = this.physics.playerPosition();
    const target = new THREE.Vector3(p.x - Math.sin(this.camYaw) * this.camDist * -1, p.y + this.camHeight, p.z - Math.cos(this.camYaw) * this.camDist * -1);
    if (snap) this.camera.position.copy(target); else this.camera.position.lerp(target, 0.15);
    this.camera.lookAt(p.x, p.y + 0.8, p.z);
    this.camera.updateMatrixWorld();
  }
  state() {
    const ph = this.physics; const p = ph.playerPosition();
    return { pos: [p.x, p.y, p.z], vy: ph.player.vy, grounded: ph.player.grounded, ground: ph.player.groundEntry?.name ?? null,
      spawn: this.spawn.toArray(), goal: { pos: this.goal.pos.toArray(), extent: this.goal.extent.toArray() },
      collected: this.collected, collectibles_total: this.collectiblesTotal, deaths: this.deaths, won: this.won, won_at: this.wonAt };
  }
}
