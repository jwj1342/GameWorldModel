// scene.js — 把 program.json 的 static[] 与 objects[] 变成 three.js 节点，并给每个物体分配 24 位 ID 颜色
import * as THREE from 'three';
import { makePose } from './motions.js';

const MATERIALS = {
  default: { color: 0x9a9a9a, roughness: 0.8 },
  grass: { color: 0x5f9e4a, roughness: 0.95 },
  stone: { color: 0x8c8c8c, roughness: 0.9 },
  wood: { color: 0x9b6b3f, roughness: 0.8 },
  metal: { color: 0x9aa3ad, roughness: 0.4, metalness: 0.6 },
  concrete: { color: 0xb5b5b5, roughness: 0.95 },
  plastic: { color: 0x3a7bd5, roughness: 0.5 },
  rubber: { color: 0x222222, roughness: 1.0 },
  gold: { color: 0xffc933, roughness: 0.3, metalness: 0.8 },
  red: { color: 0xd94b3d, roughness: 0.6 },
  blue: { color: 0x3d6fd9, roughness: 0.6 },
  green: { color: 0x3fb04a, roughness: 0.6 },
  yellow: { color: 0xf0d43a, roughness: 0.6 },
  white: { color: 0xf2f2f2, roughness: 0.7 },
  black: { color: 0x1a1a1a, roughness: 0.7 },
  lava: { color: 0xff5a1f, emissive: 0xff2200, emissiveIntensity: 0.8, roughness: 1 },
  water: { color: 0x3aa0d8, roughness: 0.2, metalness: 0.1, transparent: true, opacity: 0.8 },
  glass: { color: 0xbfe4ea, roughness: 0.1, transparent: true, opacity: 0.4 },
  cardboard: { color: 0xc49a6c, roughness: 0.9 },
};

export function makeMaterial(name) {
  const spec = MATERIALS[name] ?? (typeof name === 'string' && /^#[0-9a-f]{6}$/i.test(name) ? { color: parseInt(name.slice(1), 16), roughness: 0.7 } : MATERIALS.default);
  return new THREE.MeshStandardMaterial(spec);
}

// ---- primitive fallback for asset / generated ----
function fallbackForClass(cls, extent) {
  const [w, h, d] = extent;
  const c = (cls || '').toLowerCase();
  if (/coin|disc|ring|wheel/.test(c)) return { kind: 'primitive', shape: 'cylinder', radius: Math.max(w, d) / 2, height: h, axis: 'z' };
  if (/ball|sphere|orb|marble/.test(c)) return { kind: 'primitive', shape: 'sphere', radius: Math.max(w, h, d) / 2 };
  if (/tree|plant|bush/.test(c)) return { kind: 'composite', parts: [
    { shape: 'cylinder', radius: Math.min(w, d) * 0.12, height: h * 0.45, offset: [0, -h * 0.275, 0], material: 'wood' },
    { shape: 'sphere', radius: Math.max(w, d) * 0.5, offset: [0, h * 0.2, 0], material: 'green' } ] };
  if (/train|car|cart|vehicle|truck|bus|locomotive/.test(c)) return { kind: 'composite', parts: [
    { shape: 'box', extent: [w, h * 0.7, d], offset: [0, h * 0.05, 0] },
    { shape: 'box', extent: [w * 0.6, h * 0.3, d * 0.9], offset: [0, h * 0.5, 0] } ] };
  if (/door|gate|panel/.test(c)) return { kind: 'primitive', shape: 'box', extent: [w, h, Math.min(d, 0.12)] };
  return { kind: 'primitive', shape: 'box', extent };
}

function primitiveGeometry(g) {
  switch (g.shape) {
    case 'sphere': return new THREE.SphereGeometry(g.radius ?? 0.5, 24, 16);
    case 'cylinder': {
      const geo = new THREE.CylinderGeometry(g.radius_top ?? g.radius ?? 0.5, g.radius_bottom ?? g.radius ?? 0.5, g.height ?? 1, 24);
      if (g.axis === 'z') geo.rotateX(Math.PI / 2); else if (g.axis === 'x') geo.rotateZ(Math.PI / 2);
      return geo;
    }
    case 'cone': return new THREE.ConeGeometry(g.radius ?? 0.5, g.height ?? 1, 24);
    case 'plane': { const s = g.size ?? [10, 10]; const geo = new THREE.PlaneGeometry(s[0], s[1]); geo.rotateX(-Math.PI / 2); return geo; }
    case 'heightfield': // MVP: flat slab of the given scale
    case 'box':
    default: { const e = g.extent ?? g.scale ?? [1, 1, 1]; return new THREE.BoxGeometry(e[0], e[1], e[2]); }
  }
}

export function buildNode(spec, material) {
  // returns THREE.Group containing one or more meshes; also returns collider description list
  const group = new THREE.Group();
  const geom = spec.geom ?? { kind: 'primitive', shape: 'box', extent: [1, 1, 1] };
  const extent = geom.extent ?? geom.size ?? [1, 1, 1];
  let resolved = geom;
  if (geom.kind === 'asset' || geom.kind === 'generated') resolved = fallbackForClass(spec.class ?? geom.query, extent.length === 2 ? [extent[0], 0.1, extent[1]] : extent);
  const colliders = [];
  const mat = makeMaterial(spec.material ?? resolved.material ?? (geom.kind !== 'primitive' ? guessMaterial(spec.class) : 'default'));
  if (resolved.kind === 'composite') {
    for (const p of resolved.parts) {
      const m = new THREE.Mesh(primitiveGeometry(p), p.material ? makeMaterial(p.material) : mat);
      m.position.set(...(p.offset ?? [0, 0, 0])); m.castShadow = m.receiveShadow = true; group.add(m);
      colliders.push({ shape: p.shape, extent: p.extent, radius: p.radius, height: p.height, offset: p.offset ?? [0, 0, 0] });
    }
  } else {
    const m = new THREE.Mesh(primitiveGeometry(resolved), mat); m.castShadow = m.receiveShadow = true; group.add(m);
    colliders.push({ shape: resolved.shape, extent: resolved.extent ?? resolved.scale, size: resolved.size, radius: resolved.radius, height: resolved.height, axis: resolved.axis, offset: [0, 0, 0] });
  }
  return { group, colliders };
}

function guessMaterial(cls) {
  const c = (cls || '').toLowerCase();
  if (/coin|gold/.test(c)) return 'gold';
  if (/box|carton|crate|cardboard/.test(c)) return 'cardboard';
  if (/train|car|cart/.test(c)) return 'red';
  if (/door|table|shelf/.test(c)) return 'wood';
  if (/track|rail|belt|conveyor/.test(c)) return 'plastic';
  if (/lava/.test(c)) return 'lava';
  if (/water/.test(c)) return 'water';
  return 'default';
}

export function idToColor(i) {
  // i in [1, 2^24-1]; spread ids so neighbours differ visibly: multiply by a large odd constant mod 2^24
  const v = ((i * 2654435) % 0xffffff) || 1;
  return new THREE.Color(((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255);
}

export function buildScene(program) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(program.style?.background ?? '#87a7c7');
  scene.fog = null;
  const hemi = new THREE.HemisphereLight(0xffffff, 0x556655, 1.1); scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(6, 12, 8); sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024); sun.shadow.camera.left = sun.shadow.camera.bottom = -25; sun.shadow.camera.right = sun.shadow.camera.top = 25;
  scene.add(sun);

  const registry = []; // {id, kind:'static'|'object', group, colliders, pose(t), base, idIndex, idColor, class, dynamic, spec, instanceIndex}
  let idIndex = 1;
  const add = (spec, kind, base, motion, instanceIndex = 0) => {
    const { group, colliders } = buildNode(spec);
    group.name = instanceIndex ? `${spec.id}#${instanceIndex}` : spec.id;
    group.position.set(...base.pos); group.quaternion.set(...(base.quat ?? [0, 0, 0, 1])).normalize();
    scene.add(group);
    const entry = { id: spec.id, name: group.name, kind, group, colliders, base, spec, class: spec.class ?? kind,
      pose: makePose(motion, base), dynamic: !!motion && motion.type !== 'static', idIndex, idColor: idToColor(idIndex), instanceIndex,
      visible: true, events: spec.events ?? [] };
    idIndex++;
    group.traverse(o => { if (o.isMesh) o.userData.entry = entry; });
    registry.push(entry);
    return entry;
  };
  for (const s of program.static ?? []) add(s, 'static', { pos: s.pose?.pos ?? [0, 0, 0], quat: s.pose?.quat ?? [0, 0, 0, 1] }, null);
  for (const o of program.objects ?? []) {
    if (o.instances?.length) o.instances.forEach((inst, k) => add(o, 'object', { pos: inst.pos, quat: inst.quat ?? o.pose?.quat ?? [0, 0, 0, 1] }, o.motion, k + 1));
    else add(o, 'object', { pos: o.pose?.pos ?? [0, 0, 0], quat: o.pose?.quat ?? [0, 0, 0, 1] }, o.motion);
  }
  return { scene, registry, lights: { hemi, sun } };
}
