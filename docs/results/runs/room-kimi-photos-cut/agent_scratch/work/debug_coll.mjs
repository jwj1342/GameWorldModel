import path from 'node:path';
import { pathToFileURL } from 'node:url';
import * as THREE from 'three';
import { initRapier } from '/Users/jwj/Code/GameWorldModel/kernel/physics.js';
import { buildBenchScene, makeContext } from '/Users/jwj/Code/GameWorldModel/kernel/bench_scene.js';
import { makeWorldFrom } from '/Users/jwj/Code/GameWorldModel/harness/interaction_probe.mjs';
import fs from 'node:fs';

await initRapier();
const module = await import(pathToFileURL(path.resolve('scene.js')).href);
const materials = JSON.parse(fs.readFileSync('materials.json', 'utf8'));
const world = makeWorldFrom(buildBenchScene(module, makeContext({ materials, mode: 'flat' })));
const w = world.physics.world;
for (const id of ['coffee_table', 'sofa']) {
  const e = world.registry.find(r => r.id === id);
  console.log('==', id, 'body t', e.body.translation(), 'colliders:');
  for (const h of e.colliderHandles) {
    const c = w.getCollider(h);
    const t = c.translation();
    const he = c.shape.halfExtents;
    console.log('  cuboid he', [he.x, he.y, he.z].map(v => +v.toFixed(3)), 'at', [t.x, t.y, t.z].map(v => +v.toFixed(3)));
  }
}
