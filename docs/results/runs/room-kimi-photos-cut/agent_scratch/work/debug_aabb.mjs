import path from 'node:path';
import { pathToFileURL } from 'node:url';
import * as THREE from 'three';
import { initRapier } from '/Users/jwj/Code/GameWorldModel/kernel/physics.js';
import { buildBenchScene, makeContext } from '/Users/jwj/Code/GameWorldModel/kernel/bench_scene.js';
import fs from 'node:fs';

await initRapier();
const module = await import(pathToFileURL(path.resolve('scene.js')).href);
const materials = JSON.parse(fs.readFileSync('materials.json', 'utf8'));
const { registry } = buildBenchScene(module, makeContext({ materials, mode: 'flat' }));
for (const id of ['coffee_table', 'sofa', 'wine_glass', 'bowl', 'carafe']) {
  const e = registry.find(r => r.id === id);
  e.group.updateMatrixWorld(true);
  console.log('==', id, 'group pos', e.group.position.toArray().map(v => +v.toFixed(3)));
  e.group.traverse(n => {
    if (!n.isMesh) return;
    const b = new THREE.Box3().setFromObject(n);
    console.log('  mesh', n.geometry.type, 'min', b.min.toArray().map(v => +v.toFixed(3)), 'max', b.max.toArray().map(v => +v.toFixed(3)));
  });
}
