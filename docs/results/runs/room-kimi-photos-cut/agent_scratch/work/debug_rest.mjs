// 调试：单独把某个物体设为 dynamic，打印它 150 帧内的位置和接触
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import * as THREE from 'three';
import { initRapier } from '/Users/jwj/Code/GameWorldModel/kernel/physics.js';
import { buildBenchScene, makeContext } from '/Users/jwj/Code/GameWorldModel/kernel/bench_scene.js';
import { makeWorldFrom } from '/Users/jwj/Code/GameWorldModel/harness/interaction_probe.mjs';
import fs from 'node:fs';

const target = process.argv[2] || 'bowl';
await initRapier();
const module = await import(pathToFileURL(path.resolve('scene.js')).href);
const materials = JSON.parse(fs.readFileSync('materials.json', 'utf8'));
const world = makeWorldFrom(buildBenchScene(module, makeContext({ materials, mode: 'flat' })));
const entry = world.registry.find(e => e.id === target);
console.log('start pos', entry.group.position.toArray().map(v => +v.toFixed(3)));
console.log('colliders', JSON.stringify(entry.colliders));
world.physics.makeDynamic(entry);
entry.dynamic = true; entry.simulated = true;
const w = world.physics.world;
function contacts(tag) {
  for (const h of entry.colliderHandles ?? []) {
    const c = w.getCollider(h);
    w.contactPairsWith(c, (other) => {
      const hit = world.physics.colliderToEntry.get(other.handle);
      w.contactPair(c, other, (m) => {
        let deepest = 0;
        for (let i = 0; i < m.numContacts(); i++) deepest = Math.max(deepest, -m.contactDist(i));
        if (deepest > 0.0005) {
          const t = other.translation(), tc = c.translation();
          console.log(tag, 'contact with', hit?.id, 'deepest', deepest.toFixed(4),
            'mine', [tc.x, tc.y, tc.z].map(v => +v.toFixed(2)), 'theirs', [t.x, t.y, t.z].map(v => +v.toFixed(2)));
        }
      });
    });
  }
}
for (let f = 0; f < 150; f++) {
  world.step(1);
  if (f < 6) { console.log('f' + f, entry.group.position.toArray().map(v => +v.toFixed(4))); contacts('f' + f); }
  if (f % 30 === 0 || f === 149) {
    const p = entry.group.position;
    console.log('f' + f, p.toArray().map(v => +v.toFixed(4)));
  }
}
// contacts at end
for (const h of entry.colliderHandles ?? []) {
  const c = w.getCollider(h);
  w.contactPairsWith(c, (other) => {
    const hit = world.physics.colliderToEntry.get(other.handle);
    w.contactPair(c, other, (m) => {
      let deepest = 0;
      for (let i = 0; i < m.numContacts(); i++) deepest = Math.max(deepest, -m.contactDist(i));
      console.log('contact with', hit?.id, 'deepest', deepest.toFixed(4), 'n', m.numContacts());
    });
  });
}
