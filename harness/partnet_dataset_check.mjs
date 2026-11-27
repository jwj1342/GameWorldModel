// Verify compiled style-pair samples against the imported source kinematics.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { openGame, parseArgs, savePng } from './common.mjs';

const a = parseArgs(process.argv.slice(2));
if (!a.dataset || !a.reference || !a.out) throw new Error('requires --dataset --reference --out');
const raw = fs.readFileSync(a.reference);
const reference = JSON.parse(raw);
const manifest = JSON.parse(fs.readFileSync(path.join(a.dataset, 'manifest.json')));
if (crypto.createHash('sha256').update(raw).digest('hex') !== manifest.source_mapping_sha256) {
  throw new Error('source mapping identity mismatch');
}
fs.mkdirSync(a.out, { recursive:false });
const samples = [];
for (const sample of manifest.samples) {
  const dir = path.join(a.out, sample.clip);
  fs.mkdirSync(dir);
  const root = path.join(a.dataset, sample.clip);
  const g = await openGame(path.join(root, 'game'), {width:320, height:240, sceneOnly:true,
    logFile:path.join(dir, 'browser.log')});
  try {
    const assets = await g.page.evaluate(() => window.__game.assets);
    if (assets.loaded !== 2 || assets.failures.length) throw new Error('asset loading failed');
    const frames = [];
    for (const [i, ref] of reference.states.entries()) {
      const t = await g.page.evaluate(t => window.__game.seek(t), ref.t);
      const state = await g.page.evaluate(() => {
        const game = window.__game;
        game.scene.updateMatrixWorld(true);
        return game.registry.map(e => {
          let vertices=0, triangles=0;
          e.group.traverse(n => {if (n.isMesh) {vertices+=n.geometry.attributes.position.count;
            triangles+=(n.geometry.index?.count??n.geometry.attributes.position.count)/3;}});
          return {id:e.id, idx:e.idIndex, pos:e.group.position.toArray(),
            matrix:e.group.matrixWorld.toArray(), vertices, triangles};
        });
      });
      const lid = state.find(s => s.id === 'lid');
      const positionError = Math.max(...lid.pos.map((v,j)=>Math.abs(v-ref.lid_pos[j])));
      const rotationError = Math.max(...ref.lid_rotation.flatMap((row,r)=>row.map((v,c)=>Math.abs(v-lid.matrix[c*4+r]))));
      if (Math.abs(t-ref.t)>1e-9 || positionError>1e-5 || rotationError>1e-5) throw new Error('source state mismatch');
      if (state.length !== 2) throw new Error('unexpected instances');
      for (const node of state) {
        const counts = reference.mesh_counts[node.id];
        if (!counts || node.vertices!==counts.vertices || node.triangles!==counts.triangles) throw new Error('topology mismatch');
      }
      const rgb=`rgb_${i}.png`, id=`id_${i}.png`;
      savePng(await g.page.evaluate(()=>window.__game.render('rgb')),path.join(dir,rgb));
      savePng(await g.page.evaluate(()=>window.__game.render('id')),path.join(dir,id));
      frames.push({t,state,positionError,rotationError,rgb,id});
    }
    const errors=g.logs.filter(s=>/^\[(error|pageerror|requestfailed)\]/.test(s));
    if (errors.length) throw new Error('browser errors: '+errors.join('\n'));
    samples.push({clip:sample.clip, assets, frames, browser_errors:errors});
  } finally {await g.close();}
}
fs.writeFileSync(path.join(a.out,'runtime.json'),JSON.stringify({ok:true,samples,
  limits:['source colors replaced; no source textures','no mouse/physics validation','three sampled states, not full clip']},null,2));
