// node harness/partnet_check.mjs --import DIR --out NEW_DIR
import fs from 'node:fs';
import path from 'node:path';
import { openGame, parseArgs, savePng } from './common.mjs';

const args = parseArgs(process.argv.slice(2));
const root = args.import;
const reference = JSON.parse(fs.readFileSync(path.join(root,'source_mapping.json')));
fs.mkdirSync(args.out, { recursive:false });
const g = await openGame(path.join(root,'game'), { width:640,height:480,sceneOnly:true,
  logFile:path.join(args.out,'browser.log') });
try {
  const assets = await g.page.evaluate(() => window.__game.assets);
  if (assets.loaded!==2 || assets.failures.length) throw new Error('real meshes failed to load');
  const frames = [];
  for (const ref of reference.states) {
    const actualTime = await g.page.evaluate(t => window.__game.seek(t),ref.t);
    const state = await g.page.evaluate(() => {
      const g = window.__game;
      g.scene.updateMatrixWorld(true);
      return g.registry.map(e => {
        const matrix = e.group.matrixWorld.toArray();
        let vertices=0,triangles=0;
        e.group.traverse(n => { if(n.isMesh) { vertices+=n.geometry.attributes.position.count;
          triangles+=(n.geometry.index?.count??n.geometry.attributes.position.count)/3; }});
        return {id:e.id,pos:e.group.position.toArray(),matrix,vertices,triangles};
      });
    });
    const lid=state.find(x=>x.id==='lid');
    const positionError=Math.max(...lid.pos.map((v,i)=>Math.abs(v-ref.lid_pos[i])));
    const rotationError=Math.max(...ref.lid_rotation.flatMap((row,r)=>row.map((v,c)=>Math.abs(v-lid.matrix[c*4+r]))));
    if(Math.abs(actualTime-ref.t)>1e-9 || positionError>1e-5 || rotationError>1e-5) throw new Error('URDF to runtime state mismatch');
    for(const entry of state) {
      const expected=reference.mesh_counts[entry.id];
      if(!expected || entry.vertices!==expected.vertices || entry.triangles!==expected.triangles) throw new Error('source topology changed');
    }
    const rgb=`rgb_${ref.t}.png`;
    savePng(await g.page.evaluate(()=>window.__game.render('rgb')),path.join(args.out,rgb));
    frames.push({t:actualTime,positionError,rotationError,state,rgb});
  }
  // Keep the loaded source scene alive: navigating it away can cancel late
  // browser resource requests. Inspect the UI in a separate, fully logged page.
  const preview = await g.browser.newPage();
  const record = (kind,text) => {
    const line=`[${kind}] ${text}`;
    g.logs.push(line);
    fs.appendFileSync(path.join(args.out,'browser.log'),line+'\n');
  };
  preview.on('console',m=>record(m.type(),m.text()));
  preview.on('pageerror',e=>record('pageerror',e.message));
  preview.on('requestfailed',r=>record('requestfailed',`${r.url()} ${r.failure()?.errorText}`));
  await preview.goto(g.url+'/joint_preview.html');
  await preview.waitForFunction(() => window.previewReady===true);
  await preview.locator('#state').evaluate(el=>{el.value='0.5';el.dispatchEvent(new Event('input',{bubbles:true}));});
  const sliderTime=await preview.evaluate(()=>document.querySelector('iframe').contentWindow.__game.t);
  if(Math.abs(sliderTime-0.5)>1e-9) throw new Error('state slider failed');
  const bad=g.logs.filter(line=>/^\[(error|pageerror|requestfailed)\]/.test(line));
  if(bad.length) throw new Error('browser errors: '+bad.join('\n'));
  fs.writeFileSync(path.join(args.out,'result.json'),JSON.stringify({ok:true,assets,frames,sliderTime,
    browser_events:g.logs,limitations:['no source textures','no collision/dragging validation','existing flat DSL subset']},null,2));
} finally { await g.close(); }
