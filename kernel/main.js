// main.js — 固定运行时入口：解释 program.json，固定步长，window.__game 控制面
import * as THREE from 'three';
import { buildScene } from './scene.js';
import { renderPass } from './passes.js';
import { makeReplayCamera } from './replay_camera.js';
import { initRapier, Physics } from './physics.js';
import { PlatformerTemplate } from './templates/platformer_3p.js';

const params = new URLSearchParams(location.search);
const HEADLESS = params.get('headless') === '1';
const DT = 1 / 60;

const game = {
  readyState: 'loading', error: null, t: 0, frame: 0, modeName: HEADLESS ? 'replay' : 'play',
  input: { keys: new Set(), move: null, jump: false },
};
window.__game = game;
const hud = document.getElementById('hud');
const log = (...a) => { console.log('[gwm]', ...a); };

async function boot() {
  const program = await (await fetch('./program.json', { cache: 'no-store' })).json();
  game.program = program;
  const W = Number(params.get('w') ?? innerWidth), H = Number(params.get('h') ?? innerHeight);
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setSize(W, H, false); renderer.setPixelRatio(1);
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.domElement.style.width = '100vw'; renderer.domElement.style.height = '100vh';
  document.body.appendChild(renderer.domElement);
  await initRapier();
  const { scene, registry } = buildScene(program);
  const physics = new Physics(DT);
  for (const e of registry) physics.addEntry(e);
  const replay = makeReplayCamera(program.camera, W / H);
  const template = new PlatformerTemplate({ program, scene, registry, physics });
  template.camera.aspect = W / H; template.camera.updateProjectionMatrix();
  const far = program.camera?.intrinsics?.far ?? 100;

  function updateMotions(t) {
    for (const e of registry) {
      if (!e.dynamic) continue;
      const tt = e.spec.motion?.trigger ? (e.triggerTime == null ? 0 : t - e.triggerTime) : t;
      const p = e.pose(tt);
      e.delta = p.pos.clone().sub(e.group.position);
      e.group.position.copy(p.pos); e.group.quaternion.copy(p.quat);
      physics.syncKinematic(e);
    }
  }
  function activeCamera() { return game.modeName === 'replay' ? replay.camera : template.camera; }
  function updateCamera() { if (game.modeName === 'replay') replay.update(game.t); }

  Object.assign(game, {
    dt: DT, registry, scene, renderer, physics, template,
    mode(m) { game.modeName = m === 'replay' ? 'replay' : 'play'; if (m === 'play') { template.playerMesh.visible = true; template.goalMesh.visible = true; } else { template.playerMesh.visible = false; template.goalMesh.visible = false; } updateCamera(); return game.modeName; },
    step(dt = DT) {
      game.t += dt; game.frame++;
      updateMotions(game.t);
      if (game.modeName === 'play') template.update(game.input, game.t);
      physics.step();
      updateCamera();
      return game.t;
    },
    seek(t) { // replay-only: motions are pure functions of t
      game.t = t; game.frame = Math.round(t / DT); updateMotions(t); updateCamera(); return game.t;
    },
    render(pass = 'rgb') { return renderPass(renderer, scene, activeCamera(), registry, pass, { far }); },
    draw() { renderer.render(scene, activeCamera()); },
    setSize(w, h) { renderer.setSize(w, h, false); replay.camera.aspect = w / h; replay.camera.updateProjectionMatrix(); template.camera.aspect = w / h; template.camera.updateProjectionMatrix(); },
    setInput(i) { if (i.move !== undefined) game.input.move = i.move; if (i.jump !== undefined) game.input.jump = !!i.jump; if (i.keys) game.input.keys = new Set(i.keys.map(k => k.toLowerCase())); },
    reset(seed = 0) { game.t = 0; game.frame = 0; for (const e of registry) { e.triggerTime = null; if (!e.visible) { e.visible = true; e.group.visible = true; physics.addEntry(e); } } template.collected = 0; template.deaths = 0; template.won = false; template.wonAt = null; physics.teleportPlayer(template.spawn); updateMotions(0); updateCamera(); },
    state() {
      return { t: game.t, frame: game.frame, mode: game.modeName, player: template.state(),
        objects: registry.map(e => ({ id: e.id, name: e.name, kind: e.kind, class: e.class, pos: e.group.position.toArray(), quat: e.group.quaternion.toArray(), visible: e.visible, dynamic: e.dynamic })) };
    },
    cameraInfo() { const c = activeCamera(); return { pos: c.position.toArray(), quat: c.quaternion.toArray(), fov: c.fov, aspect: c.aspect, near: c.near, far: c.far }; },
  });
  updateMotions(0); game.mode(game.modeName);
  game.readyState = 'ready'; log('ready', { objects: registry.length, headless: HEADLESS });

  if (!HEADLESS) {
    addEventListener('keydown', e => { game.input.keys.add(e.key.toLowerCase()); if (e.key === ' ') e.preventDefault(); if (e.key.toLowerCase() === 'r') game.mode(game.modeName === 'play' ? 'replay' : 'play'); });
    addEventListener('keyup', e => game.input.keys.delete(e.key.toLowerCase()));
    addEventListener('resize', () => game.setSize(innerWidth, innerHeight));
    let last = performance.now(), acc = 0;
    const loop = (now) => {
      acc += Math.min((now - last) / 1000, 0.1); last = now;
      while (acc >= DT) { game.step(DT); acc -= DT; }
      game.draw();
      const s = template.state();
      hud.textContent = `t=${game.t.toFixed(1)}s  mode=${game.modeName} (R)  collected ${s.collected}/${s.collectibles_total}  deaths ${s.deaths}  ${s.won ? 'GOAL REACHED' : 'WASD/arrows move, space jump'}`;
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }
}
boot().catch(err => { game.readyState = 'error'; game.error = String(err?.stack ?? err); console.error('[gwm] boot failed', err); if (hud) hud.textContent = 'boot failed: ' + err; });
