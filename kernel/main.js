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
  const kernelCfg = await fetch('./kernel_config.json', { cache: 'no-store' }).then(r => r.ok ? r.json() : null).catch(() => null);
  const W = Number(params.get('w') ?? innerWidth), H = Number(params.get('h') ?? innerHeight);
  // 无头模式下关抗锯齿：depth 和 id 两个 pass 把数值编码进像素，多重采样会在物体边缘把相邻编码混成
  // 不存在的颜色，Python 侧精确匹配会整片丢掉这些像素，掩码 IoU 被系统性压低
  const renderer = new THREE.WebGLRenderer({ antialias: !HEADLESS, preserveDrawingBuffer: true });
  renderer.setSize(W, H, false); renderer.setPixelRatio(1);
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.domElement.style.width = '100vw'; renderer.domElement.style.height = '100vh';
  document.body.appendChild(renderer.domElement);
  await initRapier();
  const { scene, registry } = buildScene(program, kernelCfg);
  const physics = new Physics(DT);
  for (const e of registry) physics.addEntry(e);
  const replay = makeReplayCamera(program.camera, W / H);
  const template = new PlatformerTemplate({ program, scene, registry, physics, kernelCfg });
  template.camera.aspect = W / H; template.camera.updateProjectionMatrix();
  const far = program.camera?.intrinsics?.far ?? 100;

  const hasSimulated = registry.some(e => e.simulated);

  function updateMotions(t) {
    for (const e of registry) {
      if (!e.dynamic || e.simulated) continue;   // 物理驱动的由 Rapier 决定位姿，不在这里算
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
      if (!Number.isFinite(dt) || dt <= 0) throw new Error('step needs a positive finite dt');
      if (hasSimulated && Math.abs(dt - DT) > 1e-12) throw new Error('dynamic step requires the fixed dt (1/60 s); use seek for sampling');
      game.frame++;
      game.t = hasSimulated ? game.frame * DT : game.t + dt;
      updateMotions(game.t);
      if (game.modeName === 'play') template.update(game.input, game.t);
      physics.step();
      for (const e of registry) physics.syncSimulated(e);   // 仿真之后把位姿读回来
      updateCamera();
      return game.t;
    },
    // 脚本运动是时间的纯函数，可以直接跳到 t。
    // 物理驱动的物体是仿真出来的，跳不过去，只能一步步推。固定步长加固定初值保证可复现。
    // 往前走就从当前状态接着推，只有往回跳才重置，这样顺序录制是线性而不是平方的。
    seek(t) {
      if (!Number.isFinite(t) || t < 0) throw new Error('seek needs a non-negative finite time');
      if (!hasSimulated) { game.t = t; game.frame = Math.round(t / DT); updateMotions(t); updateCamera(); return game.t; }
      const targetFrame = Math.round(t / DT);
      if (!Number.isSafeInteger(targetFrame)) throw new Error('seek time is outside the supported frame range');
      if (targetFrame < game.frame) game.reset();
      const steps = targetFrame - game.frame;
      for (let i = 0; i < steps; i++) game.step(DT);
      updateCamera(); return game.t;
    },
    render(pass = 'rgb') { return renderPass(renderer, scene, activeCamera(), registry, pass, { far }); },
    draw() { renderer.render(scene, activeCamera()); },
    setSize(w, h) { renderer.setSize(w, h, false); replay.camera.aspect = w / h; replay.camera.updateProjectionMatrix(); template.camera.aspect = w / h; template.camera.updateProjectionMatrix(); },
    setInput(i) { if (i.move !== undefined) game.input.move = i.move; if (i.jump !== undefined) game.input.jump = !!i.jump; if (i.keys) game.input.keys = new Set(i.keys.map(k => k.toLowerCase())); },
    reset(seed = 0) { game.t = 0; game.frame = 0; for (const e of registry) { e.triggerTime = null; if (!e.visible) { e.visible = true; e.group.visible = true; physics.addEntry(e); } physics.respawnSimulated(e); } template.collected = 0; template.deaths = 0; template.won = false; template.wonAt = null; physics.teleportPlayer(template.spawn); updateMotions(0); updateCamera(); },
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
