// main.js — 固定运行时入口：解释 program.json，固定步长，window.__game 控制面
import * as THREE from 'three';
import { buildScene } from './scene.js';
import { loadAssets } from './assets.js';
import { Manipulator } from './manipulate.js';
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
  // 资产先全部读完再建场景：buildScene 保持同步，物体 id 的注册顺序也就不受加载快慢影响。
  const { scenes: assetScenes, failures: assetFailures } = await loadAssets('.');
  game.assets = { loaded: Object.keys(assetScenes).length, failures: assetFailures };
  const { scene, registry } = buildScene(program, kernelCfg, assetScenes);
  const physics = new Physics(DT);
  for (const e of registry) physics.addEntry(e);
  const replay = makeReplayCamera(program.camera, W / H);
  const template = new PlatformerTemplate({ program, scene, registry, physics, kernelCfg });
  template.camera.aspect = W / H; template.camera.updateProjectionMatrix();
  const far = program.camera?.intrinsics?.far ?? 100;

  // 直接操纵：鼠标拖动场景里的物体。构造放在控制块之前，这样 tests/test_physics_runtime.mjs
  // 提取出来的那段代码只依赖 manipulator 这个对象本身，不依赖 Manipulator 这个类。
  const manipulator = new Manipulator({ camera: template.camera, registry, physics, scene, dt: DT });
  if (!HEADLESS) manipulator.attach(renderer.domElement);

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
    dt: DT, registry, scene, renderer, physics, template, manipulator,
    grab: (x, y) => manipulator.grabAt(x, y)?.id ?? null,
    dragTo: (x, y, z) => manipulator.dragToPoint(x, y, z),
    dragNdc: (x, y) => manipulator.dragTo(x, y),
    release: () => manipulator.release()?.id ?? null,
    grabLog: () => manipulator.log,
    // 切模式时要把操纵器的相机也换过去：画面用 activeCamera() 渲染，而鼠标拾取是
    // 从操纵器自己那台相机打射线的。两者不一致的话，在回放视角下点哪儿都打偏——
    // 看到的是视频的视角，射线却从玩家那台相机出发。
    mode(m) { game.modeName = m === 'replay' ? 'replay' : 'play'; if (m === 'play') { template.playerMesh.visible = true; template.goalMesh.visible = true; } else { template.playerMesh.visible = false; template.goalMesh.visible = false; } updateCamera(); manipulator.camera = activeCamera(); return game.modeName; },
    step(dt = DT) {
      if (!Number.isFinite(dt) || dt <= 0) throw new Error('step needs a positive finite dt');
      if (hasSimulated && Math.abs(dt - DT) > 1e-12) throw new Error('dynamic step requires the fixed dt (1/60 s); use seek for sampling');
      game.frame++;
      game.t = hasSimulated ? game.frame * DT : game.t + dt;
      updateMotions(game.t);
      if (game.modeName === 'play') template.update(game.input, game.t);
      manipulator.update();
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
    // 把场景放下来自己稳一稳：只动没有脚本运动的物体——它们的位姿完全来自感知，
    // 而感知给的是按可见表面拟合的包围盒中心，常常半悬空或者和别的物体穿插。
    // 有脚本运动的物体位姿由 pose(t) 决定，不参与。
    settle(seconds = 1.5, separateSeconds = 0) {
      const movable = registry.filter(e => e.kind === 'object'
        && (!e.spec.motion || e.spec.motion.type === 'static'));
      const before = new Map(movable.map(e => [e.id, e.group.position.clone()]));
      const saved = movable.map(e => ({ e, dynamic: e.dynamic, simulated: e.simulated }));
      for (const e of movable) { physics.makeDynamic(e); e.dynamic = true; e.simulated = true; }
      // 第一段：只解穿插，不开重力。
      //
      // 重力解不开互相咬住的物体：它只有竖直方向的分量，两个物体横向插在一起的时候，
      // 重力没有把它们分开的方向可用，跑多久都还是插着。接触求解本身会沿着接触法线
      // 把重叠顶开，所以先关掉重力单跑一段让它只做这件事，再开重力落到支撑面上。
      // 两件事混在一个仿真里会互相抵消——顶开一点、重力又压回去。
      const sepSteps = Math.max(0, Math.round(separateSeconds / DT));
      if (sepSteps) {
        for (const e of movable) {
          e.body.setGravityScale(0, true);
          e.body.setLinearDamping(4); e.body.setAngularDamping(4);
        }
        for (let i = 0; i < sepSteps; i++) {
          physics.step();
          for (const e of registry) physics.syncSimulated(e);
        }
        // makeDynamic 把重力、阻尼调回正常并清零速度，正好是这里要的
        for (const e of movable) physics.makeDynamic(e);
      }
      const steps = Math.max(1, Math.round(seconds / DT));
      for (let i = 0; i < steps; i++) {
        physics.step();
        for (const e of registry) physics.syncSimulated(e);
      }
      const result = movable.map(e => ({
        id: e.id,
        pos: e.group.position.toArray(),
        quat: e.group.quaternion.toArray(),
        moved: e.group.position.distanceTo(before.get(e.id)),
      }));
      for (const s of saved) { s.e.dynamic = s.dynamic; s.e.simulated = s.simulated; }
      return { seconds, separate_seconds: separateSeconds, settled: result };
    },
    // 玩法绑定是怎么来的。判定结果要能说出自己走的哪条路：
    //   slot_source  每个槽位是程序填的（走过 gwm/binding/）还是模板猜的
    //   game_scale   Python 那边做过的尺度归一化系数；没走绑定就没有归一化，
    //                而试玩角色是固定尺寸的胶囊，比例会错得很离谱
    bindingInfo() {
      return {
        template: program.binding?.template ?? null,
        slot_source: template.slotSource ?? null,
        inferred_slots: Object.entries(template.slotSource ?? {})
          .filter(([, v]) => v === 'inferred').map(([k]) => k),
        game_scale: program.meta?.game_scale ?? null,
      };
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
