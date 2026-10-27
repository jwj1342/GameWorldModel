// CPU boot/control tests: real scene, camera, Rapier and gameplay template;
// fake canvas/renderer only. No browser, WebGL or rendered images.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';
import * as THREE from 'three';
import {buildScene} from '../kernel/scene.js';
import {Physics, initRapier} from '../kernel/physics.js';
import {makeReplayCamera} from '../kernel/replay_camera.js';
import {PlatformerTemplate} from '../kernel/templates/platformer_3p.js';
import {Manipulator} from '../kernel/manipulate.js';
import {renderPass} from '../kernel/passes.js';
await initRapier();
const source = fs.readFileSync(new URL('../kernel/main.js', import.meta.url), 'utf8');
const program = () => ({
  binding: {template: 'platformer_3p', slots: {}},
  camera: {interp:'linear', keyframes:[{t:0,pos:[0,3,8],quat:[0,0,0,1]},{t:1,pos:[1,3,8],quat:[0,0,0,1]}]},
  static: [{id:'base', class:'floor', pose:{pos:[0,-0.5,0]}, geom:{kind:'primitive',shape:'box',extent:[10,1,10]}}],
  objects: [
    {id:'fixed',class:'generic',pose:{pos:[0,2,0],quat:[0,0,0,1]},geom:{kind:'primitive',shape:'box',extent:[1,1,1]}},
    {id:'moving',class:'generic',pose:{pos:[2,3,0],quat:[0,0,0,1]},geom:{kind:'primitive',shape:'sphere',radius:0.2},motion:{type:'prismatic',axis:[1,0,0],rate:1}},
  ],
});
async function boot(p, query='headless=1', {forbidTemplate=false}={}) {
  const stats={templates:0,renders:0,raf:0,events:0};
  class Renderer {
    constructor() {this.domElement={style:{},addEventListener(){},toDataURL:()=>'data:image/png;base64,AA=='};this.shadowMap={};}
    setSize(w,h) {this.size=[w,h];}
    setPixelRatio() {}
    render() {stats.renders++;}
  }
  class Template extends PlatformerTemplate {
    constructor(...args) {stats.templates++;if(forbidTemplate) throw new Error('template must not be constructed');super(...args);}
  }
  const window={};
  const context={THREE:{...THREE,WebGLRenderer:Renderer}, buildScene, loadAssets:async()=>({scenes:{},failures:[]}), Manipulator,
    renderPass,makeReplayCamera,initRapier,Physics,PlatformerTemplate:Template,
    URLSearchParams,location:{search:'?'+query},window,document:{getElementById:()=>({}),body:{appendChild(){}}},
    innerWidth:320,innerHeight:180,fetch:async url=>({ok:true,json:async()=>url==='./program.json'?p:{}}),
    console:{log(){},error(){}},addEventListener(){stats.events++;},performance:{now:()=>0},requestAnimationFrame(){stats.raf++;},
  };
  const code=source.replace(/^import .*;\r?\n/gm,'').replace('boot().catch(', 'globalThis.bootPromise = boot().catch(');
  vm.createContext(context);vm.runInContext(code,context);await context.bootPromise;
  const game=window.__game;
  assert.equal(game.readyState,'ready',game.error);
  return {game,stats,close:()=>game.physics.world.free()};
}
const plain=value=>JSON.parse(JSON.stringify(value));
const poses=g=>plain(g.state().objects.map(({id,name,pos,quat,kind})=>({id,name,pos,quat,kind})));

test('default headless boot keeps gameplay template, inferred slots and original replay controls',async()=>{
  const p=program(),before=JSON.stringify(p),h=await boot(p);
  try {
    assert.equal(h.stats.templates,1);assert.equal(h.game.sceneOnly,false);
    assert(h.game.template instanceof PlatformerTemplate);assert(h.game.physics.player);
    assert.equal(h.game.scene.getObjectByName('player'),h.game.template.playerMesh);
    assert.equal(h.game.scene.getObjectByName('goal'),h.game.template.goalMesh);
    assert.equal(h.game.bindingInfo().inferred_slots.length,4);
    assert.equal(h.game.modeName,'replay');assert.equal(h.game.template.playerMesh.visible,false);
    assert.equal(h.game.mode('play'),'play');assert.equal(h.game.template.playerMesh.visible,true);
    h.game.step();h.game.reset();assert(h.game.state().player);
    assert.equal(JSON.stringify(p),before);
  } finally {h.close();}
});
test('default interactive boot retains play mode and event/animation setup',async()=>{
  const h=await boot(program(),'');
  try {assert.equal(h.stats.templates,1);assert.equal(h.game.modeName,'play');assert.equal(h.stats.raf,1);assert(h.stats.events>=3);}
  finally {h.close();}
});
test('scene-only boot skips template/inferSlots, player/goal and preserves registry IDs and initial poses',async()=>{
  const p=program(),before=JSON.stringify(p),legacy=await boot(p),h=await boot(p,'headless=1&scene_only=1',{forbidTemplate:true});
  try {
    assert.equal(h.stats.templates,0);assert.equal(h.game.sceneOnly,true);assert.equal(h.game.template,null);
    assert.equal(h.game.physics.player,null);assert.equal(h.game.scene.getObjectByName('player'),undefined);assert.equal(h.game.scene.getObjectByName('goal'),undefined);
    assert.deepEqual(plain(h.game.bindingInfo()),{template:null,slot_source:null,inferred_slots:[],game_scale:null});
    assert.deepEqual(poses(h.game),poses(legacy.game));
    assert.deepEqual(h.game.registry.map(e=>e.idIndex),legacy.game.registry.map(e=>e.idIndex));
    assert.equal(h.game.state().player,null);assert.equal(JSON.stringify(p),before);assert.deepEqual(p.binding.slots,{});
  } finally {h.close();legacy.close();}
});
test('scene-only seek, state, cameraInfo, setSize, draw and RGB/depth/ID control paths remain available',async()=>{
  const h=await boot(program(),'scene_only=1',{forbidTemplate:true});
  try {
    assert.equal(h.stats.events,0);assert.equal(h.stats.raf,0);
    for(const name of ['seek','step','reset','mode','render','draw','state','cameraInfo','setSize']) assert.equal(typeof h.game[name],'function');
    const start=poses(h.game);assert.equal(h.game.mode('replay'),'replay');
    assert.throws(()=>h.game.mode('play'),/scene-only/);assert.equal(h.game.modeName,'replay');
    // Rejected before moving any object/body; no settling runs.
    assert.throws(()=>h.game.settle(),/unavailable/);assert.deepEqual(poses(h.game),start);
    assert.equal(h.game.seek(0.123456789),0.123456789);assert.notDeepEqual(poses(h.game),start);
    assert(Math.abs(h.game.cameraInfo().pos[0]-0.123456789)<1e-12);
    h.game.setSize(960,540);assert.equal(h.game.cameraInfo().aspect,960/540);
    for(const pass of ['rgb','depth','id']) assert.equal(h.game.render(pass),'AA==');
    h.game.draw();assert.equal(h.stats.renders,4); // stub calls only, no pixels/images
    h.game.reset();assert.deepEqual(poses(h.game),start);
    h.game.step();assert.equal(h.game.state().t,1/60);
  } finally {h.close();}
});
test('scene-only dynamic physics motion still seeks, resets and replays deterministically',async()=>{
  const p=program();p.objects[1].motion={type:'dynamic'};
  const h=await boot(p,'headless=1&scene_only=1',{forbidTemplate:true});
  try {
    const before=h.game.registry[2].group.position.y;
    h.game.seek(0.25);const pose=poses(h.game);assert(h.game.registry[2].group.position.y<before);
    h.game.reset();h.game.seek(0.25);assert.deepEqual(poses(h.game),pose);assert.equal(h.game.physics.player,null);
  } finally {h.close();}
});
test('dynamic seek records the nearest physics frame and preserves positive half-step ties',async()=>{
  const p=program();p.objects[1].motion={type:'dynamic'};
  const h=await boot(p,'headless=1&scene_only=1',{forbidTemplate:true});
  try {
    for (const t of [0.04,0.025,1/24,1/25,1001/30000,0.007+1/30]) {
      h.game.reset();
      const actual=h.game.seek(t),expected=Math.round(t/(1/60))*(1/60);
      assert.equal(actual,expected);
      assert.equal(h.game.state().t,expected);
      assert(Math.abs(actual-t)<=1/120+1e-12);
    }
  } finally {h.close();}
});
for (const withBinding of [false, true]) {
  test(`scene-only preserves synthetic authored scene with binding=${withBinding}`,async()=>{
    const p=program();
    if (withBinding) {
      p.binding.slots={player_spawn:[1,8,1],goal_volume:{pos:[3,8,3],extent:[1,2,1]},collectibles:['fixed'],hazards:[]};
    } else {
      delete p.binding;
    }
    const before=JSON.stringify(p),baseline=buildScene(p),h=await boot(p,'headless=1&scene_only=1',{forbidTemplate:true});
    try {
      const expected=baseline.registry.map(e=>({id:e.id,name:e.name,pos:e.group.position.toArray(),quat:e.group.quaternion.toArray(),kind:e.kind}));
      assert.deepEqual(poses(h.game),plain(expected));
      assert.deepEqual(h.game.registry.map(e=>e.idIndex),baseline.registry.map(e=>e.idIndex));
      assert.equal(h.game.registry.length,baseline.registry.length);assert.equal(h.stats.templates,0);
      assert.equal(h.game.physics.player,null);assert.equal(h.game.state().player,null);
      assert.deepEqual(plain(h.game.bindingInfo()),{template:null,slot_source:null,inferred_slots:[],game_scale:null});
      h.game.seek(0.5);h.game.reset();assert.deepEqual(poses(h.game),plain(expected));
      assert.equal(JSON.stringify(p),before);
    } finally {h.close();}
  });
}
