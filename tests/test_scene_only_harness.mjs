// CPU wiring only: stub browser and filesystem, no browser/HTTP/pixel rendering.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import test from 'node:test';
const common=fs.readFileSync(new URL('../harness/common.mjs',import.meta.url),'utf8');
async function opened(options) {
  let url=null,launched=false,serverClosed=false,browserClosed=false;
  const page={on(){},goto:async value=>{url=value;},waitForFunction:async()=>{},evaluate:async()=>({readyState:'ready'})};
  const ctx={http:{createServer:()=>({listen(_port,_host,fn){fn();},address:()=>({port:1234}),close(){serverClosed=true;}})},
    fs:{},path,chromium:{launch:async()=>{launched=true;return{newPage:async()=>page,close:async()=>{browserClosed=true;}};}},process:{env:{}},console};
  vm.createContext(ctx);vm.runInContext(common.replace(/^import .*;\r?\n/gm,'').replaceAll('export ',''),ctx);
  const g=await ctx.openGame('synthetic',options);await g.close();
  return {url,launched,serverClosed,browserClosed};
}
test('openGame default URL remains compatible; explicit sceneOnly reaches kernel',async()=>{
  const normal=await opened({});assert.equal(normal.url,'http://localhost:1234/index.html?headless=1&w=640&h=360');
  const scene=await opened({sceneOnly:true});assert(scene.url.includes('headless=1&scene_only=1&'));assert(scene.serverClosed&&scene.browserClosed);
  const interactive=await opened({interactive:true});assert.equal(interactive.url,'http://localhost:1234/index.html?w=640&h=360');
});
test('openGame rejects ambiguous types and scene-only interactive gameplay before launch',async()=>{
  await assert.rejects(()=>opened({sceneOnly:'false'}),/boolean/);
  await assert.rejects(()=>opened({sceneOnly:true,interactive:true}),/interactive gameplay/);
});
async function renderScript(args) {
  let options=null,index=null;
  const game={registry:[],dt:1/60,program:{},mode(){},seek:t=>t,cameraInfo:()=>({}),render:()=>'',state:()=>({})};
  const ctx={fs:{mkdirSync(){},writeFileSync(_p,s){index=JSON.parse(s);}},path,process:{argv:[]},parseArgs:()=>({game:'fixture',out:'fixture-out',...args}),
    openGame:async(_game,opts)=>{options=opts;return{page:{evaluate:async(fn,v)=>fn(v)},logs:[],close:async()=>{}};},
    savePng(){},window:{__game:game},Date,Number,console:{log(){}}};
  const code=fs.readFileSync(new URL('../harness/render.mjs',import.meta.url),'utf8').replace(/^import .*;\r?\n/gm,'');
  vm.createContext(ctx);await vm.runInContext(`(async()=>{${code}})()`,ctx);
  return {options,index};
}
test('render CLI opt-in reaches openGame and is recorded; default stays false',async()=>{
  for(const [value,want] of [[undefined,false],['false',false],['0',false],['true',true],['1',true]]) {
    const args=value===undefined?{}:{'scene-only':value};const result=await renderScript(args);
    assert.equal(result.options.sceneOnly,want);assert.equal(result.index.scene_only,want);
  }
  await assert.rejects(()=>renderScript({'scene-only':'yes'}),/expects/);
});
