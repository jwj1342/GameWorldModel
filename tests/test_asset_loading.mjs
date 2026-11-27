import { test } from 'node:test';
import assert from 'node:assert/strict';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { loadAssets } from '../kernel/assets.js';

function glb() {
  const json=Buffer.from(JSON.stringify({asset:{version:'2.0'},scene:0,scenes:[{nodes:[]}],nodes:[]}));
  const padded=Buffer.concat([json,Buffer.alloc((-json.length)&3,32)]);
  const out=Buffer.alloc(20+padded.length);
  out.write('glTF');out.writeUInt32LE(2,4);out.writeUInt32LE(out.length,8);
  out.writeUInt32LE(padded.length,12);out.write('JSON',16);padded.copy(out,20);
  return out;
}
async function fixture(action,run) {
  const oldFetch=globalThis.fetch,oldDocument=globalThis.document;
  const calls=[];
  globalThis.document={baseURI:'https://example.invalid/game/index.html'};
  globalThis.fetch=async input=>{const url=String(input);calls.push(url);return action(url);};
  try {return await run(calls);} finally {globalThis.fetch=oldFetch;globalThis.document=oldDocument;}
}
test('reads native bytes and retains resource directory for relative dependencies',async()=>{
  const paths=[],original=GLTFLoader.prototype.parseAsync;
  GLTFLoader.prototype.parseAsync=function(data,base){paths.push(base);return original.call(this,data,base);};
  try {
    await fixture(url=>url.endsWith('assets.json')
      ?Response.json({objects:{box:{file:'models/box/model.glb'}}})
      :new Response(glb()),async calls=>{
        const result=await loadAssets('.');
        assert.deepEqual(Object.keys(result.scenes),['box']);assert.deepEqual(result.failures,[]);
        assert.equal(calls.length,2);
        assert.equal(calls[1],'https://example.invalid/game/models/box/model.glb');
        assert.deepEqual(paths,['https://example.invalid/game/models/box/']);
      });
  } finally {GLTFLoader.prototype.parseAsync=original;}
});
for(const mode of ['http','transport','body','parse']) {
  test(`keeps ${mode} failures explicit`,async()=>{
    await fixture(url=>{
      if(url.endsWith('assets.json'))return Response.json({objects:{box:{file:'assets/box.glb'}}});
      if(mode==='http')return new Response('missing',{status:404});
      if(mode==='transport')throw new Error('simulated connection failure');
      if(mode==='body')return new Response(new ReadableStream({start(controller){controller.error(new Error('simulated interrupted body'));}}));
      return new Response('not a glTF');
    },async()=>{
      const result=await loadAssets('.');
      assert.deepEqual(Object.keys(result.scenes),[]);
      assert.equal(result.failures.length,1);assert.equal(result.failures[0].id,'box');
      if(mode==='http')assert.match(result.failures[0].error,/HTTP 404/);
    });
  });
}
test('manifest with no assets does not fetch model data',async()=>{
  await fixture(()=>Response.json({objects:{}}),async calls=>{
    assert.deepEqual(await loadAssets('.'),{scenes:{},failures:[]});assert.equal(calls.length,1);
  });
});
