import fs from 'node:fs';
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {buildScene} from '../kernel/scene.js';

const fixture=process.env.GWM_GENERATED_ASSET_FIXTURE;
test('Generated asset extents match real GLTFLoader visuals and collision boxes',{skip:!fixture},async()=>{
  const f=JSON.parse(fs.readFileSync(fixture,'utf8')),loader=new GLTFLoader();
  async function load(file){const b=fs.readFileSync(file);return (await loader.parseAsync(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),'' )).scene;}
  const asset=await load(f.file),multi=await load(f.multi_file);
  const close=(a,b)=>a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<1e-5,`${a} != ${b}`));
  close(new THREE.Box3().setFromObject(asset).getSize(new THREE.Vector3()).toArray(),f.raw_extent);
  close(new THREE.Box3().setFromObject(multi).getSize(new THREE.Vector3()).toArray(),f.multi_extent);
  for(const node of f.nodes){
    const {scene,registry}=buildScene({static:[],objects:[node]},null,{[node.id]:asset});scene.updateMatrixWorld(true);
    const visual=new THREE.Box3().setFromObject(registry[0].group);
    close(visual.getSize(new THREE.Vector3()).toArray(),node.geom.extent);
    close(registry[0].colliders[0].extent,node.geom.extent);
    assert.ok(Math.abs(visual.min.y)<1e-6,`Grounded collider but visual bottom ${visual.min.y}`);
  }
});
