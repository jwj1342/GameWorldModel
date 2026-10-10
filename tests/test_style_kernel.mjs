// 样式那一半在运行时真的生效吗。CPU 上跑，不开浏览器、不碰 WebGL。
// Node 20: node --experimental-default-type=module --test tests/test_style_kernel.mjs
import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import { makeMaterial, buildScene, MATERIALS, DEFAULT_LIGHTING } from '../kernel/scene.js';

const program = (style) => ({
  meta: { clip: 'x', duration: 4, fps: 30, units: 'm', up: 'y' },
  camera: { keyframes: [{ t: 0, pos: [8, 6, 8], look_at: [0, 0, 0] }] },
  static: [], objects: [], ...(style ? { style } : {}),
});

test('调色板的词照旧', () => {
  const m = makeMaterial('wood');
  assert.equal(m.color.getHex(), MATERIALS.wood.color);
});

test('结构化材质按字段逐项生效', () => {
  const m = makeMaterial({ base_color: '#c04a3b', roughness: 0.2, metalness: 0.8 });
  assert.equal(m.color.getHex(), 0xc04a3b);
  assert.equal(m.roughness, 0.2);
  assert.equal(m.metalness, 0.8);
});

test('自发光与透明要真的打开对应开关', () => {
  const lava = makeMaterial({ base_color: '#ff5a1f', emissive: '#ff2200', emissive_intensity: 0.8 });
  assert.equal(lava.emissive.getHex(), 0xff2200);
  assert.equal(lava.emissiveIntensity, 0.8);
  const glass = makeMaterial({ base_color: '#bfe4ea', opacity: 0.4 });
  assert.equal(glass.transparent, true);
  assert.equal(glass.opacity, 0.4);
  // 不透明的不该被误开 transparent：开了会进透明队列，排序和性能都变
  assert.equal(makeMaterial({ base_color: '#bfe4ea', opacity: 1 }).transparent, false);
});

test('不写 lighting 的老程序，灯和原来逐项相同', () => {
  const { lights } = buildScene(program());
  assert.equal(lights.hemi.color.getHex(), 0xffffff);
  assert.equal(lights.hemi.groundColor.getHex(), 0x556655);
  assert.equal(lights.hemi.intensity, 1.1);
  assert.equal(lights.sun.color.getHex(), 0xffffff);
  assert.equal(lights.sun.intensity, 1.6);
  assert.equal(lights.sun.castShadow, true);
  // 原来写死的是 position.set(6, 12, 8)，方向要一致
  const want = new THREE.Vector3(6, 12, 8).normalize();
  assert.ok(lights.sun.position.clone().normalize().distanceTo(want) < 1e-9);
});

test('写了 lighting 就照着写的来', () => {
  const { lights } = buildScene(program({
    background: '#102030',
    lighting: { ambient: { sky_color: '#ffeedd', ground_color: '#223344', intensity: 0.6 },
                key: { color: '#fff2cc', intensity: 2.4, direction: [-4, 9, 3], shadow: false } },
  }));
  assert.equal(lights.hemi.color.getHex(), 0xffeedd);
  assert.equal(lights.hemi.groundColor.getHex(), 0x223344);
  assert.equal(lights.hemi.intensity, 0.6);
  assert.equal(lights.sun.color.getHex(), 0xfff2cc);
  assert.equal(lights.sun.intensity, 2.4);
  assert.equal(lights.sun.castShadow, false);
  const want = new THREE.Vector3(-4, 9, 3).normalize();
  assert.ok(lights.sun.position.clone().normalize().distanceTo(want) < 1e-9);
});

test('只写一半也行，另一半落回缺省', () => {
  const { lights } = buildScene(program({ lighting: { key: { intensity: 3 } } }));
  assert.equal(lights.sun.intensity, 3);
  assert.equal(lights.hemi.intensity, DEFAULT_LIGHTING.ambient.intensity);
  assert.equal(lights.sun.castShadow, DEFAULT_LIGHTING.key.shadow);
});
