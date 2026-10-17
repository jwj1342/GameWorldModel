// 资产按证据包围盒摆正：居中、缩放、不走样。
// Node 20: node --experimental-default-type=module --test tests/test_asset_fit.mjs
import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import { fitToExtent } from '../kernel/assets.js';
import { buildNode } from '../kernel/scene.js';

function cube(size = 2, offset = [5, 5, 5]) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(size, size, size), new THREE.MeshBasicMaterial());
  mesh.position.set(...offset);       // 故意不在原点：库里的模型原点五花八门
  return mesh;
}

function worldBox(object) {
  object.updateMatrixWorld(true);
  return new THREE.Box3().setFromObject(object);
}

test('摆正之后包围盒中心回到原点', () => {
  const fitted = fitToExtent(cube(2, [5, -3, 7]), [2, 2, 2]);
  const centre = worldBox(fitted).getCenter(new THREE.Vector3());
  assert.ok(centre.length() < 1e-6, `中心应该在原点，实际 ${centre.toArray()}`);
});

test('等比缩放：尺寸对上了，形状不走样', () => {
  const fitted = fitToExtent(cube(2, [0, 0, 0]), [6, 6, 6]);
  const size = worldBox(fitted).getSize(new THREE.Vector3());
  assert.ok(Math.abs(size.x - 6) < 1e-5 && Math.abs(size.y - 6) < 1e-5 && Math.abs(size.z - 6) < 1e-5);
});

test('包围盒各轴比例不同时不把模型压扁', () => {
  // 感知给出的包围盒是拟合到可见表面的，本身有系统偏差，逐轴拉伸等于把误差画进画面
  const fitted = fitToExtent(cube(2, [0, 0, 0]), [8, 2, 2]);
  const size = worldBox(fitted).getSize(new THREE.Vector3());
  assert.ok(Math.abs(size.x - size.y) < 1e-5 && Math.abs(size.y - size.z) < 1e-5,
    `应该保持立方体，实际 ${size.toArray().map(v => v.toFixed(2))}`);
});

test('显式要求拉伸时才逐轴缩放', () => {
  const fitted = fitToExtent(cube(2, [0, 0, 0]), [8, 2, 4], 'stretch');
  const size = worldBox(fitted).getSize(new THREE.Vector3());
  assert.ok(Math.abs(size.x - 8) < 1e-5 && Math.abs(size.y - 2) < 1e-5 && Math.abs(size.z - 4) < 1e-5);
});

test('包围盒缺失或非法时原样返回，不炸也不缩成零', () => {
  const before = worldBox(cube(2, [0, 0, 0])).getSize(new THREE.Vector3());
  for (const bad of [undefined, [0, 0, 0], [NaN, 1, 1]]) {
    const size = worldBox(fitToExtent(cube(2, [0, 0, 0]), bad)).getSize(new THREE.Vector3());
    assert.ok(size.x > 0 && size.y > 0 && size.z > 0, `extent=${JSON.stringify(bad)} 不该把模型缩没`);
  }
  assert.ok(before.x > 0);
});

test('挂了资产之后碰撞体和没挂时一模一样', () => {
  // 这是整条资产路线的核心约束：外形换成什么样，都不影响物理和可交互性
  const spec = { id: 'x', class: 'toy train', geom: { kind: 'asset', query: 'toy train', extent: [2, 1, 1] },
                 pose: { pos: [0, 0, 0] } };
  const plain = buildNode(spec, null);
  const withAsset = buildNode(spec, cube(1, [0, 0, 0]));
  assert.deepEqual(withAsset.colliders, plain.colliders);
  assert.notEqual(withAsset.group.children.length, 0);
});
