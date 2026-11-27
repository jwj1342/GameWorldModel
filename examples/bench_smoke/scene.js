// 两轴评测管线的冒烟场景：手写的，用来检查渲染和交互清单两条路都走得通。
// 不是智能体的产出，也不对应任何真实场景。里面故意放了两个有毛病的物体，
// 用来确认清单确实会把它们判成不通过。
export function describe(THREE, ctx) {
  const box = (w, h, d, material) => new THREE.Mesh(new THREE.BoxGeometry(w, h, d), material);
  const at = (object, x, y, z) => { object.position.set(x, y, z); return object; };
  const out = [];

  out.push({ id: 'floor', kind: 'static', object3D: at(box(6, 0.2, 6, ctx.material('oak_floor', { repeat: [4, 4] })), 0, -0.1, 0) });
  out.push({ id: 'wall', kind: 'static', object3D: at(box(6, 2.6, 0.2, ctx.material('white_plaster', { repeat: [3, 1] })), 0, 1.3, -3) });

  const table = new THREE.Group();
  const wood = ctx.material('oak_floor');
  table.add(at(box(1.2, 0.05, 0.7, wood), 0, 0.725, 0));
  for (const [x, z] of [[-0.55, -0.3], [0.55, -0.3], [-0.55, 0.3], [0.55, 0.3]]) table.add(at(box(0.05, 0.7, 0.05, wood), x, 0.35, z));
  out.push({ id: 'table', kind: 'movable', mass_kg: 12, object3D: at(table, 0, 0, 0) });

  out.push({ id: 'mug', kind: 'movable', mass_kg: 0.3, object3D: at(box(0.09, 0.1, 0.09, ctx.material('grey_fabric')), 0.2, 0.8, 0) });
  // 故意悬空 40 厘米：放开会掉下去，rests 应该不通过
  out.push({ id: 'floating_box', kind: 'movable', object3D: at(box(0.3, 0.3, 0.3, ctx.material('grey_fabric')), 1.6, 0.55, 0.8) });
  // 故意声明成搬不动的沙发：movable 应该不通过
  out.push({ id: 'sofa', kind: 'static', object3D: at(box(1.8, 0.8, 0.9, ctx.material('grey_fabric', { repeat: [2, 1] })), -1.8, 0.4, -1.2) });
  return out;
}

export function environment(THREE) {
  const sun = new THREE.DirectionalLight(0xffffff, 1.8);
  sun.position.set(3, 6, 4);
  sun.castShadow = true;
  return { background: '#cfd6dd', lights: [new THREE.HemisphereLight(0xffffff, 0x555566, 1.1), sun] };
}
