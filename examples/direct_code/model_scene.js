// 手写的样例，用来验证直接出代码这条路径的宿主能不能跑通。
// 内容刻意和 examples/handwritten 的核心部分对齐：一块地面、一个上下运动的平台、
// 一个自转的小物体，这样可以直接比较两条路径。
export function describe(THREE) {
  const mat = (c) => new THREE.MeshStandardMaterial({ color: c, roughness: 0.8 });

  const ground = new THREE.Mesh(new THREE.BoxGeometry(24, 0.5, 16), mat(0x6b8f4e));
  ground.position.set(0, -0.25, 0);

  const lift = new THREE.Mesh(new THREE.BoxGeometry(2.5, 0.3, 2.5), mat(0x8899aa));
  lift.position.set(4.5, 1.2, -5);

  const coin = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.25, 0.06, 24), mat(0xd4af37));
  coin.rotation.x = Math.PI / 2;
  coin.position.set(-3, 1.0, 2);

  return [
    { id: 'ground', class: 'floor', kind: 'static', object3D: ground },
    {
      id: 'lift', class: 'platform', kind: 'object', object3D: lift,
      pose: (t) => ({ pos: [4.5, 1.2 + 1.2 * Math.sin(2 * Math.PI * t / 4), -5], quat: [0, 0, 0, 1] }),
    },
    {
      id: 'coin', class: 'coin', kind: 'object', object3D: coin,
      pose: (t) => {
        const a = (180 * Math.PI / 180) * t;        // 每秒 180 度
        // pose 替换根的绝对朝向，因此显式包含初始 x 轴转向。
        const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), a)
          .multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2));
        return { pos: [-3, 1.0, 2], quat: q.toArray() };
      },
    },
  ];
}
