// scene.js — 真实客厅（钢琴、转角沙发、电视墙）的 three.js 重建。
// 坐标：y 朝上，地面 y=0，墙大致沿 x/z 轴。单位米。
// 房间轮廓：主体 x[-1.5,1.4] z[-2.45,2.45]，门厅凹角 x[-1.5,-0.44] z[2.45,3.42]。

function box(THREE, w, h, d, mat) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  m.castShadow = true; m.receiveShadow = true;
  return m;
}
function cyl(THREE, rTop, rBot, h, mat, seg = 20) {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(rTop, rBot, h, seg), mat);
  m.castShadow = true; m.receiveShadow = true;
  return m;
}
function sph(THREE, r, mat) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(r, 18, 14), mat);
  m.castShadow = true; m.receiveShadow = true;
  return m;
}
// 放一个盒子：中心 (x,y,z)，尺寸 (w,h,d)
function put(parent, mesh, x, y, z, ry = 0, rx = 0, rz = 0) {
  mesh.position.set(x, y, z);
  if (ry) mesh.rotation.y = ry;
  if (rx) mesh.rotation.x = rx;
  if (rz) mesh.rotation.z = rz;
  parent.add(mesh);
  return mesh;
}
// 以范围方式放盒子
function span(parent, THREE, mat, x0, x1, y0, y1, z0, z1) {
  return put(parent, box(THREE, x1 - x0, y1 - y0, z1 - z0, mat),
    (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2);
}

export function describe(THREE, ctx) {
  const out = [];
  const M = (name, opts) => ctx.material(name, opts);

  // ---------- 地板 ----------
  {
    const g = new THREE.Group();
    const mat = M('oak_floor', { repeat: [3, 6] });
    span(g, THREE, mat, -1.58, 1.48, -0.05, 0.0, -2.53, 3.46);
    out.push({ id: 'floor', kind: 'static', object3D: g });
  }

  // ---------- 墙（含天花、窗帘、窗外套件、墙上挂饰） ----------
  {
    const g = new THREE.Group();
    const wall = M('wall_cream', { repeat: [3, 2] });
    const ceil = M('ceiling_white', { repeat: [3, 6] });
    const dark = M('hallway_dark');
    const frame = M('door_white');
    const curtain = M('curtain_olive', { repeat: [1.5, 2] });

    // 天花
    span(g, THREE, ceil, -1.58, 1.48, 2.50, 2.56, -2.53, 3.46);
    // -z 墙（扶手椅后面的窗）
    span(g, THREE, wall, -1.58, -0.75, 0, 2.5, -2.53, -2.45);
    span(g, THREE, wall, 0.35, 1.48, 0, 2.5, -2.53, -2.45);
    span(g, THREE, wall, -0.75, 0.35, 0, 1.00, -2.53, -2.45);
    span(g, THREE, wall, -0.75, 0.35, 2.10, 2.5, -2.53, -2.45);
    // +x 墙（两个壁龛洞口，中间是挂电视的平墙）
    span(g, THREE, wall, 1.40, 1.48, 0, 2.5, -2.45, -2.22);
    span(g, THREE, wall, 1.40, 1.48, 0, 2.5, -1.75, 0.18);
    span(g, THREE, wall, 1.40, 1.48, 0, 2.5, 1.12, 2.45);
    span(g, THREE, wall, 1.40, 1.48, 2.30, 2.5, -2.22, -1.75);
    span(g, THREE, wall, 1.40, 1.48, 2.30, 2.5, 0.18, 1.12);
    // 远端墙 z=2.45（钢琴窗所在）
    span(g, THREE, wall, -0.44, 0.15, 0, 2.5, 2.45, 2.53);
    span(g, THREE, wall, 1.10, 1.48, 0, 2.5, 2.45, 2.53);
    span(g, THREE, wall, 0.15, 1.10, 0, 1.12, 2.45, 2.53);
    span(g, THREE, wall, 0.15, 1.10, 2.10, 2.5, 2.45, 2.53);
    // 门厅凹角
    span(g, THREE, wall, -0.48, -0.40, 0, 2.5, 2.45, 3.46);
    span(g, THREE, wall, -1.58, -0.40, 0, 2.5, 3.38, 3.46);
    // -x 墙 + 门洞
    span(g, THREE, wall, -1.58, -1.50, 0, 2.5, -2.45, 2.40);
    span(g, THREE, wall, -1.58, -1.50, 0, 2.5, 3.26, 3.46);
    span(g, THREE, wall, -1.58, -1.50, 2.04, 2.5, 2.40, 3.26);
    // 门洞里的黑暗过道
    span(g, THREE, dark, -1.70, -1.64, 0, 2.2, 2.30, 3.36);

    // -z 窗的窗帘（橄榄色落地帘）
    span(g, THREE, curtain, -1.18, -0.72, 0.02, 2.30, -2.44, -2.28);
    span(g, THREE, curtain, 0.32, 0.98, 0.02, 2.30, -2.44, -2.28);

    // 电视上方和左侧墙上的挂钟和相框
    const clock = cyl(THREE, 0.17, 0.17, 0.03, M('clock_face'), 24);
    put(g, clock, 1.395, 1.78, -0.42, 0, 0, Math.PI / 2);
    span(g, THREE, M('photo_frame'), 1.38, 1.41, 1.41, 1.83, -1.13, -0.77);
    span(g, THREE, M('flower_pink'), 1.385, 1.405, 1.47, 1.77, -1.07, -0.83);
    span(g, THREE, M('photo_frame'), 1.38, 1.41, 1.52, 1.88, -0.08, 0.12);

    out.push({ id: 'walls', kind: 'static', object3D: g });
  }

  // ---------- 扶手椅后的窗（名单外） ----------
  {
    const g = new THREE.Group();
    const bright = M('window_bright', { emissive: new THREE.Color('#c9d4d6'), emissiveIntensity: 0.9 });
    const frame = M('door_white');
    span(g, THREE, bright, -0.75, 0.35, 1.00, 2.10, -2.505, -2.485);
    span(g, THREE, frame, -0.79, 0.39, 2.10, 2.15, -2.52, -2.46);
    span(g, THREE, frame, -0.79, 0.39, 0.96, 1.00, -2.52, -2.44);
    span(g, THREE, frame, -0.79, -0.75, 1.00, 2.10, -2.52, -2.46);
    span(g, THREE, frame, 0.35, 0.39, 1.00, 2.10, -2.52, -2.46);
    span(g, THREE, frame, -0.02, 0.02, 1.00, 2.10, -2.51, -2.47);
    const sheer = M('sheer_white', { transparent: true, opacity: 0.55 });
    span(g, THREE, sheer, -0.73, 0.33, 1.02, 2.12, -2.45, -2.435);
    out.push({ id: 'window_south', kind: 'static', object3D: g });
  }

  // ---------- 钢琴后面的窗（名单：window） ----------
  {
    const g = new THREE.Group();
    const bright = M('window_bright', { emissive: new THREE.Color('#cdd8da'), emissiveIntensity: 0.9 });
    const frame = M('door_white');
    span(g, THREE, bright, 0.15, 1.10, 1.12, 2.10, 2.475, 2.495);
    span(g, THREE, frame, 0.11, 1.14, 2.10, 2.15, 2.44, 2.50);
    span(g, THREE, frame, 0.10, 1.15, 1.06, 1.12, 2.38, 2.52); // 窗台
    span(g, THREE, frame, 0.11, 0.15, 1.12, 2.10, 2.44, 2.50);
    span(g, THREE, frame, 1.10, 1.14, 1.12, 2.10, 2.44, 2.50);
    span(g, THREE, frame, 0.60, 0.64, 1.12, 2.10, 2.46, 2.49);
    span(g, THREE, M('blind_grey', { repeat: [2, 1] }), 0.12, 1.13, 1.80, 2.14, 2.41, 2.445);
    out.push({ id: 'window', kind: 'static', object3D: g });
  }

  // ---------- 门（铰链） ----------
  {
    const g = new THREE.Group();
    const white = M('door_white', { repeat: [1, 2] });
    // 局部坐标：门板中心在原点，铰链边在局部 z=+0.40
    span(g, THREE, white, -0.021, 0.021, -0.995, 1.025, -0.40, 0.40);
    // 门板凹槽线条（两面各两块）
    for (const sx of [-0.024, 0.024]) {
      span(g, THREE, white, sx - 0.004, sx + 0.004, -0.85, -0.15, -0.30, 0.28);
      span(g, THREE, white, sx - 0.004, sx + 0.004, 0.05, 0.85, -0.30, 0.28);
    }
    // 把手
    span(g, THREE, M('chrome'), -0.075, -0.025, -0.03, 0.01, -0.36, -0.26);
    g.position.set(-1.117, 1.005, 3.032);
    g.quaternion.setFromEuler(new THREE.Euler(0, -1.08, 0));
    out.push({
      id: 'door', kind: 'movable', mass_kg: 18, object3D: g,
      interaction: { grab: 'hinge', axis: [0, 1, 0], pivot: [-1.47, 1.0, 3.22], limit: [-0.45, 1.05] },
    });
  }

  // ---------- 壁龛搁架（两处白色凹龛 + 下面的桌子/柜子） ----------
  {
    const g = new THREE.Group();
    const white = M('shelf_white', { repeat: [1, 1] });
    const books = M('book_spines', { repeat: [1, 1] });
    const games = M('game_boxes');
    const frameM = M('photo_frame');

    // 凹龛一（游戏）：z -2.22..-1.73
    span(g, THREE, white, 1.62, 1.66, 0, 2.30, -2.22, -1.73);        // 背板
    span(g, THREE, white, 1.40, 1.66, 0, 2.30, -2.26, -2.22);        // 侧帮
    span(g, THREE, white, 1.40, 1.66, 0, 2.30, -1.73, -1.69);
    span(g, THREE, white, 1.40, 1.65, 1.40, 1.435, -2.22, -1.73);    // 层板
    span(g, THREE, white, 1.40, 1.65, 1.80, 1.835, -2.22, -1.73);
    // 顶层游戏盒
    span(g, THREE, games, 1.42, 1.62, 1.835, 1.95, -2.20, -2.02);
    span(g, THREE, games, 1.42, 1.62, 1.835, 1.93, -2.00, -1.86);
    span(g, THREE, books, 1.42, 1.60, 1.835, 2.02, -1.85, -1.75);
    // 中层：相框和杂物
    span(g, THREE, frameM, 1.44, 1.60, 1.435, 1.79, -2.14, -1.96);
    span(g, THREE, M('glass_clear'), 1.46, 1.58, 1.435, 1.60, -1.90, -1.80);
    // 龛下白桌 + 打印机
    span(g, THREE, white, 1.02, 1.40, 0.70, 0.75, -2.14, -1.76);
    for (const [lx, lz] of [[1.05, -2.11], [1.05, -1.79], [1.37, -2.11], [1.37, -1.79]]) {
      span(g, THREE, white, lx - 0.02, lx + 0.02, 0, 0.70, lz - 0.02, lz + 0.02);
    }
    span(g, THREE, M('speaker_black'), 1.10, 1.36, 0.75, 0.90, -2.08, -1.86);
    // 桌上斜靠的粉色花相框
    put(g, box(THREE, 0.03, 0.34, 0.28, frameM), 1.16, 0.90, -1.70, 0, 0, 0.15);
    put(g, box(THREE, 0.032, 0.24, 0.20, M('flower_pink')), 1.15, 0.90, -1.70, 0, 0, 0.15);

    // 凹龛三（书）：z 0.18..1.14
    span(g, THREE, white, 1.62, 1.66, 0, 2.30, 0.18, 1.14);
    span(g, THREE, white, 1.40, 1.66, 0, 2.30, 0.14, 0.18);
    span(g, THREE, white, 1.40, 1.66, 0, 2.30, 1.14, 1.18);
    for (const sy of [0.85, 1.28, 1.70, 2.06]) {
      span(g, THREE, white, 1.40, 1.65, sy, sy + 0.03, 0.18, 1.14);
    }
    // 各层书
    span(g, THREE, books, 1.42, 1.62, 0.88, 1.24, 0.22, 1.06);
    span(g, THREE, books, 1.42, 1.62, 1.31, 1.66, 0.22, 0.98);
    span(g, THREE, books, 1.42, 1.62, 1.73, 2.04, 0.30, 1.10);
    span(g, THREE, books, 1.42, 1.62, 2.09, 2.28, 0.24, 0.90);
    // 底层黑胶唱片
    span(g, THREE, books, 1.44, 1.63, 0.02, 0.38, 0.22, 0.66);
    // 龛下白柜 + 斜靠的相框
    span(g, THREE, white, 1.14, 1.40, 0, 0.72, 0.74, 1.12);
    put(g, box(THREE, 0.03, 0.40, 0.32, frameM), 1.16, 0.90, 0.90, 0, 0, 0.17);

    out.push({ id: 'alcove_shelves', kind: 'static', object3D: g });
  }

  // ---------- 松木书柜（名单外，static，立在 +x 墙尽头、钢琴窗右侧） ----------
  {
    const g = new THREE.Group();
    const pine = M('pine_bookcase', { repeat: [1, 2] });
    const books = M('book_spines', { repeat: [1.4, 1] });
    span(g, THREE, pine, 1.08, 1.42, 0, 1.92, 1.25, 1.28);    // 侧板
    span(g, THREE, pine, 1.08, 1.42, 0, 1.92, 2.12, 2.15);
    span(g, THREE, pine, 1.39, 1.42, 0, 1.92, 1.25, 2.15);    // 背板
    span(g, THREE, pine, 1.08, 1.42, 1.89, 1.92, 1.25, 2.15); // 顶板
    for (const sy of [0.02, 0.40, 0.78, 1.16, 1.54]) {
      span(g, THREE, pine, 1.08, 1.40, sy, sy + 0.03, 1.28, 2.12);
      span(g, THREE, books, 1.10, 1.38, sy + 0.03, sy + 0.36, 1.30, 2.10);
    }
    // 地球仪
    span(g, THREE, M('black_metal'), 1.20, 1.28, 1.92, 1.96, 1.66, 1.74);
    put(g, sph(THREE, 0.115, M('globe_tan')), 1.24, 2.06, 1.70);
    out.push({ id: 'bookcase_pine', kind: 'static', object3D: g });
  }

  // ---------- 转角沙发（static：一个人搬不动） ----------
  {
    const g = new THREE.Group();
    const fab = M('sofa_grey', { repeat: [2.5, 1.5] });
    // 主排 z -1.72..0.52
    span(g, THREE, fab, -1.47, -0.95, 0.02, 0.18, -1.72, 0.52);   // 底座（前面内收，给 C 形边几留位）
    span(g, THREE, fab, -1.47, -0.72, 0.18, 0.34, -1.72, 0.52);   // 座框
    span(g, THREE, fab, -1.44, -0.72, 0.34, 0.46, -1.58, 0.50);   // 座垫
    span(g, THREE, fab, -1.47, -1.22, 0.18, 0.88, -1.72, 0.52);   // 靠背
    span(g, THREE, fab, -1.26, -1.08, 0.46, 0.86, -1.60, 0.46);   // 靠垫
    span(g, THREE, fab, -1.47, -0.72, 0.18, 0.62, -1.72, -1.56);  // -z 端扶手
    // 转角榻 z 0.52..1.30
    span(g, THREE, fab, -1.47, -0.60, 0.02, 0.18, 0.52, 1.30);
    span(g, THREE, fab, -1.47, -0.40, 0.18, 0.45, 0.52, 1.14);    // 榻面
    span(g, THREE, fab, -1.47, -0.40, 0.18, 0.58, 1.14, 1.30);    // +z 端扶手
    // 抱枕、毯子、笔记本
    put(g, box(THREE, 0.13, 0.32, 0.36, M('pillow_red')), -1.20, 0.62, -1.42, 0, 0, 0.25);
    put(g, box(THREE, 0.13, 0.30, 0.34, M('pillow_paisley')), -1.16, 0.60, -1.02, 0, 0, 0.22);
    span(g, THREE, M('blanket_dark', { repeat: [1.5, 1] }), -1.38, -0.55, 0.45, 0.49, 0.58, 1.10);
    span(g, THREE, M('speaker_black'), -1.12, -0.80, 0.49, 0.515, 0.72, 1.04);
    out.push({ id: 'sofa', kind: 'static', object3D: g });
  }

  // ---------- 深蓝色单人躺椅（可挪动） ----------
  {
    const g = new THREE.Group();
    const fab = M('armchair_navy', { repeat: [1.5, 1.5] });
    span(g, THREE, fab, -0.66, 0.22, 0.0, 0.30, -2.32, -1.52);    // 底座
    span(g, THREE, fab, -0.60, 0.16, 0.30, 0.46, -2.14, -1.54);   // 座垫
    span(g, THREE, fab, -0.58, 0.14, 0.16, 0.40, -1.56, -1.40);   // 前伸脚踏
    span(g, THREE, fab, -0.62, 0.18, 0.30, 1.00, -2.42, -2.14);   // 靠背
    span(g, THREE, fab, -0.56, 0.12, 0.98, 1.10, -2.42, -2.18);   // 头枕
    span(g, THREE, fab, -0.70, -0.56, 0.30, 0.60, -2.32, -1.56);  // 左扶手
    span(g, THREE, fab, 0.12, 0.28, 0.30, 0.60, -2.32, -1.56);    // 右扶手
    g.position.set(0.02, 0, 0);
    out.push({ id: 'armchair', kind: 'movable', mass_kg: 38, object3D: g });
  }

  // ---------- 躺椅上的奶牛纹靠垫 ----------
  {
    const g = new THREE.Group();
    put(g, box(THREE, 0.44, 0.36, 0.13, M('cushion_cow')), 0, 0, 0, 0, -0.35, 0);
    g.position.set(-0.22, 0.655, -2.04);
    out.push({ id: 'armchair_cushion', kind: 'movable', mass_kg: 0.8, object3D: g });
  }

  // ---------- 脚凳（弯木 + 蓝垫） ----------
  {
    const g = new THREE.Group();
    const wood = M('bentwood');
    // 弯木侧框（两片，略斜）
    put(g, box(THREE, 0.05, 0.34, 0.50, wood), -0.23, 0.17, 0, 0, 0, 0.10);
    put(g, box(THREE, 0.05, 0.34, 0.50, wood), 0.23, 0.17, 0, 0, 0, -0.10);
    // 横向撑
    span(g, THREE, wood, -0.24, 0.24, 0.08, 0.12, -0.21, -0.17);
    span(g, THREE, wood, -0.24, 0.24, 0.08, 0.12, 0.17, 0.21);
    // 坐垫
    span(g, THREE, M('footstool_blue'), -0.26, 0.26, 0.30, 0.40, -0.23, 0.23);
    g.position.set(-0.40, 0.014, -0.82);
    out.push({ id: 'footstool', kind: 'movable', mass_kg: 4, object3D: g });
  }

  // ---------- C 形边几（木面黑金属腿） ----------
  {
    const g = new THREE.Group();
    const metal = M('black_metal');
    span(g, THREE, metal, -0.92, -0.52, 0.004, 0.032, -0.16, 0.52);  // 落地底框
    span(g, THREE, metal, -0.56, -0.52, 0.03, 0.56, -0.16, -0.12);   // 立柱
    span(g, THREE, metal, -0.56, -0.52, 0.03, 0.56, 0.48, 0.52);
    span(g, THREE, M('walnut_top', { repeat: [1, 1] }), -0.92, -0.52, 0.56, 0.60, -0.18, 0.54);
    out.push({ id: 'coffee_table', kind: 'movable', mass_kg: 7, object3D: g });
  }

  // ---------- 边几上的不锈钢碗 ----------
  {
    const g = new THREE.Group();
    const steel = M('steel_bowl');
    put(g, cyl(THREE, 0.115, 0.075, 0.065, steel, 24), 0, 0.0325, 0);
    put(g, cyl(THREE, 0.105, 0.105, 0.004, M('hallway_dark'), 24), 0, 0.062, 0);
    g.position.set(-0.70, 0.602, 0.02);
    out.push({ id: 'bowl', kind: 'movable', mass_kg: 0.5, object3D: g });
  }

  // ---------- 玻璃水瓶 ----------
  {
    const g = new THREE.Group();
    const glass = M('glass_clear', { transparent: true, opacity: 0.45 });
    put(g, cyl(THREE, 0.055, 0.055, 0.19, glass), 0, 0.095, 0);
    put(g, cyl(THREE, 0.020, 0.048, 0.05, glass), 0, 0.215, 0);
    g.position.set(-0.80, 0.602, 0.30);
    out.push({ id: 'carafe', kind: 'movable', mass_kg: 0.9, object3D: g });
  }

  // ---------- 玻璃杯 ----------
  {
    const g = new THREE.Group();
    const glass = M('glass_clear', { transparent: true, opacity: 0.45 });
    put(g, cyl(THREE, 0.030, 0.030, 0.006, glass), 0, 0.003, 0);
    put(g, cyl(THREE, 0.006, 0.006, 0.065, glass), 0, 0.038, 0);
    put(g, cyl(THREE, 0.034, 0.024, 0.075, glass), 0, 0.108, 0);
    g.position.set(-0.62, 0.602, 0.40);
    out.push({ id: 'wine_glass', kind: 'movable', mass_kg: 0.25, object3D: g });
  }

  // ---------- 拖鞋 ----------
  function slipper(id, x, z, ry) {
    const g = new THREE.Group();
    const tan = M('slipper_tan');
    put(g, box(THREE, 0.095, 0.022, 0.26, tan), 0, 0.011, 0);
    put(g, box(THREE, 0.095, 0.045, 0.13, tan), 0, 0.032, -0.055);
    g.position.set(x, 0.014, z);
    g.quaternion.setFromEuler(new THREE.Euler(0, ry, 0));
    out.push({ id, kind: 'movable', mass_kg: 0.15, object3D: g });
  }
  slipper('slipper_left', -0.07, -0.70, 0.30);
  slipper('slipper_right', 0.10, -0.66, -0.12);

  // ---------- 地毯 ----------
  {
    const g = new THREE.Group();
    const dark = M('rug_dark_green', { repeat: [1.2, 1] });
    span(g, THREE, dark, -0.48, 0.55, 0, 0.013, -0.31, 0.46);   // 大块
    span(g, THREE, dark, 0.48, 0.68, 0, 0.013, -1.15, -0.31);   // 沿电视侧的条
    out.push({ id: 'rug_dark', kind: 'movable', mass_kg: 3, object3D: g });
  }
  {
    const g = new THREE.Group();
    const lt = M('rug_light_grey', { repeat: [1, 1.2] });
    span(g, THREE, lt, -0.70, 0.38, 0, 0.012, -0.79, -0.32);   // 靠沙发的一块
    span(g, THREE, lt, -0.70, -0.05, 0, 0.012, -1.36, -0.79);  // 伸向扶手椅的窄条
    out.push({ id: 'rug_light', kind: 'movable', mass_kg: 2, object3D: g });
  }
  {
    const g = new THREE.Group();
    const pr = M('rug_piano_pattern', { repeat: [1.2, 1.6] });
    span(g, THREE, pr, 0.30, 1.32, 0, 0.008, 1.02, 2.38);   // 钢琴身下
    span(g, THREE, pr, 0.05, 0.55, 0, 0.008, 0.58, 1.02);   // 伸向琴凳的一角
    out.push({ id: 'piano_rug', kind: 'movable', mass_kg: 3, object3D: g });
  }

  // ---------- 低音炮 ----------
  {
    const g = new THREE.Group();
    span(g, THREE, M('speaker_black'), 0.06, 0.46, 0.002, 0.46, -1.16, -0.80);
    out.push({ id: 'subwoofer', kind: 'movable', mass_kg: 14, object3D: g });
  }

  // ---------- 电视 ----------
  {
    const g = new THREE.Group();
    const scr = M('tv_screen');
    span(g, THREE, scr, 1.03, 1.09, 0.575, 1.265, -1.08, 0.10);   // 屏体
    const foot = M('black_metal');
    span(g, THREE, foot, 0.95, 1.27, 0.562, 0.585, -0.95, -0.80);
    span(g, THREE, foot, 0.95, 1.27, 0.562, 0.585, -0.10, 0.05);
    out.push({ id: 'tv', kind: 'movable', mass_kg: 16, object3D: g });
  }

  // ---------- 电视柜（玻璃 + 金属，内含音箱） ----------
  {
    const g = new THREE.Group();
    const glass = M('glass_shelf', { transparent: true, opacity: 0.5 });
    const chrome = M('chrome');
    span(g, THREE, glass, 0.86, 1.32, 0.53, 0.56, -1.10, 0.18);
    span(g, THREE, glass, 0.86, 1.32, 0.30, 0.33, -1.10, 0.18);
    span(g, THREE, glass, 0.86, 1.32, 0.09, 0.12, -1.10, 0.18);
    for (const [lx, lz] of [[0.89, -1.06], [0.89, 0.14], [1.29, -1.06], [1.29, 0.14]]) {
      span(g, THREE, chrome, lx - 0.015, lx + 0.015, 0, 0.53, lz - 0.015, lz + 0.015);
    }
    // 中置音箱和功放
    span(g, THREE, M('speaker_black'), 0.98, 1.26, 0.33, 0.46, -0.70, -0.20);
    span(g, THREE, M('speaker_black'), 0.95, 1.28, 0.12, 0.27, -0.55, -0.15);
    out.push({ id: 'tv_stand', kind: 'movable', mass_kg: 12, object3D: g });
  }

  // ---------- 支架音箱 ----------
  function speaker(id, cx, cz, flowers) {
    const g = new THREE.Group();
    const black = M('speaker_black');
    span(g, THREE, M('black_metal'), cx - 0.12, cx + 0.12, 0.002, 0.025, cz - 0.12, cz + 0.12);
    span(g, THREE, M('black_metal'), cx - 0.04, cx + 0.04, 0.025, 0.72, cz - 0.04, cz + 0.04);
    span(g, THREE, black, cx - 0.14, cx + 0.14, 0.72, 1.06, cz - 0.15, cz + 0.15);
    const cone = cyl(THREE, 0.055, 0.055, 0.012, M('cone_white'), 20);
    put(g, cone, cx - 0.145, 0.90, cz, 0, 0, Math.PI / 2);
    if (flowers) {
      put(g, sph(THREE, 0.075, M('flower_pink')), cx, 1.11, cz);
      span(g, THREE, M('terracotta'), cx - 0.05, cx + 0.05, 1.06, 1.10, cz - 0.05, cz + 0.05);
    }
    out.push({ id, kind: 'movable', mass_kg: 9, object3D: g });
  }
  speaker('speaker_left', 0.75, -1.40, false);
  speaker('speaker_right', 1.02, 0.55, true);

  // ---------- 绿植 ----------
  {
    const g = new THREE.Group();
    put(g, cyl(THREE, 0.145, 0.105, 0.30, M('terracotta')), 0, 0.17, 0);
    put(g, cyl(THREE, 0.16, 0.16, 0.02, M('terracotta')), 0, 0.012, 0);
    put(g, cyl(THREE, 0.026, 0.032, 0.55, M('bentwood')), 0, 0.57, 0);
    const leaf = M('plant_green');
    for (let i = 0; i < 8; i++) {
      const a = i * Math.PI / 4 + 0.3;
      const m = box(THREE, 0.05, 0.42, 0.09, leaf);
      m.position.set(Math.cos(a) * 0.07, 0.98, Math.sin(a) * 0.07);
      m.rotation.y = -a;
      m.rotation.z = Math.cos(a) * 0.18;
      m.rotation.x = Math.sin(a) * 0.18;
      g.add(m);
    }
    g.position.set(0.72, 0.0, 0.72);
    out.push({ id: 'plant', kind: 'movable', mass_kg: 7, object3D: g });
  }

  // ---------- 三角钢琴（static） ----------
  {
    const g = new THREE.Group();
    const black = M('piano_black');
    // 琴身三段近似弯曲侧（直边在 +x）
    span(g, THREE, black, 0.40, 1.42, 0.55, 0.98, 0.90, 1.42);
    span(g, THREE, black, 0.58, 1.42, 0.55, 0.98, 1.42, 2.00);
    span(g, THREE, black, 0.86, 1.42, 0.55, 0.98, 2.00, 2.40);
    // 键床和白键
    span(g, THREE, black, 0.44, 1.38, 0.66, 0.80, 0.80, 0.90);
    span(g, THREE, M('keys_white', { repeat: [3, 1] }), 0.48, 1.34, 0.715, 0.745, 0.815, 0.885);
    // 三条腿 + 踏板
    const leg = M('piano_black');
    put(g, cyl(THREE, 0.045, 0.035, 0.54, leg), 0.50, 0.28, 1.00);
    put(g, cyl(THREE, 0.045, 0.035, 0.54, leg), 1.32, 0.28, 1.00);
    put(g, cyl(THREE, 0.045, 0.035, 0.54, leg), 1.18, 0.28, 2.28);
    span(g, THREE, leg, 0.82, 1.00, 0.10, 0.30, 0.95, 1.00);
    // 乐谱架 + 琴谱
    put(g, box(THREE, 0.55, 0.26, 0.02, black), 0.91, 1.10, 1.32, 0, -0.28, 0);
    put(g, box(THREE, 0.42, 0.22, 0.008, M('paper_white')), 0.91, 1.12, 1.30, 0, -0.28, 0);
    // 琴盖上的书堆
    span(g, THREE, M('book_spines'), 1.05, 1.35, 0.98, 1.05, 1.55, 1.85);
    span(g, THREE, M('game_boxes'), 1.08, 1.32, 1.05, 1.10, 1.60, 1.82);
    out.push({ id: 'piano', kind: 'static', object3D: g });
  }

  // ---------- 琴凳（在键盘左端前面） ----------
  {
    const g = new THREE.Group();
    span(g, THREE, M('bench_black'), 0.07, 0.57, 0.44, 0.50, 0.62, 0.92);
    for (const [lx, lz] of [[0.11, 0.67], [0.11, 0.87], [0.53, 0.67], [0.53, 0.87]]) {
      span(g, THREE, M('piano_black'), lx - 0.02, lx + 0.02, 0.012, 0.44, lz - 0.02, lz + 0.02);
    }
    out.push({ id: 'piano_bench', kind: 'movable', mass_kg: 7, object3D: g });
  }

  // ---------- 暖气片（名单外） ----------
  {
    const g = new THREE.Group();
    span(g, THREE, M('radiator_white'), 0.18, 0.98, 0.42, 1.05, 2.31, 2.43);
    out.push({ id: 'radiator', kind: 'static', object3D: g });
  }

  // ---------- 钢琴边的纸箱（名单外） ----------
  {
    const g = new THREE.Group();
    span(g, THREE, M('bentwood'), 0.16, 0.46, 0.01, 0.42, 1.48, 1.82);
    out.push({ id: 'cardboard_box', kind: 'static', object3D: g });
  }

  // ---------- 沙发边的小桌（名单外，键盘放在上面） ----------
  {
    const g = new THREE.Group();
    const wood = M('bentwood');
    span(g, THREE, wood, -1.45, -1.00, 0.55, 0.59, -2.14, -1.76);
    for (const [lx, lz] of [[-1.42, -2.11], [-1.42, -1.79], [-1.03, -2.11], [-1.03, -1.79]]) {
      span(g, THREE, M('black_metal'), lx - 0.018, lx + 0.018, 0, 0.55, lz - 0.018, lz + 0.018);
    }
    put(g, box(THREE, 0.36, 0.025, 0.14, M('speaker_black')), -1.22, 0.605, -1.95, 0.1);
    put(g, cyl(THREE, 0.03, 0.025, 0.09, M('glass_clear', { transparent: true, opacity: 0.45 })), -1.36, 0.635, -1.82);
    out.push({ id: 'side_table', kind: 'static', object3D: g });
  }

  // ---------- 空气净化器（名单外） ----------
  {
    const g = new THREE.Group();
    put(g, cyl(THREE, 0.155, 0.155, 0.90, M('purifier_white'), 24), 0, 0.452, 0);
    put(g, cyl(THREE, 0.150, 0.155, 0.05, M('speaker_black'), 24), 0, 0.925, 0);
    g.position.set(-1.30, 0.002, 1.55);
    out.push({ id: 'purifier', kind: 'movable', mass_kg: 5, object3D: g });
  }

  return out;
}

export function environment(THREE) {
  const lights = [];
  const hemi = new THREE.HemisphereLight(0xfff1dc, 0x4a4038, 0.85);
  lights.push(hemi);
  // 主光：从 -z 窗进来的暖色平行光
  const sun = new THREE.DirectionalLight(0xffe3c0, 1.5);
  sun.position.set(-0.3, 2.1, -4.5);
  sun.target.position.set(0.2, 0.6, 0.5);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.left = -4; sun.shadow.camera.right = 4;
  sun.shadow.camera.top = 4; sun.shadow.camera.bottom = -4;
  sun.shadow.camera.near = 0.5; sun.shadow.camera.far = 12;
  sun.shadow.bias = -0.0004;
  lights.push(sun);
  // 钢琴窗方向的补光
  const fill = new THREE.DirectionalLight(0xf4ecdc, 0.55);
  fill.position.set(0.7, 1.9, 4.5);
  fill.target.position.set(0, 0.7, 0);
  lights.push(fill);
  // 顶部柔光
  const top = new THREE.DirectionalLight(0xfff6e8, 0.35);
  top.position.set(0.2, 4, 0.6);
  top.target.position.set(0, 0, 0.5);
  lights.push(top);
  return { background: '#191713', lights };
}
