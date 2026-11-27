// 真实房间重建：客厅 + 钢琴角。依据 24 张照片和稀疏点云测量。
// 坐标系：y 轴朝上，地面 y = 0，单位大致是米。
//   z = -2.55  南墙：凸窗 + 窗帘，窗前是躺椅
//   z = +3.85  北墙：钢琴后面的窗户
//   x = -1.38  西墙：沙发；北端有门洞（z 1.9..2.7），门开着
//   x = +1.38  东墙：电视 / 壁炉 / 两处凹进搁架 / 松木书架

export function describe(THREE, ctx) {
  const out = [];
  const H = 2.6; // 层高
  const RUG_TOP = 0.012; // 地毯厚度（可动物体，放在地板上面，留 2mm 缝）
  const GAP = 0.002;

  const r2 = (v) => Math.max(0.15, Math.round(v * 100) / 100);
  function mat(name, a = 1, b = 1) {
    return ctx.material(name, { repeat: [r2(a), r2(b)] });
  }
  // 盒子的贴图平铺次数按最大面估（BoxGeometry: x 面 u->z,v->y；y 面 u->x,v->z；z 面 u->x,v->y）
  function autoMat(name, sx, sy, sz) {
    if (sx <= sy && sx <= sz) return mat(name, sz, sy);
    if (sy <= sx && sy <= sz) return mat(name, sx, sz);
    return mat(name, sx, sy);
  }
  function group(x, y, z, rotY = 0) {
    const g = new THREE.Group();
    g.position.set(x, y, z);
    g.rotation.y = rotY;
    return g;
  }
  // 局部坐标 min/max 盒子
  function lbox(g, m, x0, x1, y0, y1, z0, z1) {
    const sx = x1 - x0, sy = y1 - y0, sz = z1 - z0;
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(sx, sy, sz),
      typeof m === 'string' ? autoMat(m, sx, sy, sz) : m,
    );
    mesh.position.set((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2);
    g.add(mesh);
    return mesh;
  }
  // 世界坐标 min/max 盒子（组无旋转时用）
  function wbox(g, m, x0, x1, y0, y1, z0, z1) {
    const p = g.position;
    return lbox(g, m, x0 - p.x, x1 - p.x, y0 - p.y, y1 - p.y, z0 - p.z, z1 - p.z);
  }
  // 直立圆柱，局部坐标底面中心
  function lcyl(g, m, x, y0, z, rTop, rBot, h, seg = 20) {
    const mesh = new THREE.Mesh(
      new THREE.CylinderGeometry(rTop, rBot, h, seg),
      typeof m === 'string' ? mat(m, Math.max(rTop, rBot) * 6, h) : m,
    );
    mesh.position.set(x, y0 + h / 2, z);
    g.add(mesh);
    return mesh;
  }
  function wcyl(g, m, x, y0, z, rTop, rBot, h, seg = 20) {
    const p = g.position;
    return lcyl(g, m, x - p.x, y0 - p.y, z - p.z, rTop, rBot, h, seg);
  }
  // 轴线沿 x 的圆片（音箱单元、挂钟）
  function xdisc(g, m, x, y, z, r, t = 0.008) {
    const mesh = new THREE.Mesh(new THREE.CylinderGeometry(r, r, t, 24), mat(m, r * 2, r * 2));
    mesh.rotation.z = Math.PI / 2;
    mesh.position.set(x, y, z);
    g.add(mesh);
    return mesh;
  }
  // 把可动物体的原点挪到几何中心：抓取点在中部，拖起来才稳
  function recenter(g) {
    const box = new THREE.Box3();
    for (const child of g.children) {
      child.updateMatrix();
      child.geometry.computeBoundingBox();
      box.union(child.geometry.boundingBox.clone().applyMatrix4(child.matrix));
    }
    const c = box.getCenter(new THREE.Vector3());
    for (const child of g.children) child.position.sub(c);
    g.position.add(c.applyQuaternion(g.quaternion));
    return g;
  }
  const add = (id, kind, object3D, extra = {}) => {
    if (kind === 'movable') recenter(object3D);
    out.push({ id, kind, object3D, ...extra });
  };

  // ------------------------------------------------------------------ 地板
  {
    const g = group(0, 0, 0);
    wbox(g, mat('pine_floor', 4, 9), -1.44, 1.44, -0.06, 0, -2.61, 3.91);
    add('floor', 'static', g);
  }

  // ------------------------------------------------------------------ 墙、天花板、门洞、走廊
  {
    const g = group(0, 0, 0);
    const W = 'wall_cream';
    // 西墙（带门洞 z 1.9..2.7）
    wbox(g, W, -1.44, -1.38, 0, H, -2.61, 1.9);
    wbox(g, W, -1.44, -1.38, 0, H, 2.7, 3.91);
    wbox(g, W, -1.44, -1.38, 2.02, H, 1.9, 2.7); // 门楣
    // 门洞另一侧的黑暗走廊
    wbox(g, 'hall_dark', -2.3, -2.2, 0, H, 1.75, 2.85);
    wbox(g, 'hall_dark', -2.2, -1.44, 0, H, 1.75, 1.85);
    wbox(g, 'hall_dark', -2.2, -1.44, 0, H, 2.75, 2.85);
    wbox(g, 'hall_dark', -2.2, -1.44, H, H + 0.05, 1.75, 2.85);
    wbox(g, 'hall_tile', -2.2, -1.44, -0.06, 0.001, 1.85, 2.75);
    // 门框
    wbox(g, 'trim_white', -1.385, -1.375, 0, 2.06, 1.86, 1.9);
    wbox(g, 'trim_white', -1.385, -1.375, 0, 2.06, 2.7, 2.74);
    wbox(g, 'trim_white', -1.385, -1.375, 2.02, 2.06, 1.9, 2.7);
    // 东墙（两处凹架开口 z -1.98..-1.18 和 z 0.12..1.12，y 0..2.05）
    wbox(g, W, 1.38, 1.44, 0, H, -2.61, -1.98);
    wbox(g, W, 1.38, 1.44, 2.05, H, -1.98, -1.18);
    wbox(g, W, 1.38, 1.44, 0, H, -1.18, 0.12);
    wbox(g, W, 1.38, 1.44, 2.05, H, 0.12, 1.12);
    wbox(g, W, 1.38, 1.44, 0, H, 1.12, 3.91);
    // 南墙（带凸窗洞 x -0.75..0.75, y 0.75..2.05）
    wbox(g, W, -1.44, -0.75, 0, H, -2.61, -2.55);
    wbox(g, W, 0.75, 1.44, 0, H, -2.61, -2.55);
    wbox(g, W, -0.75, 0.75, 0, 0.75, -2.61, -2.55);
    wbox(g, W, -0.75, 0.75, 2.05, H, -2.61, -2.55);
    // 北墙（带窗洞 x 0.35..1.35, y 0.95..2.10）
    wbox(g, W, -1.44, 0.35, 0, H, 3.85, 3.91);
    wbox(g, W, 1.35, 1.44, 0, H, 3.85, 3.91);
    wbox(g, W, 0.35, 1.35, 0, 0.95, 3.85, 3.91);
    wbox(g, W, 0.35, 1.35, 2.10, H, 3.85, 3.91);
    // 踢脚线
    wbox(g, 'trim_white', -1.38, -1.365, 0, 0.12, -2.55, 1.9);
    wbox(g, 'trim_white', -1.38, -1.365, 0, 0.12, 2.7, 3.85);
    wbox(g, 'trim_white', 1.365, 1.38, 0, 0.12, -2.55, 3.85);
    wbox(g, 'trim_white', -1.38, 1.38, 0, 0.12, 3.835, 3.85);
    // 天花板
    wbox(g, 'ceiling_white', -1.44, 1.44, H, H + 0.06, -2.61, 3.91);
    add('walls', 'static', g);
  }

  // ------------------------------------------------------------------ 门（开着，铰链在北门柱）
  {
    const hinge = [-1.34, 2.68];
    const g = group(hinge[0], 0, hinge[1], 0.349); // 门板指向 (0.94, -0.34)
    lbox(g, mat('door_white', 0.8, 2), 0.02, 0.80, 0.02, 2.0, -0.02, 0.02);
    for (const [x0, x1] of [[0.10, 0.36], [0.44, 0.70]]) {
      for (const [y0, y1] of [[0.25, 0.85], [1.05, 1.80]]) {
        lbox(g, 'door_panel', x0, x1, y0, y1, -0.024, -0.02);
        lbox(g, 'door_panel', x0, x1, y0, y1, 0.02, 0.024);
      }
    }
    lbox(g, 'metal_black', 0.68, 0.74, 0.94, 0.97, -0.065, 0.065); // 把手
    add('door', 'movable', g, {
      mass_kg: 15,
      interaction: { grab: 'hinge', axis: [0, 1, 0], pivot: [hinge[0], 1.0, hinge[1]], limit: [-0.35, 1.75] },
    });
  }

  // ------------------------------------------------------------------ 北窗（钢琴后面）+ 暖气片
  {
    const g = group(0, 0, 0);
    wbox(g, mat('window_daylight', 1, 1.2), 0.35, 1.35, 0.95, 2.10, 3.875, 3.895);
    const F = 'trim_white';
    wbox(g, F, 0.35, 0.40, 0.95, 2.10, 3.855, 3.885);
    wbox(g, F, 1.30, 1.35, 0.95, 2.10, 3.855, 3.885);
    wbox(g, F, 0.35, 1.35, 0.95, 1.00, 3.855, 3.885);
    wbox(g, F, 0.35, 1.35, 1.50, 1.54, 3.855, 3.885);
    wbox(g, F, 0.83, 0.87, 0.95, 2.10, 3.86, 3.885);
    wbox(g, F, 0.30, 1.40, 0.90, 0.95, 3.74, 3.885); // 窗台
    wbox(g, 'blind_grey', 0.32, 1.38, 1.82, 2.22, 3.79, 3.85); // 罗马帘（半下）
    wbox(g, 'radiator_white', 0.45, 1.25, 0.15, 0.70, 3.74, 3.83); // 暖气片
    add('window', 'static', g);
  }

  // ------------------------------------------------------------------ 南凸窗 + 窗帘（躺椅后面）
  {
    const g = group(0, 0, 0);
    wbox(g, mat('window_daylight', 1.5, 1.3), -0.75, 0.75, 0.75, 2.05, -2.605, -2.585);
    wbox(g, 'net_curtain', -0.78, 0.78, 0.72, 2.08, -2.545, -2.525); // 白纱
    wbox(g, 'trim_white', -0.75, 0.75, 0.70, 0.75, -2.58, -2.45); // 窗台
    wbox(g, 'blind_white', -0.72, 0.72, 1.42, 2.06, -2.53, -2.51); // 白色卷帘（半下）
    // 橄榄绿落地窗帘，两边各一
    wbox(g, mat('curtain_olive', 0.7, 2.4), -1.30, -0.74, 0.05, 2.35, -2.50, -2.34);
    wbox(g, mat('curtain_olive', 0.7, 2.4), 0.74, 1.30, 0.05, 2.35, -2.50, -2.34);
    wbox(g, 'metal_black', -1.32, 1.32, 2.36, 2.39, -2.44, -2.41); // 窗帘杆
    add('window_south', 'static', g);
  }

  // ------------------------------------------------------------------ 壁炉（电视后面）+ 壁炉架上的画和钟
  {
    const g = group(0, 0, 0);
    wbox(g, 'hearth_marble', 0.82, 1.38, 0, 0.03, -1.10, 0.18);
    wbox(g, 'mantel_white', 1.28, 1.38, 1.30, 1.34, -1.12, 0.18); // 壁炉架
    // 架上的两幅印刷画和挂钟
    wbox(g, 'paper_white', 1.325, 1.35, 1.34, 1.62, -1.04, -0.70);
    wbox(g, 'photo_pink', 1.32, 1.325, 1.39, 1.57, -0.99, -0.75);
    wbox(g, 'paper_white', 1.325, 1.35, 1.34, 1.60, -0.12, 0.14);
    wbox(g, 'photo_green', 1.32, 1.325, 1.39, 1.55, -0.07, 0.09);
    xdisc(g, 'clock_orange', 1.335, 1.48, -0.42, 0.125, 0.02);
    add('fireplace', 'static', g);
  }

  // ------------------------------------------------------------------ 凹进搁架（南：桌游；北：书）+ 松木书架
  {
    const g = group(0, 0, 0);
    const S = 'shelf_white';
    // 南凹架 z -1.98..-1.18
    wbox(g, S, 1.60, 1.64, 0, 2.05, -1.98, -1.18); // 背板
    wbox(g, S, 1.38, 1.64, 2.01, 2.05, -1.98, -1.18);
    wbox(g, S, 1.38, 1.64, 0, 2.05, -1.98, -1.96); // 侧壁
    wbox(g, S, 1.38, 1.64, 0, 2.05, -1.20, -1.18);
    wbox(g, S, 1.38, 1.64, 1.35, 1.38, -1.98, -1.18);
    wbox(g, S, 1.38, 1.64, 1.68, 1.71, -1.98, -1.18);
    wbox(g, S, 1.10, 1.38, 0, 0.80, -1.98, -1.18); // 底柜
    wbox(g, 'speaker_black', 1.14, 1.44, 0.80, 1.00, -1.88, -1.66); // 底柜上的小黑音箱
    wbox(g, 'boardgame_red', 1.42, 1.60, 1.82, 1.92, -1.96, -1.68);
    wbox(g, 'boardgame_red', 1.42, 1.60, 1.76, 1.82, -1.94, -1.70);
    wbox(g, 'paper_white', 1.42, 1.60, 1.71, 1.76, -1.96, -1.68);
    wbox(g, 'boardgame_blue', 1.42, 1.60, 1.71, 1.78, -1.66, -1.38);
    wbox(g, 'paper_white', 1.42, 1.60, 1.78, 1.86, -1.64, -1.36);
    wbox(g, 'books_mixed', 1.44, 1.60, 1.38, 1.62, -1.94, -1.60);
    wbox(g, 'paper_white', 1.43, 1.45, 1.40, 1.66, -1.55, -1.28); // 相框
    wbox(g, 'photo_dark', 1.425, 1.43, 1.46, 1.62, -1.51, -1.32);
    wbox(g, 'magazines_pale', 1.15, 1.36, 0.80, 1.10, -1.94, -1.70);
    // 北凹架 z 0.12..1.12
    wbox(g, S, 1.60, 1.64, 0, 2.05, 0.12, 1.12);
    wbox(g, S, 1.38, 1.64, 2.01, 2.05, 0.12, 1.12);
    wbox(g, S, 1.38, 1.64, 0, 2.05, 0.12, 0.14); // 侧壁
    wbox(g, S, 1.38, 1.64, 0, 2.05, 1.10, 1.12);
    wbox(g, S, 1.38, 1.64, 1.20, 1.23, 0.12, 1.12);
    wbox(g, S, 1.38, 1.64, 1.55, 1.58, 0.12, 1.12);
    wbox(g, S, 1.38, 1.64, 1.88, 1.91, 0.12, 1.12);
    wbox(g, S, 1.14, 1.38, 0, 0.85, 0.12, 1.12); // 底柜
    wbox(g, 'books_bright', 1.42, 1.60, 1.23, 1.50, 0.16, 1.08);
    wbox(g, 'books_bright', 1.42, 1.60, 1.58, 1.84, 0.16, 1.08);
    wbox(g, 'books_bright', 1.42, 1.60, 1.91, 2.01, 0.40, 1.08);
    wbox(g, 'paper_white', 1.41, 1.43, 1.58, 1.86, 0.16, 0.42); // 斜靠的印刷画
    wbox(g, 'photo_dark', 1.405, 1.41, 1.66, 1.82, 0.20, 0.38);
    wbox(g, 'speaker_black', 1.16, 1.36, 0.85, 0.94, 0.18, 0.62); // 底柜上的功放
    wbox(g, 'paper_white', 1.13, 1.15, 0.86, 1.18, 0.14, 0.42);
    wbox(g, 'magazines_pale', 1.18, 1.36, 0.85, 1.14, 0.66, 1.10);
    // 松木书架 z 1.18..2.18
    const P = 'pine_bookcase';
    wbox(g, P, 1.08, 1.36, 0, 1.90, 1.18, 1.20);
    wbox(g, P, 1.08, 1.36, 0, 1.90, 2.16, 2.18);
    wbox(g, P, 1.34, 1.36, 0, 1.90, 1.20, 2.16);
    for (const y of [0.04, 0.40, 0.75, 1.10, 1.45, 1.86]) wbox(g, P, 1.08, 1.34, y, y + 0.025, 1.20, 2.16);
    for (const y of [0.065, 0.425, 0.775, 1.125, 1.475]) {
      wbox(g, mat('books_mixed', 0.9, 0.3), 1.12, 1.32, y, y + 0.26, 1.23, 2.13);
    }
    wcyl(g, 'pine_bookcase', 1.22, 1.90, 1.55, 0.02, 0.05, 0.06, 12); // 地球仪底座
    const globe = new THREE.Mesh(new THREE.SphereGeometry(0.10, 20, 14), mat('globe_tan', 0.4, 0.3));
    globe.position.set(1.22, 2.06, 1.55);
    g.add(globe);
    add('alcove_shelves', 'static', g);
  }

  // ------------------------------------------------------------------ 地毯（平铺，按静止处理；L 形咬合，互不重叠）
  {
    const g = group(0, 0, 0);
    const D = (x0, x1, z0, z1) => wbox(g, mat('rug_dark_green', x1 - x0, z1 - z0), x0, x1, -0.03, RUG_TOP, z0, z1);
    D(0.20, 0.88, -1.32, -0.90);
    D(0.00, 0.88, -0.90, 0.45);
    add('rug_dark', 'static', g);
  }
  {
    const g = group(0, 0, 0);
    const L = (x0, x1, z0, z1) => wbox(g, mat('rug_light_blue', x1 - x0, z1 - z0), x0, x1, -0.03, RUG_TOP, z0, z1);
    L(-0.85, 0.20, -1.50, -0.90);
    L(-0.85, 0.00, -0.90, -0.28);
    add('rug_light', 'static', g);
  }
  {
    const g = group(0, 0, 0);
    wbox(g, mat('rug_piano', 1.8, 1.4), -0.85, 0.95, -0.03, RUG_TOP, 2.20, 3.60);
    add('piano_rug', 'static', g);
  }

  // ------------------------------------------------------------------ 灰色转角沙发（西墙）
  {
    const g = group(0, 0, 0);
    const F = 'sofa_grey';
    for (const z of [-1.65, -0.45, 0.75]) {
      wbox(g, 'metal_black', -1.32, -1.26, 0, 0.06, z - 0.03, z + 0.03);
      wbox(g, 'metal_black', -0.68, -0.62, 0, 0.06, z - 0.03, z + 0.03);
    }
    wbox(g, F, -1.36, -0.60, 0.06, 0.26, -1.72, 0.70); // 底座
    wbox(g, F, -1.38, -1.12, 0.26, 0.72, -1.42, 0.45); // 靠背框
    const seats = [[-1.42, -0.78], [-0.76, -0.14], [-0.12, 0.45]];
    for (const [z0, z1] of seats) {
      wbox(g, F, -1.30, -0.64, 0.26, 0.42, z0, z1); // 坐垫
      wbox(g, F, -1.34, -1.08, 0.42, 0.90, z0 + 0.01, z1 - 0.01); // 靠垫
      wbox(g, F, -1.32, -1.12, 0.88, 1.00, z0 + 0.03, z1 - 0.03); // 头枕
    }
    wbox(g, F, -1.36, -0.58, 0.26, 0.60, 0.45, 0.70); // 北端扶手
    wbox(g, F, -1.36, -0.58, 0.26, 0.55, -1.72, -1.42); // 南端转角块
    // 沙发上的东西
    wbox(g, 'jacket_red', -1.28, -0.98, 0.42, 0.66, -1.38, -1.08); // 红靠枕
    wbox(g, 'cushion_floral', -1.26, -1.02, 0.42, 0.72, -1.04, -0.74); // 花靠枕
    wbox(g, 'blanket_dark', -1.28, -0.72, 0.42, 0.50, -0.05, 0.42); // 深色毯子
    wbox(g, 'laptop_silver', -1.20, -0.72, 0.60, 0.625, 0.48, 0.68); // 扶手上的笔记本
    add('sofa', 'static', g);
  }

  // ------------------------------------------------------------------ 沙发南端的边桌（键盘、鼠标）
  {
    const g = group(-1.05, 0, -1.90);
    wbox(g, mat('table_walnut', 0.5, 0.3), -1.30, -0.80, 0.60, 0.64, -2.05, -1.75);
    for (const [x, z] of [[-1.27, -2.02], [-1.27, -1.78], [-0.83, -2.02], [-0.83, -1.78]]) {
      wbox(g, 'metal_black', x - 0.015, x + 0.015, GAP, 0.60, z - 0.015, z + 0.015);
    }
    wbox(g, 'keyboard_black', -1.20, -0.85, 0.64, 0.665, -1.95, -1.80);
    wbox(g, 'laptop_silver', -0.98, -0.88, 0.64, 0.66, -2.02, -1.94); // 鼠标
    wcyl(g, 'glass_clear', -1.24, 0.64, -1.82, 0.03, 0.025, 0.09, 14);
    add('side_table', 'movable', g, { mass_kg: 5 });
  }

  // ------------------------------------------------------------------ 落地灯（沙发北端）
  {
    const g = group(-1.27, 0, 0.82);
    lcyl(g, 'lamp_white', 0, 0, 0, 0.11, 0.11, 1.30, 24);
    add('floor_lamp', 'movable', g, { mass_kg: 4 });
  }

  // ------------------------------------------------------------------ 深蓝躺椅 + 花纹靠垫
  {
    const g = group(-0.10, 0, -1.90);
    const A = 'armchair_blue';
    wbox(g, A, -0.52, 0.32, 0.02, 0.22, -2.38, -1.48); // 底座
    wbox(g, A, -0.52, -0.36, 0.02, 0.52, -2.36, -1.48); // 左扶手
    wbox(g, A, 0.16, 0.32, 0.02, 0.52, -2.36, -1.48); // 右扶手
    wbox(g, A, -0.36, 0.16, 0.22, 0.42, -2.20, -1.44); // 座垫
    wbox(g, A, -0.38, 0.18, 0.40, 0.70, -2.42, -2.18); // 靠背下段
    wbox(g, A, -0.38, 0.18, 0.68, 0.98, -2.50, -2.28); // 靠背上段（略后仰）
    wbox(g, A, -0.34, 0.14, 0.96, 1.05, -2.52, -2.34); // 头枕
    add('armchair', 'movable', g, { mass_kg: 25 });
  }
  {
    const g = group(-0.16, 0.42, -2.12);
    lbox(g, 'cushion_cream', -0.18, 0.18, 0, 0.20, -0.08, 0.08);
    lbox(g, 'cushion_brown', -0.13, -0.03, 0.10, 0.17, 0.08, 0.084);
    lbox(g, 'cushion_brown', 0.05, 0.14, 0.03, 0.10, 0.08, 0.084);
    lbox(g, 'cushion_brown', -0.02, 0.06, 0.13, 0.19, 0.08, 0.084);
    lbox(g, 'cushion_brown', -0.16, -0.09, 0.01, 0.06, 0.08, 0.084);
    add('armchair_cushion', 'movable', g, { mass_kg: 0.6 });
  }

  // ------------------------------------------------------------------ 脚凳（蓝垫 + 弯木框，站在浅地毯上）
  {
    const g = group(-0.33, 0, -0.80, 0.20);
    const B = 'birch_frame';
    for (const z of [-0.24, 0.24]) {
      lbox(g, B, -0.24, -0.19, RUG_TOP + GAP, 0.26, z - 0.02, z + 0.02);
      lbox(g, B, 0.19, 0.24, RUG_TOP + GAP, 0.26, z - 0.02, z + 0.02);
      lbox(g, B, -0.24, 0.24, 0.26, 0.29, z - 0.02, z + 0.02);
    }
    lbox(g, B, -0.21, 0.21, 0.26, 0.29, -0.22, 0.22);
    lbox(g, mat('footstool_blue', 0.5, 0.45), -0.235, 0.235, 0.29, 0.38, -0.26, 0.26);
    add('footstool', 'movable', g, { mass_kg: 5 });
  }

  // ------------------------------------------------------------------ 边几（C 形：木面悬到沙发座上方，黑腿在东侧）和上面的东西
  const TABLE_TOP = 0.52;
  {
    const g = group(-0.72, 0, 0.05);
    wbox(g, mat('table_walnut', 0.45, 0.8), -0.94, -0.50, TABLE_TOP - 0.04, TABLE_TOP, -0.35, 0.45);
    const K = 'metal_black';
    for (const z of [-0.26, 0.42]) {
      wbox(g, K, -0.55, -0.52, GAP, TABLE_TOP - 0.04, z - 0.015, z + 0.015); // 立柱（东侧）
      wbox(g, K, -0.65, -0.52, GAP, 0.03, z - 0.015, z + 0.015); // 地面横档
      wbox(g, K, -0.94, -0.52, TABLE_TOP - 0.06, TABLE_TOP - 0.04, z - 0.015, z + 0.015); // 桌面下横档
    }
    wbox(g, K, -0.55, -0.52, GAP, 0.03, -0.26, 0.42);
    wbox(g, mat('table_walnut', 0.15, 0.7), -0.65, -0.54, 0.12, 0.145, -0.24, 0.40); // 下层搁板
    add('coffee_table', 'movable', g, { mass_kg: 6 });
  }
  {
    const g = group(-0.71, TABLE_TOP + GAP, -0.15);
    lcyl(g, 'steel_bowl', 0, 0, 0, 0.10, 0.06, 0.09, 24);
    lcyl(g, 'steel_inside', 0, 0.088, 0, 0.088, 0.088, 0.002, 24);
    lbox(g, 'chopstick_wood', -0.01, 0.01, 0.05, 0.20, -0.09, -0.07); // 碗里的筷子
    lbox(g, 'chopstick_wood', 0.02, 0.04, 0.05, 0.19, -0.08, -0.06);
    add('bowl', 'movable', g, { mass_kg: 0.4 });
  }
  {
    const g = group(-0.79, TABLE_TOP + GAP, 0.05);
    lcyl(g, 'glass_clear', 0, 0, 0, 0.032, 0.048, 0.15, 20);
    lcyl(g, 'glass_clear', 0, 0.15, 0, 0.034, 0.024, 0.07, 20);
    lcyl(g, 'cap_blue', 0, 0.22, 0, 0.020, 0.020, 0.025, 14);
    add('carafe', 'movable', g, { mass_kg: 0.9 });
  }
  {
    const g = group(-0.64, TABLE_TOP + GAP, 0.20);
    lcyl(g, 'glass_clear', 0, 0, 0, 0.030, 0.034, 0.19, 18);
    lcyl(g, 'cap_blue', 0, 0.19, 0, 0.016, 0.016, 0.022, 12);
    add('bottle', 'movable', g, { mass_kg: 0.5 });
  }
  {
    const g = group(-0.73, TABLE_TOP + GAP, 0.33);
    lcyl(g, 'glass_clear', 0, 0, 0, 0.042, 0.032, 0.10, 20);
    add('wine_glass', 'movable', g, { mass_kg: 0.2 });
  }

  // ------------------------------------------------------------------ 拖鞋（浅地毯上）
  function slipper(id, x, z, rotY, len) {
    const g = group(x, RUG_TOP + GAP, z, rotY);
    const h = len / 2;
    lbox(g, 'slipper_tan', -h, h, 0, 0.022, -0.048, 0.048);
    lbox(g, 'slipper_tan', 0, h, 0.022, 0.070, -0.048, 0.048);
    lbox(g, 'slipper_trim', -0.02, 0.0, 0.022, 0.072, -0.048, 0.048);
    add(id, 'movable', g, { mass_kg: 0.2 });
  }
  slipper('slipper_left', 0.02, -0.88, 0.6, 0.26);
  slipper('slipper_right', 0.22, -0.82, -1.4, 0.24);

  // ------------------------------------------------------------------ 低音炮
  {
    const g = group(0.58, 0, -0.95, 0.12);
    lbox(g, mat('speaker_black', 0.4, 0.45), -0.20, 0.20, GAP, 0.46, -0.20, 0.20);
    add('subwoofer', 'movable', g, { mass_kg: 12 });
  }

  // ------------------------------------------------------------------ 电视柜（玻璃 + 木柱 + 功放）和电视
  {
    const g = group(0, 0, 0);
    wbox(g, 'stand_glass', 0.88, 1.30, 0.04, 0.06, -1.05, 0.15);
    wbox(g, 'stand_glass', 0.88, 1.30, 0.30, 0.32, -1.05, 0.15);
    wbox(g, 'stand_glass', 0.88, 1.30, 0.52, 0.54, -1.05, 0.15);
    for (const x of [0.91, 1.27]) {
      for (const z of [-1.01, 0.11]) wcyl(g, 'post_wood', x, 0.001, z, 0.024, 0.024, 0.54, 14);
    }
    wbox(g, 'receiver_silver', 0.94, 1.24, 0.07, 0.27, -0.72, -0.28); // 功放
    xdisc(g, 'speaker_cone', 0.935, 0.17, -0.58, 0.055);
    xdisc(g, 'speaker_cone', 0.935, 0.17, -0.36, 0.055);
    wbox(g, 'speaker_black', 0.94, 1.20, 0.33, 0.50, -0.62, -0.30); // 中置
    xdisc(g, 'speaker_cone', 0.935, 0.415, -0.46, 0.05);
    add('tv_stand', 'static', g);
  }
  {
    const g = group(1.07, 0.54, -0.44, -0.04);
    lbox(g, 'tv_black', -0.10, 0.10, 0, 0.025, -0.28, 0.28); // 底座
    lbox(g, 'tv_black', -0.02, 0.02, 0.025, 0.08, -0.15, 0.15); // 颈
    lbox(g, mat('tv_black', 1.2, 0.7), -0.03, 0.03, 0.08, 0.76, -0.58, 0.58); // 屏幕
    add('tv', 'movable', g, { mass_kg: 15 });
  }

  // ------------------------------------------------------------------ 支架音箱（右音箱上有一盆兰花）
  function speaker(id, x, z, withFlowers) {
    const g = group(x, 0, z);
    const K = 'speaker_black';
    lbox(g, K, -0.15, 0.15, RUG_TOP + GAP, 0.044, -0.14, 0.14); // 底座（踩在地毯上）
    lbox(g, K, -0.025, 0.025, 0.044, 0.55, -0.07, 0.07); // 立柱
    lbox(g, K, -0.13, 0.13, 0.53, 0.55, -0.11, 0.11); // 托板
    lbox(g, mat(K, 0.4, 0.32), -0.15, 0.15, 0.55, 0.95, -0.16, 0.16); // 箱体
    xdisc(g, 'speaker_cone', -0.152, 0.67, 0, 0.062);
    xdisc(g, 'speaker_cone', -0.152, 0.85, 0, 0.024);
    if (withFlowers) {
      lbox(g, 'post_wood', -0.06, 0.06, 0.95, 0.97, -0.06, 0.06);
      lcyl(g, 'terracotta', 0, 0.97, 0, 0.032, 0.026, 0.04, 12);
      lbox(g, 'orchid_pink', -0.05, 0.05, 1.01, 1.08, -0.05, 0.045);
    }
    add(id, 'movable', g, { mass_kg: 8 });
  }
  speaker('speaker_left', 0.94, -1.28, false);
  speaker('speaker_right', 0.98, 0.38, true);

  // ------------------------------------------------------------------ 陶盆绿植（丝兰）
  {
    const g = group(0.72, 0, 0.80);
    lcyl(g, 'terracotta', 0, GAP, 0, 0.14, 0.12, 0.025, 20); // 托盘
    lcyl(g, 'terracotta', 0, 0.027, 0, 0.115, 0.085, 0.24, 20); // 盆
    lcyl(g, 'soil_dark', 0, 0.267, 0, 0.10, 0.10, 0.005, 20);
    lcyl(g, 'plant_trunk', 0.01, 0.27, 0, 0.020, 0.026, 0.60, 10);
    lcyl(g, 'plant_trunk', -0.05, 0.27, 0.04, 0.014, 0.018, 0.22, 10);
    const crown = (cx, cy, cz, n, len, phase) => {
      for (let i = 0; i < n; i++) {
        const az = phase + (i / n) * Math.PI * 2;
        const el = -0.25 + 1.3 * ((i * 7) % 6) / 5;
        const leaf = new THREE.Mesh(new THREE.BoxGeometry(len, 0.004, 0.034), mat('plant_leaf', len, 0.15));
        const dx = Math.cos(el) * Math.cos(az), dy = Math.sin(el), dz = -Math.cos(el) * Math.sin(az);
        leaf.rotation.set(0, az, el, 'YXZ');
        leaf.position.set(cx + (dx * len) / 2, cy + (dy * len) / 2, cz + (dz * len) / 2);
        g.add(leaf);
      }
    };
    crown(0.01, 0.85, 0, 16, 0.35, 0.2);
    crown(0.01, 0.90, 0, 7, 0.27, 0.7);
    crown(-0.05, 0.48, 0.04, 12, 0.30, 0.0);
    add('plant', 'movable', g, { mass_kg: 4 });
  }

  // ------------------------------------------------------------------ 钢琴旁的纸箱
  {
    const g = group(1.05, 0, 3.35);
    lbox(g, 'cardboard', -0.25, 0.20, GAP, 0.36, -0.30, 0.28);
    lbox(g, 'cardboard', -0.18, 0.16, 0.36, 0.60, -0.22, 0.20);
    add('cardboard_boxes', 'movable', g, { mass_kg: 3 });
  }

  // ------------------------------------------------------------------ 三角钢琴（键盘朝西南，转 28°）
  const PIANO_ROT = -0.489;
  {
    const g = group(0.12, 0, 2.48, PIANO_ROT);
    const B = 'piano_black';
    lbox(g, B, -0.65, 0.65, 0.56, 0.62, 0, 0.20); // 键床
    lbox(g, mat('piano_keys', 1.2, 0.15), -0.55, 0.55, 0.62, 0.66, 0.01, 0.16);
    lbox(g, B, -0.65, -0.56, 0.62, 0.76, 0, 0.20); // 键侧板
    lbox(g, B, 0.56, 0.65, 0.62, 0.76, 0, 0.20);
    lbox(g, B, -0.56, 0.56, 0.66, 0.84, 0.165, 0.20); // 谱架挡板
    // 琴身（低音侧在 +x，高音侧阶梯内收）
    lbox(g, B, -0.65, 0.65, 0.60, 0.88, 0.20, 0.58);
    lbox(g, B, -0.55, 0.65, 0.60, 0.88, 0.58, 0.88);
    lbox(g, B, -0.36, 0.65, 0.60, 0.88, 0.88, 1.05);
    lbox(g, B, -0.10, 0.65, 0.60, 0.88, 1.05, 1.15);
    // 琴盖：前段盖在键床上方，主盖沿琴脊（+x 侧）掀起约 50°
    lbox(g, B, -0.66, 0.66, 0.88, 0.90, 0.20, 0.59);
    const lid = new THREE.Mesh(new THREE.BoxGeometry(1.30, 0.02, 0.95), mat('piano_black', 1.3, 0.95));
    lid.rotation.z = 2.269; // 绕 z 轴，从琴脊向上掀起
    lid.position.set(0.232, 1.398, 0.675);
    g.add(lid);
    // 谱架和摊开的琴谱、琴盖上的书
    lbox(g, B, -0.32, 0.32, 0.90, 1.10, 0.26, 0.275);
    lbox(g, 'paper_white', -0.15, 0.20, 0.90, 1.14, 0.248, 0.258);
    lbox(g, 'books_mixed', 0.32, 0.55, 0.90, 0.96, 0.30, 0.48);
    lbox(g, 'book_green', -0.46, -0.26, 0.90, 0.93, 0.32, 0.47);
    // 腿和踏板
    for (const [x, z] of [[-0.56, 0.28], [0.56, 0.28], [0.28, 1.00]]) {
      lbox(g, B, x - 0.05, x + 0.05, 0.03, 0.60, z - 0.05, z + 0.05);
      lbox(g, 'metal_brass', x - 0.03, x + 0.03, 0.005, 0.03, z - 0.03, z + 0.03);
    }
    lbox(g, B, -0.08, 0.08, 0.07, 0.60, 0.22, 0.27);
    lbox(g, B, -0.12, 0.12, 0.03, 0.09, 0.17, 0.29);
    add('piano', 'static', g);
  }
  {
    const g = group(0.28, 0, 2.14, PIANO_ROT);
    lbox(g, mat('bench_leather', 0.55, 0.34), -0.27, 0.27, 0.42, 0.50, -0.17, 0.17);
    lbox(g, 'piano_black', -0.24, 0.24, 0.36, 0.42, -0.14, 0.14);
    for (const x of [-0.22, 0.22]) {
      for (const z of [-0.12, 0.12]) lbox(g, 'piano_black', x - 0.02, x + 0.02, GAP, 0.36, z - 0.02, z + 0.02);
    }
    add('piano_bench', 'movable', g, { mass_kg: 9 });
  }

  return out;
}

export function environment(THREE) {
  // 照片是柔和的室内暖光；materials.json 里的底色已经按照片里的颜色给，光就保持中性均匀
  const amb = new THREE.AmbientLight(0xffffff, 2.2);
  const hemi = new THREE.HemisphereLight(0xffffff, 0xb0a890, 0.5);
  const key = new THREE.DirectionalLight(0xffffff, 0.5);
  key.position.set(-0.6, 3, 1.2);
  return { background: '#1b1a17', lights: [amb, hemi, key] };
}
