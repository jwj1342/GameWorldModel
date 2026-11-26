// 客厅场景：浅色木地板、灰色转角沙发、蓝色躺椅、黑色三角钢琴、电视音响墙。
// 坐标系：y 朝上，地面 y=0，墙大致沿 x/z。单位米。

export function describe(THREE, ctx) {
  const objects = [];

  function box(parent, mat, x0, x1, y0, y1, z0, z1) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(x1 - x0, y1 - y0, z1 - z0), mat);
    m.position.set((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2);
    parent.add(m);
    return m;
  }
  function cyl(parent, mat, r, h, cx, cy, cz, rTop) {
    const m = new THREE.Mesh(new THREE.CylinderGeometry(rTop == null ? r : rTop, r, h, 20), mat);
    m.position.set(cx, cy, cz);
    parent.add(m);
    return m;
  }
  function group(x, y, z, ry) {
    const g = new THREE.Group();
    g.position.set(x, y, z);
    if (ry) g.rotation.y = ry;
    return g;
  }
  function add(id, kind, object3D, extra) {
    objects.push(Object.assign({ id, kind, object3D }, extra || {}));
  }

  // ---------- 材质 ----------
  const pine = ctx.material('pine_floor', { repeat: [3, 6] });
  const wallMat = ctx.material('wall_cream', { repeat: [3, 2] });
  const ceilMat = ctx.material('ceiling_white', { repeat: [3, 3], emissive: '#8f8a7c', emissiveIntensity: 0.6 });
  const curtainMat = ctx.material('curtain_olive', { repeat: [2, 2] });
  const sofaMat = ctx.material('sofa_fabric', { repeat: [2, 2] });
  const armchairMat = ctx.material('armchair_fabric', { repeat: [1, 1] });
  const crochetMat = ctx.material('cushion_crochet', { repeat: [1, 1] });
  const footFabric = ctx.material('footstool_fabric', { repeat: [1, 1] });
  const birch = ctx.material('birch_wood', { repeat: [1, 1] });
  const walnut = ctx.material('walnut_top', { repeat: [1, 1] });
  const blackMetal = ctx.material('black_metal', { repeat: [1, 1] });
  const steel = ctx.material('steel', { repeat: [1, 1] });
  const steelInner = ctx.material('steel_inner', { repeat: [1, 1] });
  const glass = ctx.material('glass_clear', { repeat: [1, 1], transparent: true, opacity: 0.35 });
  const water = ctx.material('water', { repeat: [1, 1], transparent: true, opacity: 0.55 });
  const suede = ctx.material('slipper_suede', { repeat: [1, 1] });
  const rugTeal = ctx.material('rug_teal', { repeat: [2, 2] });
  const rugPale = ctx.material('rug_pale', { repeat: [2, 2] });
  const pianoBlack = ctx.material('piano_black', { repeat: [1, 1] });
  const pianoRugMat = ctx.material('piano_rug_pattern', { repeat: [2, 2] });
  const speakerBlack = ctx.material('speaker_black', { repeat: [1, 1] });
  const tvScreen = ctx.material('tv_screen', { repeat: [1, 1] });
  const tvGlass = ctx.material('tv_glass_shelf', { repeat: [1, 1], transparent: true, opacity: 0.55 });
  const terracotta = ctx.material('terracotta', { repeat: [1, 1] });
  const leafMat = ctx.material('plant_leaf', { repeat: [1, 1] });
  const trunkMat = ctx.material('plant_trunk', { repeat: [1, 1] });
  const soil = ctx.material('soil_dark', { repeat: [1, 1] });
  const doorWhite = ctx.material('door_white', { repeat: [1, 2] });
  const shelfWhite = ctx.material('shelf_white', { repeat: [1, 1] });
  const oak = ctx.material('oak_bookcase', { repeat: [1, 2] });
  const books = ctx.material('books_multi', { repeat: [3, 1] });
  const games = ctx.material('games_boxes', { repeat: [1, 1] });
  const ivory = ctx.material('keys_ivory', { repeat: [4, 1] });
  const radiatorMat = ctx.material('radiator_white', { repeat: [2, 1] });
  const winGlass = ctx.material('window_glass', { repeat: [1, 1], emissive: '#aebec6', emissiveIntensity: 0.7 });
  const winGlass2 = ctx.material('window_glass', { repeat: [1, 1], emissive: '#d5dee2', emissiveIntensity: 1.0 });
  const blind = ctx.material('blind_grey', { repeat: [1, 1] });
  const hallDark = ctx.material('hallway_dark', { repeat: [1, 1] });
  const sheetMusic = ctx.material('sheet_music', { repeat: [1, 1] });
  const globeMat = ctx.material('globe_tan', { repeat: [1, 1] });
  const pillowRed = ctx.material('pillow_red', { repeat: [1, 1] });
  const pillowFloral = ctx.material('pillow_floral', { repeat: [1, 1] });
  const beech = ctx.material('side_table_beech', { repeat: [1, 1] });
  const keyboardMat = ctx.material('keyboard_black', { repeat: [1, 1] });

  // ---------- 地面 ----------
  {
    const g = group(0, 0, 0);
    box(g, pine, -1.88, 1.7, -0.06, 0, -2.5, 3.28);
    add('floor', 'static', g);
  }

  // ---------- 墙、天花板 ----------
  {
    const g = group(0, 0, 0);
    // 左墙（沙发一侧）
    box(g, wallMat, -1.88, -1.8, 0, 2.4, -2.54, 2.8);
    // 扶手椅一端的墙（z=-2.5），中间开窗
    box(g, wallMat, -1.88, 1.33, 0, 0.95, -2.54, -2.46);
    box(g, wallMat, -1.88, 1.33, 2.05, 2.4, -2.54, -2.46);
    box(g, wallMat, -1.88, -0.55, 0.95, 2.05, -2.54, -2.46);
    box(g, wallMat, 0.35, 1.33, 0.95, 2.05, -2.54, -2.46);
    // 右墙（x=1.25）：壁龛和书橱之间的墙垛
    box(g, wallMat, 1.25, 1.33, 0, 2.4, -2.46, -2.05);
    box(g, wallMat, 1.25, 1.33, 0, 0.85, -2.05, -1.2);        // 壁龛1 下段
    box(g, wallMat, 1.6, 1.68, 0.85, 2.15, -2.05, -1.2);      // 壁龛1 后壁
    box(g, wallMat, 1.25, 1.33, 2.15, 2.4, -2.05, -1.2);
    box(g, wallMat, 1.25, 1.33, 0, 2.4, -1.2, 0.18);          // 电视后的墙垛
    box(g, wallMat, 1.25, 1.33, 0, 0.25, 0.18, 1.12);         // 壁龛2 下段
    box(g, wallMat, 1.6, 1.68, 0.25, 2.15, 0.18, 1.12);       // 壁龛2 后壁
    box(g, wallMat, 1.25, 1.33, 2.15, 2.4, 0.18, 1.12);
    box(g, wallMat, 1.25, 1.33, 0, 2.4, 1.12, 1.3);
    box(g, wallMat, 1.66, 1.74, 0, 2.4, 1.3, 2.05);           // 书橱凹位后壁（全高）
    box(g, wallMat, 1.25, 1.33, 0, 2.4, 2.05, 3.28);
    // 钢琴一端的墙（z=3.2），开窗
    box(g, wallMat, -0.62, 0.35, 0, 2.4, 3.2, 3.28);
    box(g, wallMat, 1.2, 1.33, 0, 2.4, 3.2, 3.28);
    box(g, wallMat, 0.35, 1.2, 0, 1.05, 3.2, 3.28);
    box(g, wallMat, 0.35, 1.2, 2.1, 2.4, 3.2, 3.28);
    // 门所在的墙（z=2.72），门洞 x[-1.6,-0.62]
    box(g, wallMat, -1.88, -1.6, 0, 2.4, 2.72, 2.8);
    box(g, wallMat, -1.6, -0.62, 2.02, 2.4, 2.72, 2.8);
    // 门厅侧壁（x=-0.66，从门墙到端墙再延伸进走廊）
    box(g, wallMat, -0.7, -0.62, 0, 2.4, 2.8, 3.28);
    box(g, hallDark, -0.7, -0.62, 0, 2.1, 3.28, 3.9);
    // 电视上方墙上的画框和圆钟
    box(g, shelfWhite, 1.22, 1.25, 1.5, 1.86, -0.95, -0.55);
    box(g, pillowFloral, 1.215, 1.225, 1.54, 1.82, -0.91, -0.59);
    box(g, shelfWhite, 1.22, 1.25, 1.55, 1.85, -0.35, -0.02);
    box(g, games, 1.215, 1.225, 1.59, 1.81, -0.31, -0.06);
    // 天花板
    box(g, ceilMat, -1.88, 1.33, 2.4, 2.48, -2.54, 3.28);
    add('walls', 'static', g);
  }

  // ---------- 走廊（门后的暗空间） ----------
  {
    const g = group(0, 0, 0);
    box(g, hallDark, -1.88, -0.66, -0.04, 0.004, 2.8, 3.9);   // 地
    box(g, hallDark, -1.88, -0.66, 2.02, 2.1, 2.8, 3.9);      // 顶
    box(g, hallDark, -1.88, -1.8, 0, 2.1, 2.8, 3.9);          // 左壁
    box(g, hallDark, -1.88, -0.66, 0, 2.1, 3.82, 3.9);        // 尽头
    add('hallway', 'static', g);
  }

  // ---------- 门（白色，开着约 49°，绕铰链转） ----------
  {
    const g = group(-0.931, 1.01, 2.362, -0.855);
    box(g, doorWhite, -0.475, 0.475, -1.01, 1.01, -0.02, 0.02);
    // 四块门板线框
    box(g, doorWhite, -0.38, -0.06, 0.15, 0.85, -0.028, 0.028);
    box(g, doorWhite, 0.06, 0.38, 0.15, 0.85, -0.028, 0.028);
    box(g, doorWhite, -0.38, -0.06, -0.85, -0.15, -0.028, 0.028);
    box(g, doorWhite, 0.06, 0.38, -0.85, -0.15, -0.028, 0.028);
    // 把手
    box(g, steel, 0.3, 0.44, -0.015, 0.015, 0.028, 0.06);
    box(g, steel, 0.3, 0.44, -0.015, 0.015, -0.06, -0.028);
    add('door', 'movable', g, {
      mass_kg: 20,
      interaction: { grab: 'hinge', axis: [0, 1, 0], pivot: [-0.62, 0, 2.72], limit: [-0.6, 0.8] },
    });
  }

  // ---------- 窗（钢琴后面，端墙上） ----------
  {
    const g = group(0, 0, 0);
    box(g, doorWhite, 0.33, 0.39, 1.03, 2.12, 3.16, 3.26);
    box(g, doorWhite, 1.16, 1.22, 1.03, 2.12, 3.16, 3.26);
    box(g, doorWhite, 0.33, 1.22, 2.06, 2.12, 3.16, 3.26);
    box(g, doorWhite, 0.33, 1.22, 1.03, 1.09, 3.16, 3.26);
    box(g, doorWhite, 0.755, 0.785, 1.09, 2.06, 3.18, 3.24);
    box(g, doorWhite, 0.39, 1.16, 1.55, 1.58, 3.18, 3.24);
    box(g, winGlass, 0.39, 1.16, 1.09, 2.06, 3.23, 3.24);
    box(g, doorWhite, 0.28, 1.27, 0.98, 1.05, 3.08, 3.28);    // 窗台
    box(g, blind, 0.31, 1.24, 1.92, 2.18, 3.1, 3.17);         // 罗马帘
    add('window', 'static', g);
  }

  // ---------- 扶手椅那端的窗和窗帘 ----------
  {
    const g = group(0, 0, 0);
    box(g, doorWhite, -0.57, -0.51, 0.93, 2.06, -2.52, -2.44);
    box(g, doorWhite, 0.31, 0.37, 0.93, 2.06, -2.52, -2.44);
    box(g, doorWhite, -0.57, 0.37, 2.0, 2.06, -2.52, -2.44);
    box(g, doorWhite, -0.57, 0.37, 0.93, 0.99, -2.52, -2.44);
    box(g, winGlass2, -0.51, 0.31, 0.99, 2.0, -2.49, -2.48);
    box(g, doorWhite, -0.6, 0.4, 0.88, 0.95, -2.5, -2.38);
    add('window2', 'static', g);
  }
  {
    const g = group(0, 0, 0);
    // 左右两幅落地窗帘，各分三片做出褶皱
    box(g, curtainMat, -1.35, -1.07, 0.03, 2.2, -2.44, -2.32);
    box(g, curtainMat, -1.07, -0.79, 0.03, 2.2, -2.42, -2.3);
    box(g, curtainMat, -0.79, -0.52, 0.03, 2.2, -2.44, -2.32);
    box(g, curtainMat, 0.32, 0.6, 0.03, 2.2, -2.44, -2.32);
    box(g, curtainMat, 0.6, 0.88, 0.03, 2.2, -2.42, -2.3);
    box(g, curtainMat, 0.88, 1.15, 0.03, 2.2, -2.44, -2.32);
    add('curtains', 'static', g);
  }

  // ---------- 暖气片（钢琴窗下） ----------
  {
    const g = group(0, 0, 0);
    box(g, radiatorMat, 0.5, 1.15, 0.18, 0.66, 3.06, 3.18);
    box(g, radiatorMat, 0.48, 1.17, 0.66, 0.7, 3.05, 3.19);
    add('radiator', 'static', g);
  }

  // ---------- 壁龛搁架和书架 ----------
  {
    const g = group(0, 0, 0);
    // 壁龛1（桌游）：y 0.85..2.15, z -2.05..-1.2
    for (const y of [1.02, 1.32, 1.62, 1.92]) {
      box(g, shelfWhite, 1.26, 1.6, y, y + 0.03, -2.03, -1.22);
    }
    box(g, games, 1.3, 1.58, 1.95, 2.02, -1.95, -1.6);
    box(g, games, 1.32, 1.56, 2.02, 2.09, -1.92, -1.62);
    box(g, games, 1.3, 1.56, 1.65, 1.72, -1.9, -1.58);
    box(g, shelfWhite, 1.38, 1.56, 1.05, 1.3, -1.6, -1.56);   // 相框
    box(g, books, 1.32, 1.56, 1.35, 1.55, -1.85, -1.4);
    // 壁龛2（白色书架）：y 0.25..2.15, z 0.18..1.12
    for (const y of [0.32, 0.67, 1.02, 1.37, 1.72]) {
      box(g, shelfWhite, 1.26, 1.6, y, y + 0.03, 0.2, 1.1);
      box(g, books, 1.3, 1.56, y + 0.03, y + 0.29, 0.24, 1.06);
    }
    // 木书架（凹位 z 1.3..2.05）
    box(g, oak, 1.28, 1.64, 0.005, 2.0, 1.3, 1.34);
    box(g, oak, 1.28, 1.64, 0.005, 2.0, 2.01, 2.05);
    box(g, oak, 1.61, 1.64, 0.005, 2.0, 1.3, 2.05);
    box(g, oak, 1.28, 1.64, 1.97, 2.0, 1.3, 2.05);
    for (const y of [0.02, 0.36, 0.7, 1.04, 1.38, 1.72]) {
      box(g, oak, 1.28, 1.61, y, y + 0.03, 1.34, 2.01);
      if (y < 1.7) box(g, books, 1.32, 1.58, y + 0.03, y + 0.29, 1.37, 1.98);
    }
    // 书架顶上的地球仪
    box(g, blackMetal, 1.39, 1.49, 2.0, 2.05, 1.4, 1.5);
    const globe = new THREE.Mesh(new THREE.SphereGeometry(0.115, 20, 14), globeMat);
    globe.position.set(1.44, 2.16, 1.45);
    g.add(globe);
    add('alcove_shelves', 'static', g);
  }

  // ---------- 沙发（灰色转角，含靠枕） ----------
  {
    const g = group(-1.4, 0, -0.1);
    box(g, sofaMat, -0.38, 0.38, 0.005, 0.28, -1.5, 1.45);     // 底座
    box(g, sofaMat, -0.36, 0.42, 0.28, 0.42, -1.48, -0.52);    // 坐垫 ×3
    box(g, sofaMat, -0.36, 0.42, 0.28, 0.42, -0.5, 0.46);
    box(g, sofaMat, -0.36, 0.42, 0.28, 0.42, 0.48, 1.43);
    box(g, sofaMat, -0.38, -0.12, 0.28, 0.92, -1.5, 1.45);     // 靠背
    box(g, sofaMat, -0.16, 0.04, 0.42, 0.9, -1.46, -0.54);     // 靠垫 ×3
    box(g, sofaMat, -0.16, 0.04, 0.42, 0.9, -0.5, 0.44);
    box(g, sofaMat, -0.16, 0.04, 0.42, 0.9, 0.48, 1.41);
    box(g, sofaMat, -0.38, 0.42, 0.28, 0.62, -1.7, -1.5);      // 两端扶手
    box(g, sofaMat, -0.38, 0.42, 0.28, 0.62, 1.45, 1.6);
    box(g, sofaMat, 0.38, 0.78, 0.018, 0.4, -1.25, 0.7);       // 转角榻
    const p1 = box(g, pillowRed, -0.18, 0.18, -0.16, 0.16, -0.07, 0.07);
    p1.position.set(0.1, 0.6, -1.32); p1.rotation.set(0.1, 0.3, 0.15);
    const p2 = box(g, pillowFloral, -0.18, 0.18, -0.15, 0.15, -0.065, 0.065);
    p2.position.set(0.08, 0.55, -0.7); p2.rotation.set(-0.05, -0.2, -0.12);
    add('sofa', 'static', g);
  }

  // ---------- 沙发尽头的边桌和键盘架 ----------
  {
    const g = group(-1.37, 0, -2.0);
    box(g, beech, -0.225, 0.225, 0.5, 0.54, -0.15, 0.15);
    for (const sx of [-0.2, 0.2]) for (const sz of [-0.12, 0.12]) {
      box(g, beech, sx - 0.015, sx + 0.015, 0.005, 0.5, sz - 0.015, sz + 0.015);
    }
    add('side_table', 'movable', g, { mass_kg: 6 });
  }
  {
    const g = group(-0.88, 0, -1.52, 1.72);
    box(g, keyboardMat, -0.21, 0.21, 0.47, 0.495, -0.075, 0.075);
    for (const sx of [-0.15, 0.15]) for (const sz of [-0.05, 0.05]) {
      box(g, blackMetal, sx - 0.01, sx + 0.01, 0.005, 0.47, sz - 0.01, sz + 0.01);
    }
    add('keyboard', 'movable', g, { mass_kg: 2 });
  }

  // ---------- 躺椅和靠垫 ----------
  {
    const g = group(-0.19, 0, -1.89);
    box(g, armchairMat, -0.43, 0.43, 0.005, 0.42, -0.39, 0.39);
    box(g, armchairMat, -0.31, 0.31, 0.42, 0.52, -0.12, 0.39);
    box(g, armchairMat, -0.43, 0.43, 0.42, 1.0, -0.39, -0.17);
    box(g, armchairMat, -0.33, 0.33, 0.5, 0.92, -0.25, -0.05);
    box(g, armchairMat, -0.3, 0.3, 0.9, 1.04, -0.32, -0.1);
    box(g, armchairMat, -0.43, -0.31, 0.42, 0.64, -0.3, 0.32);
    box(g, armchairMat, 0.31, 0.43, 0.42, 0.64, -0.3, 0.32);
    add('armchair', 'movable', g, { mass_kg: 45 });
  }
  {
    const g = group(-0.19, 0.725, -1.82);
    g.rotation.x = -0.12;
    box(g, crochetMat, -0.21, 0.21, -0.19, 0.19, -0.08, 0.08);
    add('armchair_cushion', 'movable', g, { mass_kg: 0.6 });
  }

  // ---------- 脚凳 ----------
  {
    const g = group(-0.37, 0, -0.85, 0.12);
    box(g, footFabric, -0.24, 0.24, 0.26, 0.41, -0.24, 0.24);
    for (const sx of [-0.22, 0.22]) {
      box(g, birch, sx - 0.015, sx + 0.015, 0.032, 0.052, -0.28, 0.3);
      const f = box(g, birch, sx - 0.015, sx + 0.015, -0.16, 0.16, -0.0125, 0.0125);
      f.position.set(sx, 0.18, 0.26); f.rotation.x = 0.5;
      const b = box(g, birch, sx - 0.015, sx + 0.015, -0.15, 0.15, -0.0125, 0.0125);
      b.position.set(sx, 0.18, -0.22); b.rotation.x = -0.45;
    }
    box(g, birch, -0.22, 0.22, 0.24, 0.26, -0.2, -0.17);
    box(g, birch, -0.22, 0.22, 0.24, 0.26, 0.17, 0.2);
    add('footstool', 'movable', g, { mass_kg: 5 });
  }

  // ---------- 边几（C 形，木面黑腿） ----------
  {
    const g = group(-0.7, 0, -0.175);
    box(g, walnut, -0.3, 0.29, 0.43, 0.465, -0.375, 0.375);
    box(g, blackMetal, 0.26, 0.29, 0.045, 0.43, -0.355, -0.315);
    box(g, blackMetal, 0.26, 0.29, 0.045, 0.43, 0.315, 0.355);
    box(g, blackMetal, 0.26, 0.44, 0.032, 0.052, -0.355, -0.315);
    box(g, blackMetal, 0.26, 0.44, 0.032, 0.052, 0.315, 0.355);
    add('coffee_table', 'movable', g, { mass_kg: 9 });
  }

  // ---------- 边几上的东西 ----------
  {
    const g = group(-0.72, 0.467, -0.2);
    cyl(g, steel, 0.11, 0.085, 0, 0.0425, 0);
    cyl(g, steelInner, 0.095, 0.004, 0, 0.078, 0);
    add('bowl', 'movable', g, { mass_kg: 0.6 });
  }
  {
    const g = group(-0.56, 0.467, 0.0);
    cyl(g, glass, 0.052, 0.15, 0, 0.075, 0);
    cyl(g, water, 0.046, 0.09, 0, 0.05, 0);
    cyl(g, glass, 0.022, 0.07, 0, 0.185, 0);
    add('carafe', 'movable', g, { mass_kg: 0.8 });
  }
  {
    const g = group(-0.88, 0.467, 0.05);
    cyl(g, glass, 0.032, 0.006, 0, 0.003, 0);
    cyl(g, glass, 0.005, 0.06, 0, 0.036, 0);
    cyl(g, glass, 0.034, 0.075, 0, 0.1035, 0, 0.028);
    add('wine_glass', 'movable', g, { mass_kg: 0.3 });
  }
  {
    const g = group(-0.62, 0.467, -0.38);
    cyl(g, glass, 0.033, 0.095, 0, 0.0475, 0);
    cyl(g, water, 0.029, 0.05, 0, 0.03, 0);
    add('glass_cup', 'movable', g, { mass_kg: 0.3 });
  }

  // ---------- 拖鞋 ----------
  function slipper(id, x, z, ry) {
    const g = group(x, 0.032, z, ry);
    box(g, suede, -0.055, 0.055, 0, 0.022, -0.135, 0.135);
    box(g, suede, -0.052, 0.052, 0.022, 0.075, -0.13, -0.005);
    add(id, 'movable', g, { mass_kg: 0.3 });
  }
  slipper('slipper_left', -0.04, -0.95, 0.18);
  slipper('slipper_right', 0.09, -0.8, -0.12);

  // ---------- 地毯 ----------
  {
    const g = group(0, 0, 0);
    box(g, rugTeal, -0.82, 0.58, 0.002, 0.016, -0.7, 0.52);
    box(g, rugTeal, -0.82, 0.2, 0.002, 0.016, -1.38, -0.7);
    add('rug_dark', 'movable', g, { mass_kg: 6 });
  }
  {
    const g = group(0, 0, 0);
    box(g, rugPale, -0.65, 0.15, 0.017, 0.031, -1.34, -0.12);
    add('rug_light', 'movable', g, { mass_kg: 4 });
  }
  {
    const g = group(0, 0, 2.5);
    box(g, pianoRugMat, -0.85, 0.85, 0.002, 0.014, -0.6, 0.65);
    add('piano_rug', 'static', g);
  }

  // ---------- 低音炮 ----------
  {
    const g = group(0.53, 0, -1.06);
    box(g, speakerBlack, -0.21, 0.21, 0.002, 0.42, -0.22, 0.22);
    box(g, blackMetal, -0.215, -0.205, 0.03, 0.39, -0.19, 0.19);
    add('subwoofer', 'movable', g, { mass_kg: 16 });
  }

  // ---------- 电视柜和电视 ----------
  {
    const g = group(1.03, 0, -0.55);
    box(g, tvGlass, -0.21, 0.21, 0.52, 0.55, -0.85, 0.85);
    box(g, tvGlass, -0.19, 0.19, 0.27, 0.295, -0.8, 0.8);
    box(g, tvGlass, -0.2, 0.2, 0.05, 0.08, -0.82, 0.82);
    for (const sx of [-0.17, 0.17]) for (const sz of [-0.78, 0.78]) {
      cyl(g, blackMetal, 0.022, 0.52, sx, 0.26, sz);
    }
    box(g, speakerBlack, -0.15, 0.15, 0.08, 0.24, -0.28, 0.28);   // 中置音箱
    box(g, blackMetal, -0.155, -0.15, 0.09, 0.23, -0.26, 0.26);
    box(g, speakerBlack, -0.16, 0.16, 0.295, 0.42, -0.62, -0.15); // 功放
    box(g, blackMetal, -0.14, 0.14, 0.295, 0.38, 0.05, 0.5);      // 游戏机
    add('tv_stand', 'movable', g, { mass_kg: 32 });
  }
  {
    const g = group(1.02, 0, -0.52);
    box(g, tvScreen, -0.035, 0.035, 0.575, 1.38, -0.72, 0.72);
    box(g, speakerBlack, 0.035, 0.085, 0.62, 1.32, -0.62, 0.62);
    box(g, blackMetal, -0.03, 0.1, 0.552, 0.58, -0.47, -0.41);
    box(g, blackMetal, -0.03, 0.1, 0.552, 0.58, 0.41, 0.47);
    add('tv', 'movable', g, { mass_kg: 24 });
  }

  // ---------- 落地音箱（含支架） ----------
  function speaker(id, x, z, ry) {
    const g = group(x, 0, z, ry);
    box(g, blackMetal, -0.13, 0.13, 0.002, 0.025, -0.13, 0.13);
    box(g, blackMetal, -0.075, -0.045, 0.025, 0.63, -0.015, 0.015);
    box(g, blackMetal, 0.045, 0.075, 0.025, 0.63, -0.015, 0.015);
    box(g, blackMetal, -0.11, 0.11, 0.63, 0.65, -0.11, 0.11);
    box(g, speakerBlack, -0.13, 0.13, 0.65, 1.03, -0.12, 0.12);
    const w = cyl(g, steelInner, 0.068, 0.012, 0, 0.8, 0);
    w.rotation.z = Math.PI / 2; w.position.set(-0.132, 0.8, 0);
    const t = cyl(g, steelInner, 0.028, 0.012, 0, 0.94, 0);
    t.rotation.z = Math.PI / 2; t.position.set(-0.132, 0.94, 0);
    add(id, 'movable', g, { mass_kg: 11 });
  }
  speaker('speaker_left', 1.0, -1.55, 0.15);
  speaker('speaker_right', 0.92, 0.42, -0.15);

  // ---------- 黑架子（书橱前的收纳格） ----------
  {
    const g = group(0, 0, 0);
    box(g, speakerBlack, 0.8, 1.18, 0.005, 1.0, 1.55, 2.05);
    box(g, blackMetal, 0.79, 0.8, 0.05, 0.96, 1.59, 2.01);
    add('storage_shelf', 'static', g);
  }

  // ---------- 绿植 ----------
  {
    const g = group(0.62, 0, 0.88);
    cyl(g, terracotta, 0.19, 0.025, 0, 0.0145, 0);
    cyl(g, terracotta, 0.155, 0.26, 0, 0.157, 0, 0.135);
    cyl(g, terracotta, 0.17, 0.05, 0, 0.287, 0);
    cyl(g, soil, 0.145, 0.01, 0, 0.3, 0);
    cyl(g, trunkMat, 0.028, 0.55, 0, 0.575, 0);
    for (let k = 0; k < 9; k++) {
      const leaf = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.55, 0.012), leafMat);
      leaf.geometry.translate(0, 0.27, 0);
      leaf.position.set(0, 0.66, 0);
      leaf.rotation.order = 'YXZ';
      leaf.rotation.y = k * 0.7;
      leaf.rotation.x = 0.22 + 0.13 * (k % 3);
      g.add(leaf);
    }
    add('plant', 'movable', g, { mass_kg: 14 });
  }

  // ---------- 三角钢琴 ----------
  {
    const g = group(-0.12, 0, 2.75, -0.15);
    box(g, pianoBlack, -0.7, 0.5, 0.66, 1.0, -0.28, 0.4);      // 琴身
    box(g, pianoBlack, -0.68, 0.68, 0.64, 0.72, -0.5, -0.28);  // 键盘床
    box(g, ivory, -0.62, 0.62, 0.72, 0.735, -0.47, -0.31);     // 琴键
    box(g, pianoBlack, -0.62, 0.62, 0.735, 0.79, -0.31, -0.27);
    for (const [lx, lz] of [[-0.6, -0.36], [0.6, -0.36], [-0.05, 0.3]]) {
      cyl(g, pianoBlack, 0.045, 0.645, lx, 0.3375, lz);
    }
    box(g, blackMetal, -0.1, 0.1, 0.12, 0.16, -0.4, -0.3);     // 踏板
    box(g, blackMetal, -0.06, -0.045, 0.16, 0.64, -0.36, -0.345);
    box(g, blackMetal, 0.045, 0.06, 0.16, 0.64, -0.36, -0.345);
    const desk = box(g, pianoBlack, -0.35, 0.35, -0.135, 0.135, -0.015, 0.015);
    desk.position.set(0, 1.13, 0.03); desk.rotation.x = -0.18;
    const sheets = box(g, sheetMusic, -0.28, 0.28, -0.13, 0.13, -0.004, 0.004);
    sheets.position.set(0, 1.15, 0.005); sheets.rotation.x = -0.18;
    box(g, books, -0.5, -0.2, 1.0, 1.045, 0.05, 0.3);          // 琴盖上的书
    box(g, games, -0.46, -0.24, 1.045, 1.08, 0.08, 0.28);
    add('piano', 'static', g);
  }
  {
    const g = group(0.3, 0, 2.3, -0.15);
    box(g, pianoBlack, -0.27, 0.27, 0.44, 0.5, -0.16, 0.16);
    for (const sx of [-0.23, 0.23]) for (const sz of [-0.12, 0.12]) {
      box(g, pianoBlack, sx - 0.0175, sx + 0.0175, 0.015, 0.44, sz - 0.0175, sz + 0.0175);
    }
    add('piano_bench', 'movable', g, { mass_kg: 11 });
  }

  return objects;
}

export function environment(THREE) {
  const hemi = new THREE.HemisphereLight(0xfff2dd, 0x8a7a66, 1.3);
  const key = new THREE.DirectionalLight(0xffe6c0, 1.1);
  key.position.set(1.5, 3.2, 6);
  const fill = new THREE.DirectionalLight(0xdfe8f0, 0.5);
  fill.position.set(-0.5, 2.5, -6);
  return { background: '#cfc9bc', lights: [hemi, key, fill] };
}
