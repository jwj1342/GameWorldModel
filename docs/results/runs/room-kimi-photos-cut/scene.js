// 客厅场景：从 24 张照片 + 相机位姿重建。
// 坐标系：y 上，地面 y=0，大致米。房间：x ∈ [-1.38, 1.33]，z ∈ [-2.32, 2.50]。
// -z 墙：扶手椅窗（带窗帘）；+z 墙：钢琴上方的窗 + 敞开的门；
// +x 墙：两个凹架（桌游 / 书）+ 电视； -x 墙：转角沙发。

export function describe(THREE, ctx) {
  const items = [];

  function box(w, h, d, mat, x = 0, y = 0, z = 0, rot = null) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
    m.position.set(x, y, z);
    if (rot) m.rotation.set(rot[0] || 0, rot[1] || 0, rot[2] || 0);
    return m;
  }
  function cyl(rt, rb, h, mat, x = 0, y = 0, z = 0, seg = 20, rot = null) {
    const m = new THREE.Mesh(new THREE.CylinderGeometry(rt, rb, h, seg), mat);
    m.position.set(x, y, z);
    if (rot) m.rotation.set(rot[0] || 0, rot[1] || 0, rot[2] || 0);
    return m;
  }
  function grp(id, kind, children, pos, ry = 0, extra = {}) {
    const g = new THREE.Group();
    for (const c of children) g.add(c);
    g.position.set(pos[0], pos[1], pos[2]);
    if (ry) g.rotation.y = ry;
    items.push({ id, kind, object3D: g, ...extra });
    return g;
  }

  // ---------- 材质 ----------
  const pine = ctx.material('pine_floor', { repeat: [3, 5] });
  const wallMat = ctx.material('wall_cream', { repeat: [3, 2] });
  const ceilMat = ctx.material('ceiling_white', { repeat: [3, 3] });
  const doorMat = ctx.material('door_white', { repeat: [1, 2] });
  const glassMat = ctx.material('window_glass', { transparent: true, opacity: 0.45, roughness: 0.1 });
  const sheerMat = ctx.material('curtain_sheer', { transparent: true, opacity: 0.65 });
  const curtainMat = ctx.material('curtain_gray', { repeat: [2, 2] });
  const blindMat = ctx.material('blind_gray', { repeat: [2, 1] });
  const rugDarkMat = ctx.material('rug_dark_green', { repeat: [2, 2] });
  const rugLightMat = ctx.material('rug_light_blue', { repeat: [2, 2] });
  const rugPatMat = ctx.material('rug_pattern', { repeat: [2, 2] });
  const sofaMat = ctx.material('sofa_gray', { repeat: [2, 2] });
  const armchairMat = ctx.material('armchair_blue', { repeat: [2, 2] });
  const cushionMat = ctx.material('cushion_pattern', { repeat: [1, 1] });
  const footBlue = ctx.material('footstool_blue', { repeat: [1, 1] });
  const bentwood = ctx.material('bentwood', { repeat: [1, 1] });
  const tableWood = ctx.material('table_wood', { repeat: [1, 1] });
  const metalBlack = ctx.material('metal_black', { repeat: [1, 1] });
  const steel = ctx.material('steel_bowl', { repeat: [1, 1] });
  const glassClear = ctx.material('glass_clear', { transparent: true, opacity: 0.35, roughness: 0.05 });
  const pianoBlack = ctx.material('piano_black', { repeat: [1, 1] });
  const keysWhite = ctx.material('keys_white', { repeat: [1, 1] });
  const speakerMat = ctx.material('speaker_black', { repeat: [1, 1] });
  const tvBlack = ctx.material('tv_black', { repeat: [1, 1] });
  const tvScreen = ctx.material('tv_screen', { repeat: [1, 1] });
  const glassShelf = ctx.material('glass_shelf', { transparent: true, opacity: 0.4, roughness: 0.1 });
  const terra = ctx.material('pot_terra', { repeat: [1, 1] });
  const leaf = ctx.material('plant_green', { repeat: [1, 1] });
  const oak = ctx.material('oak_bookcase', { repeat: [1, 2] });
  const books = ctx.material('book_spines', { repeat: [2, 1] });
  const radWhite = ctx.material('radiator_white', { repeat: [2, 1] });
  const lampWhite = ctx.material('lamp_white', { repeat: [1, 1] });
  const hallDark = ctx.material('hallway_dark', { repeat: [2, 2] });
  const tileDark = ctx.material('tile_checker', { repeat: [2, 3] });
  const cardboard = ctx.material('box_cardboard', { repeat: [1, 1] });
  const artMat = ctx.material('art_frame', { repeat: [1, 1] });
  const gameBox = ctx.material('game_boxes', { repeat: [1, 1] });
  const capBlue = ctx.material('cap_blue', { repeat: [1, 1] });

  // ---------- 地面 ----------
  grp('floor', 'static', [
    box(2.71, 0.05, 4.82, pine, 0, -0.025, 0),
  ], [-0.025, 0, 0.09]);

  // ---------- 墙（含天花、窗洞、门洞、凹架外框、挂画） ----------
  const wallParts = [];
  const W = wallMat;
  // -z 墙 z=-2.35（厚 0.06，面在 -2.32）
  wallParts.push(box(0.78, 2.5, 0.06, W, -0.99, 1.25, -2.35));
  wallParts.push(box(0.98, 2.5, 0.06, W, 0.84, 1.25, -2.35));
  wallParts.push(box(0.95, 0.68, 0.06, W, -0.125, 0.34, -2.35));
  wallParts.push(box(0.95, 0.45, 0.06, W, -0.125, 2.275, -2.35));
  // +z 墙 z=2.53（面在 2.50）
  wallParts.push(box(0.08, 2.5, 0.06, W, -1.34, 1.25, 2.53));
  wallParts.push(box(0.84, 2.5, 0.06, W, -0.13, 1.25, 2.53));
  wallParts.push(box(0.33, 2.5, 0.06, W, 1.165, 1.25, 2.53));
  wallParts.push(box(0.71, 1.10, 0.06, W, 0.645, 0.55, 2.53));
  wallParts.push(box(0.71, 0.65, 0.06, W, 0.645, 2.175, 2.53));
  wallParts.push(box(0.75, 0.55, 0.06, W, -0.925, 2.225, 2.53));
  // -x 墙 x=-1.41（面在 -1.38）
  wallParts.push(box(0.06, 2.5, 4.88, W, -1.41, 1.25, 0.09));
  // +x 墙主面 x=1.22（面在 1.20）：分段绕开两个凹架口
  // 凹架口：games z[-2.06,-1.37] y[1.02,2.10]；books z[0.25,1.09] y[0.15,2.10]
  wallParts.push(box(0.06, 2.5, 0.26, W, 1.22, 1.25, -2.19));  // z[-2.32,-2.06]
  wallParts.push(box(0.06, 1.02, 0.69, W, 1.22, 0.51, -1.715));// games 下方 z[-2.06,-1.37]
  wallParts.push(box(0.06, 0.40, 0.69, W, 1.22, 2.30, -1.715));// games 上方 y[2.10,2.50]
  wallParts.push(box(0.06, 2.5, 1.62, W, 1.22, 1.25, -0.56));  // z[-1.37,0.25]（电视墙段）
  wallParts.push(box(0.06, 0.15, 0.84, W, 1.22, 0.075, 0.67)); // books 下方 z[0.25,1.09]
  wallParts.push(box(0.06, 0.40, 0.84, W, 1.22, 2.30, 0.67));  // books 上方
  wallParts.push(box(0.06, 2.5, 1.41, W, 1.22, 1.25, 1.795));  // z[1.09,2.50]
  // 凹架内腔（背板 x=1.345，侧板、顶底板）——桌游
  wallParts.push(box(0.03, 1.08, 0.69, W, 1.345, 1.56, -1.715)); // 背板
  wallParts.push(box(0.15, 1.08, 0.03, W, 1.275, 1.56, -2.045)); // 侧板
  wallParts.push(box(0.15, 1.08, 0.03, W, 1.275, 1.56, -1.385)); // 侧板
  wallParts.push(box(0.15, 0.03, 0.69, W, 1.275, 2.085, -1.715));// 顶板
  // 凹架内腔——书架
  wallParts.push(box(0.03, 1.95, 0.84, W, 1.345, 1.125, 0.67));
  wallParts.push(box(0.15, 1.95, 0.03, W, 1.275, 1.125, 0.265));
  wallParts.push(box(0.15, 1.95, 0.03, W, 1.275, 1.125, 1.075));
  wallParts.push(box(0.15, 0.03, 0.84, W, 1.275, 2.085, 0.67));
  wallParts.push(box(0.15, 0.03, 0.84, W, 1.275, 0.165, 0.67));
  // 天花
  wallParts.push(box(2.77, 0.06, 4.94, ceilMat, -0.025, 2.53, 0.09));
  // 电视上方挂画和钟（+x 墙面 x=1.20 上）
  wallParts.push(box(0.03, 0.42, 0.34, artMat, 1.19, 1.45, -0.80));
  wallParts.push(box(0.03, 0.42, 0.34, artMat, 1.19, 1.42, -0.12));
  wallParts.push(cyl(0.16, 0.16, 0.03, artMat, 1.19, 1.80, -0.46, 24, [0, 0, Math.PI / 2]));
  // 门框（+z 墙门洞一圈，白色）
  wallParts.push(box(0.06, 1.98, 0.10, doorMat, -1.325, 0.99, 2.50));
  wallParts.push(box(0.06, 1.98, 0.10, doorMat, -0.525, 0.99, 2.50));
  wallParts.push(box(0.86, 0.06, 0.10, doorMat, -0.925, 1.98, 2.50));
  grp('walls', 'static', wallParts, [0, 0, 0]);

  // ---------- 门（白色，开约 60°，铰链在靠钢琴一侧 x=-0.55） ----------
  const doorParts = [];
  doorParts.push(box(0.75, 1.90, 0.04, doorMat, 0, 0, 0));
  // 门板线条（四块浅浮雕板）
  doorParts.push(box(0.24, 0.70, 0.012, doorMat, -0.17, 0.42, -0.026));
  doorParts.push(box(0.24, 0.70, 0.012, doorMat, 0.17, 0.42, -0.026));
  doorParts.push(box(0.24, 0.55, 0.012, doorMat, -0.17, -0.50, -0.026));
  doorParts.push(box(0.24, 0.55, 0.012, doorMat, 0.17, -0.50, -0.026));
  doorParts.push(box(0.03, 0.12, 0.03, metalBlack, 0.31, 0, -0.045)); // 把手
  grp('door', 'movable', doorParts, [-0.7375, 0.95, 2.1752], Math.PI * 2 / 3, {
    mass_kg: 12,
    interaction: { grab: 'hinge', axis: [0, 1, 0], pivot: [-0.55, 0.95, 2.50], limit: [-0.9, 1.04] },
  });

  // ---------- 门后走廊（暗） ----------
  grp('hallway', 'static', [
    box(0.98, 0.04, 0.90, tileDark, 0, -0.02, 0),
    box(0.98, 2.5, 0.05, hallDark, 0, 1.25, 0.45),
    box(0.05, 2.5, 0.90, hallDark, -0.49, 1.25, 0),
    box(0.05, 2.5, 0.90, hallDark, 0.49, 1.25, 0),
    box(0.98, 0.05, 0.90, hallDark, 0, 2.50, 0),
  ], [-0.89, 0, 2.99]);

  // ---------- 窗（钢琴后，+z 墙） ----------
  const winParts = [];
  winParts.push(box(0.75, 0.06, 0.08, doorMat, 0, 0.375, 0));   // 上框
  winParts.push(box(0.75, 0.06, 0.08, doorMat, 0, -0.375, 0));  // 下框
  winParts.push(box(0.06, 0.81, 0.08, doorMat, -0.355, 0, 0));  // 左框
  winParts.push(box(0.06, 0.81, 0.08, doorMat, 0.355, 0, 0));   // 右框
  winParts.push(box(0.04, 0.75, 0.05, doorMat, 0, 0, 0));       // 中梃
  winParts.push(box(0.70, 0.72, 0.02, glassMat, 0, 0, 0.01));   // 玻璃
  winParts.push(box(0.85, 0.04, 0.16, doorMat, 0, -0.43, -0.06)); // 窗台
  winParts.push(box(0.85, 0.28, 0.07, blindMat, 0, 0.60, -0.03)); // 卷帘盒
  // 窗台上的两小盆植物
  winParts.push(cyl(0.045, 0.035, 0.08, terra, -0.20, -0.36, -0.06));
  winParts.push(box(0.10, 0.10, 0.10, leaf, -0.20, -0.27, -0.06));
  winParts.push(cyl(0.045, 0.035, 0.08, terra, 0.15, -0.36, -0.06));
  winParts.push(box(0.09, 0.13, 0.09, leaf, 0.15, -0.26, -0.06));
  grp('window', 'static', winParts, [0.645, 1.475, 2.47]);

  // ---------- 扶手椅窗（-z 墙，纱帘） ----------
  grp('window_armchair', 'static', [
    box(0.99, 0.06, 0.08, doorMat, 0, 0.685, 0),
    box(0.99, 0.06, 0.08, doorMat, 0, -0.685, 0),
    box(0.06, 1.43, 0.08, doorMat, -0.475, 0, 0),
    box(0.06, 1.43, 0.08, doorMat, 0.475, 0, 0),
    box(0.92, 1.36, 0.02, glassMat, 0, 0, 0.01),
    box(0.92, 1.36, 0.015, sheerMat, 0, 0, -0.05),
    box(1.05, 0.04, 0.14, doorMat, 0, -0.72, -0.05),
  ], [-0.125, 1.365, -2.30]);

  // ---------- 窗帘（-z 窗两侧落地窗帘） ----------
  grp('curtains', 'static', [
    box(0.65, 2.10, 0.10, curtainMat, -0.745, 1.10, 0),
    box(0.76, 2.10, 0.10, curtainMat, 0.58, 1.10, 0),
  ], [0, 0, -2.26]);

  // ---------- 暖气片（+z 窗下） ----------
  grp('radiator', 'static', [
    box(0.60, 0.80, 0.08, radWhite, 0, 0, 0),
    box(0.62, 0.04, 0.10, radWhite, 0, 0.42, 0),
  ], [0.80, 0.55, 2.44]);

  // ---------- 凹架搁板 + 橡木书架 ----------
  const shelfParts = [];
  const shelfWhite = doorMat;
  // 桌游凹架：3 块搁板 + 顶层桌游盒
  for (const sy of [1.05, 1.40, 1.72]) shelfParts.push(box(0.14, 0.025, 0.66, shelfWhite, 1.27, sy, -1.715));
  shelfParts.push(box(0.12, 0.10, 0.30, gameBox, 1.27, 1.79, -1.85));
  shelfParts.push(box(0.12, 0.08, 0.25, gameBox, 1.27, 1.78, -1.55));
  shelfParts.push(box(0.10, 0.22, 0.16, artMat, 1.28, 1.52, -1.80)); // 相框
  shelfParts.push(box(0.08, 0.15, 0.12, books, 1.28, 1.13, -1.60));
  // 书架凹架：5 块搁板 + 书
  for (const sy of [0.18, 0.55, 0.92, 1.30, 1.68]) shelfParts.push(box(0.14, 0.025, 0.80, shelfWhite, 1.27, sy, 0.67));
  for (const sy of [0.30, 0.67, 1.04, 1.42, 1.80]) {
    shelfParts.push(box(0.11, 0.22, 0.70, books, 1.28, sy, 0.67));
  }
  shelfParts.push(box(0.10, 0.20, 0.15, artMat, 1.28, 1.55, 0.35)); // 相框
  // 橡木书架（独立式，z[1.30,2.24]，x[1.05,1.33]）
  shelfParts.push(box(0.28, 1.90, 0.04, oak, 1.19, 0.95, 1.32));  // 侧板（前）
  shelfParts.push(box(0.28, 1.90, 0.04, oak, 1.19, 0.95, 2.22));  // 侧板（后）
  shelfParts.push(box(0.28, 0.04, 0.94, oak, 1.19, 1.88, 1.77));  // 顶板
  shelfParts.push(box(0.28, 0.04, 0.94, oak, 1.19, 0.04, 1.77));  // 底板
  shelfParts.push(box(0.03, 1.90, 0.94, oak, 1.325, 0.95, 1.77)); // 背板
  for (const sy of [0.32, 0.62, 0.92, 1.22, 1.52]) {
    shelfParts.push(box(0.26, 0.025, 0.88, oak, 1.19, sy, 1.77));
    shelfParts.push(box(0.20, 0.24, 0.82, books, 1.19, sy + 0.14, 1.77));
  }
  shelfParts.push(cyl(0.11, 0.11, 0.16, artMat, 1.19, 2.00, 1.45, 16)); // 地球仪（简化）
  grp('alcove_shelves', 'static', shelfParts, [0, 0, 0]);

  // ---------- 沙发（主段 x[-1.38,-0.90]，本地原点 (-0.99, 0, -0.55)） ----------
  const sofaParts = [];
  sofaParts.push(box(0.48, 0.42, 1.50, sofaMat, -0.13, 0.21, 0));         // 座体
  sofaParts.push(box(0.18, 0.48, 1.50, sofaMat, -0.30, 0.66, 0));         // 靠背
  sofaParts.push(box(0.48, 0.22, 0.22, sofaMat, -0.13, 0.53, -0.64));     // 扶手（-z 端）
  sofaParts.push(box(0.26, 0.10, 0.46, sofaMat, -0.12, 0.45, -0.48));     // 靠垫
  sofaParts.push(box(0.26, 0.10, 0.46, sofaMat, -0.12, 0.45, 0.02));
  sofaParts.push(box(0.26, 0.10, 0.46, sofaMat, -0.12, 0.45, 0.50));
  // 红色抱枕 + 花抱枕（-z 端角上）
  sofaParts.push(box(0.34, 0.13, 0.34, ctx.material('pillow_red'), -0.05, 0.55, -0.55, [0.15, 0.2, 0.05]));
  sofaParts.push(box(0.32, 0.12, 0.32, cushionMat, -0.05, 0.53, -0.35, [0.12, -0.15, 0.08]));
  grp('sofa', 'movable', sofaParts, [-0.99, 0, -0.55], 0, { mass_kg: 60 });

  // ---------- 单人躺椅（深蓝色） ----------
  const acParts = [];
  acParts.push(box(0.66, 0.38, 0.85, armchairMat, 0, 0.19, 0));           // 底座
  acParts.push(box(0.60, 0.14, 0.75, armchairMat, 0, 0.45, 0.03));        // 座垫
  acParts.push(box(0.64, 0.55, 0.18, armchairMat, 0, 0.66, -0.36, [-0.22, 0, 0])); // 靠背（后仰）
  acParts.push(box(0.13, 0.24, 0.70, armchairMat, -0.33, 0.52, 0.02));    // 扶手
  acParts.push(box(0.13, 0.24, 0.70, armchairMat, 0.33, 0.52, 0.02));
  acParts.push(box(0.48, 0.15, 0.18, armchairMat, 0, 0.90, -0.44, [-0.22, 0, 0])); // 头枕
  grp('armchair', 'movable', acParts, [-0.11, 0, -1.80], 0, { mass_kg: 25 });

  // ---------- 躺椅靠垫 ----------
  grp('armchair_cushion', 'movable', [
    box(0.42, 0.13, 0.40, cushionMat, 0, 0, 0),
  ], [-0.13, 0.56, -1.78], 0.15, { mass_kg: 1 });

  // ---------- 脚凳（蓝垫 + 弯木框） ----------
  const fsParts = [];
  fsParts.push(box(0.52, 0.13, 0.40, footBlue, 0, 0.315, 0));
  fsParts.push(box(0.03, 0.26, 0.42, bentwood, -0.24, 0.13, 0, [0, 0, 0.10]));
  fsParts.push(box(0.03, 0.26, 0.42, bentwood, 0.24, 0.13, 0, [0, 0, -0.10]));
  fsParts.push(box(0.50, 0.03, 0.36, bentwood, 0, 0.245, 0));
  grp('footstool', 'movable', fsParts, [-0.35, 0.011, -0.95], 0.1, { mass_kg: 4 });

  // ---------- 边几（木面 + 黑金属腿，在沙发前） ----------
  const ctParts = [];
  ctParts.push(box(0.44, 0.035, 0.48, tableWood, 0, 0.5275, 0));
  for (const [lx, lz] of [[-0.16, -0.20], [0.16, -0.20], [-0.16, 0.20], [0.16, 0.20]]) {
    ctParts.push(box(0.025, 0.51, 0.025, metalBlack, lx, 0.255, lz));
  }
  ctParts.push(box(0.32, 0.025, 0.025, metalBlack, 0, 0.08, -0.20));
  ctParts.push(box(0.32, 0.025, 0.025, metalBlack, 0, 0.08, 0.20));
  grp('coffee_table', 'movable', ctParts, [-0.66, 0, -0.02], 0, { mass_kg: 6 });

  // ---------- 边几上的东西 ----------
  grp('bowl', 'movable', [
    cyl(0.11, 0.07, 0.07, steel, 0, 0.035, 0),
  ], [-0.80, 0.547, -0.06], 0, { mass_kg: 0.4 });

  grp('carafe', 'movable', [
    cyl(0.045, 0.05, 0.22, glassClear, 0, 0.11, 0),
    cyl(0.018, 0.030, 0.07, glassClear, 0, 0.255, 0),
    cyl(0.020, 0.020, 0.025, capBlue, 0, 0.30, 0),
  ], [-0.62, 0.547, 0.06], 0, { mass_kg: 0.6 });

  grp('wine_glass', 'movable', [
    cyl(0.032, 0.032, 0.008, glassClear, 0, 0.004, 0),
    cyl(0.006, 0.006, 0.07, glassClear, 0, 0.043, 0),
    cyl(0.037, 0.025, 0.095, glassClear, 0, 0.125, 0),
  ], [-0.52, 0.547, -0.14], 0, { mass_kg: 0.2 });

  // ---------- 拖鞋 ----------
  function slipper() {
    return [
      box(0.27, 0.025, 0.105, ctx.material('slipper_brown'), 0, 0.0125, 0),
      box(0.14, 0.055, 0.105, ctx.material('slipper_brown'), 0.055, 0.05, 0),
      box(0.13, 0.03, 0.095, cushionMat, 0.05, 0.045, 0),
    ];
  }
  grp('slipper_left', 'movable', slipper(), [0.02, 0.012, -0.85], 0.35, { mass_kg: 0.2 });
  grp('slipper_right', 'movable', slipper(), [0.20, 0.012, -0.90], -0.25, { mass_kg: 0.2 });

  // ---------- 地毯 ----------
  grp('rug_light', 'movable', [
    box(0.80, 0.01, 1.00, rugLightMat, 0, 0.005, 0),
  ], [-0.15, 0, -0.78], 0, { mass_kg: 3 });
  grp('rug_dark', 'movable', [
    box(1.10, 0.01, 1.42, rugDarkMat, 0, 0.005, 0),
  ], [0.20, 0.01, -0.01], 0, { mass_kg: 4 });

  // ---------- 低音炮 ----------
  grp('subwoofer', 'movable', [
    box(0.36, 0.36, 0.38, speakerMat, 0, 0.18, 0),
    cyl(0.10, 0.10, 0.02, metalBlack, -0.10, 0.20, 0.19, 20, [Math.PI / 2, 0, 0]),
  ], [0.56, 0, -1.24], 0, { mass_kg: 12 });

  // ---------- 电视柜（玻璃 + 金属，内有中置音箱） ----------
  const standParts = [];
  standParts.push(box(0.39, 0.015, 1.30, glassShelf, 0, 0.375, 0));
  standParts.push(box(0.36, 0.015, 1.24, glassShelf, 0, 0.19, 0));
  standParts.push(box(0.39, 0.015, 1.30, glassShelf, 0, 0.03, 0));
  for (const [sx, sz] of [[-0.16, -0.60], [0.16, -0.60], [-0.16, 0.60], [0.16, 0.60]]) {
    standParts.push(box(0.035, 0.39, 0.035, metalBlack, sx, 0.195, sz));
  }
  standParts.push(box(0.20, 0.16, 0.55, speakerMat, 0, 0.27, 0)); // 中置音箱
  grp('tv_stand', 'movable', standParts, [0.995, 0, -0.45], 0, { mass_kg: 20 });

  // ---------- 电视 ----------
  grp('tv', 'movable', [
    box(0.05, 0.68, 1.15, tvBlack, 0, 0, 0),
    box(0.012, 0.63, 1.10, tvScreen, -0.028, 0, 0),
  ], [1.09, 0.735, -0.45], 0, { mass_kg: 18 });

  // ---------- 音箱（含支架） ----------
  function speaker() {
    return [
      box(0.26, 0.02, 0.26, metalBlack, 0, 0.01, 0),
      box(0.05, 0.48, 0.05, metalBlack, 0, 0.26, 0),
      box(0.22, 0.02, 0.24, metalBlack, 0, 0.51, 0),
      box(0.26, 0.40, 0.28, speakerMat, 0, 0.72, 0),
      cyl(0.075, 0.075, 0.02, metalBlack, -0.135, 0.66, 0, 20, [0, 0, Math.PI / 2]),
      cyl(0.040, 0.040, 0.02, metalBlack, -0.135, 0.84, 0, 16, [0, 0, Math.PI / 2]),
    ];
  }
  grp('speaker_left', 'movable', speaker(), [0.71, 0, -0.87], 0, { mass_kg: 7 });
  grp('speaker_right', 'movable', speaker(), [0.88, 0, 0.32], 0, { mass_kg: 7 });

  // ---------- 绿植（陶土盆） ----------
  const plantParts = [];
  plantParts.push(cyl(0.17, 0.17, 0.025, terra, 0, 0.0125, 0));
  plantParts.push(cyl(0.15, 0.11, 0.28, terra, 0, 0.16, 0));
  plantParts.push(cyl(0.025, 0.035, 0.35, bentwood, 0, 0.42, 0));
  for (let i = 0; i < 9; i++) {
    const a = i * Math.PI * 2 / 9;
    const tilt = 0.35 + (i % 3) * 0.18;
    plantParts.push(box(0.07, 0.70, 0.02, leaf, Math.cos(a) * 0.09, 0.72, Math.sin(a) * 0.09,
      [Math.sin(a) * tilt, 0, Math.cos(a) * tilt]));
  }
  for (let i = 0; i < 5; i++) {
    const a = i * Math.PI * 2 / 5 + 0.5;
    plantParts.push(box(0.06, 0.45, 0.02, leaf, Math.cos(a) * 0.06, 0.55, Math.sin(a) * 0.06,
      [Math.sin(a) * 0.7, 0, Math.cos(a) * 0.7]));
  }
  grp('plant', 'movable', plantParts, [0.72, 0, 0.87], 0, { mass_kg: 6 });

  // ---------- 三角钢琴（弧形琴身，高音侧内凹；按片比例比真琴小） ----------
  const pianoParts = [];
  // footprint（局部坐标，原点 (0.15,2.05)）
  const fp = [
    [-0.48, -0.43], [0.48, -0.43], [0.54, 0.37], [0.02, 0.43],
    [-0.25, 0.05], [-0.48, -0.33],
  ];
  const shape = new THREE.Shape();
  shape.moveTo(fp[0][0], fp[0][1]);
  for (let i = 1; i < fp.length; i++) shape.lineTo(fp[i][0], fp[i][1]);
  shape.closePath();
  const bodyGeo = new THREE.ExtrudeGeometry(shape, { depth: 0.30, bevelEnabled: false });
  const body = new THREE.Mesh(bodyGeo, pianoBlack);
  body.rotation.x = Math.PI / 2;          // shape 的 (x,y) -> (x,z)，拉伸方向转为 -y
  body.position.y = 0.95;                 // 琴体 y[0.65,0.95]
  pianoParts.push(body);
  const lidGeo = new THREE.ExtrudeGeometry(shape, { depth: 0.025, bevelEnabled: false });
  const lid = new THREE.Mesh(lidGeo, pianoBlack);
  lid.rotation.x = Math.PI / 2;
  lid.position.y = 0.975;
  lid.scale.set(1.02, 1.02, 1);
  pianoParts.push(lid);
  // 键盘（前缘外凸）
  pianoParts.push(box(0.80, 0.09, 0.14, pianoBlack, 0, 0.735, -0.47));
  pianoParts.push(box(0.72, 0.02, 0.12, keysWhite, 0, 0.775, -0.495));
  pianoParts.push(box(0.72, 0.025, 0.05, pianoBlack, 0, 0.79, -0.455));
  // 三条腿
  pianoParts.push(cyl(0.040, 0.032, 0.65, pianoBlack, 0.38, 0.325, -0.32));
  pianoParts.push(cyl(0.040, 0.032, 0.65, pianoBlack, -0.38, 0.325, -0.24));
  pianoParts.push(cyl(0.040, 0.032, 0.65, pianoBlack, 0.12, 0.325, 0.28));
  // 踏板
  pianoParts.push(box(0.16, 0.05, 0.09, pianoBlack, 0, 0.10, -0.34));
  // 谱架 + 乐谱
  pianoParts.push(box(0.42, 0.28, 0.02, pianoBlack, -0.02, 1.09, -0.20, [-0.30, 0, 0]));
  pianoParts.push(box(0.32, 0.22, 0.012, keysWhite, -0.02, 1.11, -0.215, [-0.30, 0, 0]));
  grp('piano', 'static', pianoParts, [0.15, 0.012, 2.05]);

  // ---------- 琴凳 ----------
  const benchParts = [];
  benchParts.push(box(0.46, 0.055, 0.30, ctx.material('bench_black'), 0, 0.475, 0));
  for (const [lx, lz] of [[-0.19, -0.11], [0.19, -0.11], [-0.19, 0.11], [0.19, 0.11]]) {
    benchParts.push(box(0.035, 0.45, 0.035, pianoBlack, lx, 0.225, lz));
  }
  grp('piano_bench', 'movable', benchParts, [0.12, 0.012, 1.17], 0.08, { mass_kg: 8 });

  // ---------- 钢琴地毯 ----------
  grp('piano_rug', 'movable', [
    box(1.50, 0.01, 1.30, rugPatMat, 0, 0.005, 0),
  ], [0.20, 0, 1.80], 0, { mass_kg: 3 });

  // ---------- 额外：边桌 + 电子琴（-x -z 角） ----------
  const stParts = [];
  stParts.push(box(0.50, 0.03, 0.42, bentwood, 0, 0.465, 0));
  for (const [lx, lz] of [[-0.21, -0.17], [0.21, -0.17], [-0.21, 0.17], [0.21, 0.17]]) {
    stParts.push(box(0.03, 0.45, 0.03, bentwood, lx, 0.225, lz));
  }
  grp('side_table', 'movable', stParts, [-1.08, 0, -1.95], 0, { mass_kg: 5 });

  const kbParts = [];
  kbParts.push(box(0.80, 0.07, 0.24, speakerMat, 0, 0.425, 0));
  kbParts.push(box(0.74, 0.02, 0.14, keysWhite, 0, 0.46, -0.03));
  kbParts.push(box(0.04, 0.40, 0.26, metalBlack, -0.25, 0.20, 0, [0, 0, 0.25]));
  kbParts.push(box(0.04, 0.40, 0.26, metalBlack, 0.25, 0.20, 0, [0, 0, -0.25]));
  grp('music_keyboard', 'movable', kbParts, [-0.95, 0, -1.58], 0.1, { mass_kg: 5 });

  // ---------- 额外：白色小桌（+x 墙，桌游凹架下） ----------
  const deskParts = [];
  deskParts.push(box(0.48, 0.03, 0.42, doorMat, 0, 0.705, 0));
  for (const [lx, lz] of [[-0.21, -0.17], [0.21, -0.17], [-0.21, 0.17], [0.21, 0.17]]) {
    deskParts.push(box(0.03, 0.69, 0.03, doorMat, lx, 0.345, lz));
  }
  deskParts.push(box(0.34, 0.14, 0.28, speakerMat, 0, 0.79, 0)); // 桌上黑色设备
  grp('desk', 'movable', deskParts, [0.93, 0, -1.72], 0, { mass_kg: 15 });

  // ---------- 额外：落地灯（-x 墙 +z 端） ----------
  grp('floor_lamp', 'movable', [
    cyl(0.10, 0.10, 0.02, metalBlack, 0, 0.01, 0),
    cyl(0.014, 0.014, 0.95, metalBlack, 0, 0.495, 0),
    cyl(0.13, 0.13, 0.36, lampWhite, 0, 1.13, 0),
  ], [-1.27, 0, 0.42], 0, { mass_kg: 3 });

  // ---------- 额外：纸箱 ----------
  grp('box_sofa', 'movable', [
    box(0.40, 0.32, 0.35, cardboard, 0, 0.16, 0),
  ], [-1.00, 0, 0.73], 0.2, { mass_kg: 2 });
  grp('box_piano', 'movable', [
    box(0.35, 0.28, 0.32, cardboard, 0, 0.14, 0),
  ], [0.95, 0, 1.80], -0.15, { mass_kg: 1.5 });

  return items;
}

export function environment(THREE) {
  return {
    background: '#ddd6c8',
    lights: [
      new THREE.HemisphereLight(0xfff6e8, 0x8a8072, 1.0),
      (() => { const d = new THREE.DirectionalLight(0xfff0dd, 1.1); d.position.set(0.6, 2.2, 4.5); d.target.position.set(0, 0.8, 0); return d; })(),
      (() => { const d = new THREE.DirectionalLight(0xf2f4ff, 0.55); d.position.set(-0.3, 1.8, -4.5); d.target.position.set(0, 0.8, 0); return d; })(),
      (() => { const p = new THREE.PointLight(0xffedd8, 0.55, 0, 1.6); p.position.set(0, 1.9, 0.2); return p; })(),
    ],
  };
}
