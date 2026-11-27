// Living room + piano end, reconstructed from 24 photos and the sparse point cloud.
// World: y up, floor y = 0, roughly metres. Long axis of the room is z:
//   z = -2.5  bay window with curtains (armchair in front of it)
//   z = +4.55 sash window behind the grand piano
//   x = -1.75 sofa wall (steps in to x = -1.17 at z = 1.25, where the door is)
//   x = +1.22 chimney breast / alcoves (TV side)

export function describe(THREE, ctx) {
  const out = [];
  const H = 2.75; // ceiling height
  const RUG = 0.003; // height of a rug top above the floor
  const ON_RUG = RUG + 0.002;
  const RUG_BOTTOM = -0.03; // rugs are sunk into the floor slab so their colliders are not paper-thin
  const ON_FLOOR = 0.002;

  const r2 = (v) => Math.max(0.15, Math.round(v * 100) / 100);
  function mat(name, a = 1, b = 1) {
    return ctx.material(name, { repeat: [r2(a), r2(b)] });
  }
  // Repeat for a box: sized for its largest face, in that face's own UV directions
  // (BoxGeometry: x faces map u->z, v->y; y faces u->x, v->z; z faces u->x, v->y).
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
  // box given by min/max in the group's local frame
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
  // box given by min/max in WORLD coordinates, for a group that is not rotated
  function wbox(g, m, x0, x1, y0, y1, z0, z1) {
    const p = g.position;
    return lbox(g, m, x0 - p.x, x1 - p.x, y0 - p.y, y1 - p.y, z0 - p.z, z1 - p.z);
  }
  // upright cylinder, local coords of its centre bottom
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
  // disc whose axis points along world/local x (speaker cones, clock)
  function xdisc(g, m, x, y, z, r, t = 0.008) {
    const mesh = new THREE.Mesh(new THREE.CylinderGeometry(r, r, t, 24), mat(m, r * 2, r * 2));
    mesh.rotation.z = Math.PI / 2;
    mesh.position.set(x, y, z);
    g.add(mesh);
    return mesh;
  }
  // Put the group origin at the middle of its parts, so an object is held by its middle
  // (a tall thing held by its foot tips over; a door held at its hinge has no lever arm).
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

  // ------------------------------------------------------------------ floor
  {
    const g = group(0, 0, 0);
    wbox(g, mat('pine_floor', 4.4, 8), -2.5, 1.9, -0.06, 0, -3.1, 4.9);
    add('floor', 'static', g);
  }

  // ------------------------------------------------------------------ walls
  {
    const g = group(0, 0, 0);
    const W = 'wall_cream';
    // sofa wall and the step where the wall comes in toward the door
    wbox(g, W, -1.85, -1.75, 0, H, -2.6, 1.25);
    wbox(g, W, -1.85, -1.17, 0, H, 1.25, 1.62);
    wbox(g, W, -1.27, -1.17, 1.83, H, 1.62, 2.3); // above the doorway
    wbox(g, W, -1.27, -1.17, 0, H, 2.3, 4.65);
    // hallway seen through the open door
    wbox(g, 'hall_dark', -2.45, -2.35, 0, H, 1.3, 2.7);
    wbox(g, 'hall_dark', -2.35, -1.27, 0, H, 1.52, 1.62);
    wbox(g, 'hall_dark', -2.35, -1.27, 0, H, 2.3, 2.4);
    wbox(g, 'hall_tile', -2.35, -1.17, 0, 0.004, 1.62, 2.3);
    // door architrave (kept almost flush so the door leaf never touches it)
    wbox(g, 'trim_white', -1.17, -1.166, 0, 1.91, 1.54, 1.62);
    wbox(g, 'trim_white', -1.17, -1.166, 0, 1.91, 2.3, 2.38);
    wbox(g, 'trim_white', -1.17, -1.166, 1.83, 1.91, 1.62, 2.3);
    // piano-end wall, with the opening for the sash window
    wbox(g, W, -1.27, 0.2, 0, H, 4.55, 4.65);
    wbox(g, W, 1.3, 1.65, 0, H, 4.55, 4.65);
    wbox(g, W, 0.2, 1.3, 0, 0.8, 4.55, 4.65);
    wbox(g, W, 0.2, 1.3, 2.05, H, 4.55, 4.65);
    // TV-side wall: pilaster, alcove backs, chimney breast, plain wall behind the bookcase
    wbox(g, W, 1.22, 1.65, 0, H, -2.6, -2.05);
    wbox(g, W, 1.52, 1.65, 0, H, -2.05, -1.25);
    wbox(g, W, 1.22, 1.52, 2.02, H, -2.05, -1.25);
    wbox(g, W, 1.22, 1.65, 0, H, -1.25, 0.28);
    wbox(g, W, 1.52, 1.65, 0, H, 0.28, 1.08);
    wbox(g, W, 1.22, 1.52, 2.02, H, 0.28, 1.08);
    wbox(g, W, 1.22, 1.65, 0, H, 1.08, 1.31);
    wbox(g, W, 1.55, 1.65, 0, H, 1.31, 4.65);
    // bay-window end: flanking wall pieces, returns, wall under and over the bay window
    wbox(g, W, -1.85, -1.15, 0, H, -2.6, -2.5);
    wbox(g, W, 0.95, 1.65, 0, H, -2.6, -2.5);
    wbox(g, W, -1.2, -1.15, 0, H, -2.95, -2.5);
    wbox(g, W, 0.95, 1.0, 0, H, -2.95, -2.5);
    wbox(g, W, -1.15, 0.95, 0, 0.72, -2.95, -2.85);
    wbox(g, W, -1.15, 0.95, 2.5, H, -2.95, -2.85);
    // skirting boards
    wbox(g, 'trim_white', 1.535, 1.55, 0, 0.13, 2.29, 4.55);
    wbox(g, 'trim_white', -1.17, -1.155, 0, 0.13, 2.6, 4.55);
    wbox(g, 'trim_white', -1.17, 0.2, 0, 0.13, 4.535, 4.55);
    wbox(g, 'trim_white', -1.75, -1.17, 0, 0.13, 1.235, 1.25);
    // ceiling
    wbox(g, 'ceiling_white', -1.9, 1.7, H, H + 0.05, -3.0, 4.7);
    add('walls', 'static', g);
  }

  // ------------------------------------------------------------------ door (open, hinged on the far jamb)
  {
    const hinge = [-1.15, 2.31];
    const ang = (26 * Math.PI) / 180; // leaf direction, measured from +x toward +z
    const g = group(hinge[0], 0, hinge[1], -ang);
    lbox(g, mat('door_white', 0.7, 1.8), 0.01, 0.71, 0.02, 1.82, -0.02, 0.02);
    for (const [x0, x1] of [[0.09, 0.31], [0.4, 0.62]]) {
      for (const [y0, y1] of [[0.22, 0.8], [1.02, 1.68]]) {
        lbox(g, 'door_panel', x0, x1, y0, y1, -0.023, -0.02);
        lbox(g, 'door_panel', x0, x1, y0, y1, 0.02, 0.023);
      }
    }
    lbox(g, 'metal_black', 0.62, 0.67, 0.93, 0.96, -0.06, 0.06); // handle
    add('door', 'movable', g, {
      mass_kg: 15,
      interaction: { grab: 'hinge', axis: [0, 1, 0], pivot: [hinge[0], 0.92, hinge[1]], limit: [-0.2, 1.85] },
    });
  }

  // ------------------------------------------------------------------ window behind the piano
  {
    const g = group(0.75, 1.42, 4.58);
    wbox(g, mat('window_daylight', 1, 1), 0.2, 1.3, 0.8, 2.05, 4.6, 4.62);
    const F = 'trim_white';
    wbox(g, F, 0.2, 0.25, 0.8, 2.05, 4.57, 4.6);
    wbox(g, F, 1.25, 1.3, 0.8, 2.05, 4.57, 4.6);
    wbox(g, F, 0.2, 1.3, 0.8, 0.85, 4.57, 4.6);
    wbox(g, F, 0.2, 1.3, 1.37, 1.42, 4.57, 4.6);
    wbox(g, F, 0.735, 0.765, 0.8, 2.05, 4.575, 4.6);
    wbox(g, F, 0.15, 1.35, 0.76, 0.8, 4.47, 4.57); // sill
    wbox(g, 'blind_olive', 0.17, 1.33, 1.95, 2.3, 4.49, 4.555); // roman blind
    add('window', 'static', g);
  }
  {
    const g = group(0.87, 0.45, 4.5);
    wbox(g, 'radiator_white', 0.45, 1.28, 0.18, 0.72, 4.45, 4.54);
    add('radiator', 'static', g);
  }
  {
    const g = group(1.27, 0.3, 4.0);
    wbox(g, 'cardboard', 1.05, 1.5, 0, 0.36, 3.72, 4.3);
    wbox(g, 'cardboard', 1.12, 1.46, 0.36, 0.6, 3.8, 4.2);
    add('cardboard_boxes', 'static', g);
  }

  // ------------------------------------------------------------------ bay window + curtains at the armchair end
  {
    const g = group(-0.1, 1.2, -2.7);
    wbox(g, mat('net_curtain', 2, 1.8), -1.15, 0.95, 0.72, 2.5, -2.87, -2.85);
    wbox(g, 'trim_white', -1.15, 0.95, 0.68, 0.72, -2.85, -2.7); // sill
    const C = (x0, x1, z0, z1) => wbox(g, mat('curtain_olive', 0.6, 2.5), x0, x1, 0.02, 2.55, z0, z1);
    C(-1.13, -0.58, -2.42, -2.28);
    C(0.2, 0.7, -2.53, -2.43); // kept just behind the armchair so the chair is free to move
    C(0.62, 0.98, -2.34, -2.2); // outer part of the right curtain hangs further into the room
    wbox(g, 'metal_black', -1.2, 1.0, 2.56, 2.59, -2.42, -2.39); // curtain pole
    add('bay_window', 'static', g);
  }

  // ------------------------------------------------------------------ alcove shelves + pine bookcase
  {
    const g = group(1.38, 1.0, 0.1);
    const S = 'shelf_white';
    // left alcove (board games), z -2.05 .. -1.25
    wbox(g, S, 1.1, 1.52, 0, 0.8, -2.05, -1.25);
    wbox(g, S, 1.24, 1.52, 1.37, 1.41, -2.05, -1.25);
    wbox(g, S, 1.24, 1.52, 1.68, 1.72, -2.05, -1.25);
    wbox(g, S, 1.5, 1.52, 0.8, 2.02, -2.05, -1.25);
    wbox(g, 'boardgame_red', 1.28, 1.5, 1.84, 1.93, -2.03, -1.74);
    wbox(g, 'boardgame_red', 1.28, 1.5, 1.78, 1.84, -2.0, -1.76);
    wbox(g, 'paper_white', 1.28, 1.5, 1.72, 1.78, -2.03, -1.74);
    wbox(g, 'paper_white', 1.28, 1.5, 1.79, 1.87, -1.72, -1.44);
    wbox(g, 'boardgame_blue', 1.28, 1.5, 1.72, 1.79, -1.73, -1.42);
    wbox(g, 'books_mixed', 1.3, 1.5, 1.72, 1.9, -1.4, -1.27);
    wbox(g, 'paper_white', 1.29, 1.31, 1.42, 1.69, -2.03, -1.72); // framed b/w photo
    wbox(g, 'photo_dark', 1.285, 1.29, 1.48, 1.65, -2.0, -1.78);
    wbox(g, 'paper_white', 1.2, 1.45, 0.8, 0.83, -1.6, -1.3);
    // right alcove (books), z 0.28 .. 1.08
    wbox(g, S, 1.1, 1.52, 0, 0.85, 0.28, 1.08);
    wbox(g, S, 1.24, 1.52, 1.37, 1.41, 0.28, 1.08);
    wbox(g, S, 1.24, 1.52, 1.68, 1.72, 0.28, 1.08);
    wbox(g, S, 1.5, 1.52, 0.85, 2.02, 0.28, 1.08);
    wbox(g, 'books_bright', 1.3, 1.5, 1.41, 1.65, 0.31, 1.05);
    wbox(g, 'books_bright', 1.3, 1.5, 1.72, 1.97, 0.6, 1.05);
    wbox(g, 'paper_white', 1.29, 1.31, 1.72, 1.98, 0.31, 0.56);
    wbox(g, 'photo_dark', 1.285, 1.29, 1.82, 1.96, 0.34, 0.52);
    wbox(g, 'speaker_black', 1.14, 1.44, 0.85, 0.94, 0.33, 0.76); // amplifier
    wbox(g, 'paper_white', 1.11, 1.13, 0.86, 1.2, 0.27, 0.54); // print leaning in front of it
    wbox(g, 'photo_dark', 1.105, 1.11, 0.93, 1.14, 0.32, 0.49);
    wbox(g, 'magazines_pale', 1.2, 1.5, 0.85, 1.14, 0.84, 1.07);
    // free-standing pine bookcase, z 1.35 .. 2.25
    const P = 'pine_bookcase';
    wbox(g, P, 1.3, 1.55, 0, 1.78, 1.31, 1.33);
    wbox(g, P, 1.3, 1.55, 0, 1.78, 2.26, 2.28);
    wbox(g, P, 1.53, 1.55, 0, 1.78, 1.33, 2.26);
    for (const y of [0.04, 0.38, 0.72, 1.06, 1.4, 1.76]) wbox(g, P, 1.3, 1.53, y, y + 0.02, 1.33, 2.26);
    for (const y of [0.06, 0.4, 0.74, 1.08, 1.42]) {
      wbox(g, mat('books_mixed', 0.9, 0.3), 1.33, 1.52, y, y + 0.25, 1.35, 2.24);
    }
    const globe = new THREE.Mesh(new THREE.SphereGeometry(0.1, 20, 14), mat('globe_tan', 0.4, 0.3));
    globe.position.set(1.42 - g.position.x, 1.93 - g.position.y, 1.55 - g.position.z);
    wcyl(g, 'post_wood', 1.42, 1.78, 1.55, 0.02, 0.05, 0.05, 12);
    g.add(globe);
    add('alcove_shelves', 'static', g);
  }

  // ------------------------------------------------------------------ fireplace behind the TV
  {
    const g = group(1.1, 0.5, -0.48);
    wbox(g, 'hearth_marble', 0.76, 1.22, 0, 0.02, -1.16, 0.22);
    wbox(g, 'mantel_white', 1.0, 1.22, 1.02, 1.06, -1.2, 0.22);
    wbox(g, 'mantel_white', 1.08, 1.22, 0.02, 1.02, -1.17, -0.93);
    wbox(g, 'mantel_white', 1.08, 1.22, 0.02, 1.02, -0.05, 0.19);
    wbox(g, 'mantel_white', 1.08, 1.22, 0.82, 1.02, -0.93, -0.05);
    wbox(g, 'hall_dark', 1.2, 1.22, 0.02, 0.82, -0.93, -0.05);
    // prints and clock standing on the mantel
    wbox(g, 'paper_white', 1.19, 1.215, 1.06, 1.33, -1.17, -0.79);
    wbox(g, 'photo_pink', 1.185, 1.19, 1.11, 1.29, -1.12, -0.84);
    wbox(g, 'paper_white', 1.19, 1.215, 1.06, 1.32, -0.24, 0.15);
    wbox(g, 'photo_green', 1.185, 1.19, 1.11, 1.27, -0.18, 0.09);
    const p = g.position;
    xdisc(g, 'clock_orange', 1.2 - p.x, 1.21 - p.y, -0.52 - p.z, 0.125, 0.02);
    add('fireplace', 'static', g);
  }

  // ------------------------------------------------------------------ rugs (flat, treated as part of the floor)
  {
    const g = group(0.07, 0, -0.25);
    const D = (x0, x1, z0, z1) => wbox(g, mat('rug_dark_green', x1 - x0, z1 - z0), x0, x1, RUG_BOTTOM, RUG, z0, z1);
    D(-0.56, 0.69, -0.21, 0.44);
    D(0.03, 0.69, -0.64, -0.21);
    D(0.3, 0.69, -0.95, -0.64);
    add('rug_dark', 'static', g);
  }
  {
    const g = group(-0.2, 0, -0.8);
    const L = (x0, x1, z0, z1) => wbox(g, mat('rug_light_blue', x1 - x0, z1 - z0), x0, x1, RUG_BOTTOM, RUG, z0, z1);
    L(-0.55, 0.03, -1.38, -0.21);
    L(0.03, 0.3, -1.38, -0.64);
    L(0.3, 0.38, -1.38, -0.95); // stops just short of the subwoofer
    add('rug_light', 'static', g);
  }
  const PIANO_ROT = -(31 * Math.PI) / 180; // local +x -> (cos31, 0, sin31)
  {
    const g = group(0.062, 0, 3.308, PIANO_ROT);
    lbox(g, mat('rug_piano', 1.45, 1.88), -0.725, 0.725, RUG_BOTTOM, RUG, -0.93, 0.95);
    add('piano_rug', 'static', g);
  }

  // ------------------------------------------------------------------ sofa
  {
    const g = group(-1.24, 0, -0.56);
    const F = 'sofa_grey';
    for (const z of [-1.9, -0.55, 0.77]) {
      wbox(g, 'metal_black', -1.68, -1.62, 0, 0.06, z - 0.03, z + 0.03);
      wbox(g, 'metal_black', -0.86, -0.8, 0, 0.06, z - 0.03, z + 0.03);
    }
    wbox(g, F, -1.72, -0.79, 0.06, 0.26, -1.98, 0.85); // base, set back a little under the seat
    wbox(g, F, -1.72, -1.45, 0.26, 0.7, -1.41, 0.5); // back frame
    const seats = [[-1.405, -0.775], [-0.765, -0.135], [-0.125, 0.5]];
    for (const [z0, z1] of seats) {
      wbox(g, F, -1.34, -0.745, 0.26, 0.4, z0, z1);
      wbox(g, F, -1.53, -1.3, 0.4, 0.87, z0 + 0.01, z1 - 0.01);
    }
    wbox(g, F, -1.72, -0.76, 0.26, 0.54, 0.5, 0.85); // near arm
    wbox(g, F, -1.72, -0.71, 0.26, 0.53, -1.98, -1.46); // wide corner block at the far end
    // things lying on the sofa
    wbox(g, 'tray_wood', -1.43, -0.95, 0.645, 0.67, -1.97, -1.62);
    wbox(g, 'metal_black', -1.42, -1.4, 0.53, 0.645, -1.95, -1.64);
    wbox(g, 'metal_black', -0.98, -0.96, 0.53, 0.645, -1.95, -1.64);
    wbox(g, 'keyboard_black', -1.13, -0.71, 0.53, 0.55, -1.58, -1.44);
    wbox(g, 'jacket_red', -1.44, -1.12, 0.4, 0.68, -1.4, -1.08);
    wbox(g, 'cushion_floral', -1.3, -1.12, 0.4, 0.72, -1.06, -0.74);
    wbox(g, 'blanket_dark', -1.3, -0.8, 0.4, 0.47, 0.02, 0.48);
    wbox(g, 'keyboard_black', -1.3, -0.95, 0.54, 0.558, 0.53, 0.8); // closed laptop on the arm
    add('sofa', 'static', g);
  }
  {
    const g = group(-1.62, 0, 1.0);
    lcyl(g, 'lamp_white', 0, 0, 0, 0.1, 0.1, 1.25, 24);
    add('floor_lamp', 'static', g);
  }

  // ------------------------------------------------------------------ armchair (recliner) + cushion
  {
    const g = group(-0.125, 0, -2.07);
    const A = 'armchair_blue';
    wbox(g, A, -0.5, 0.25, ON_FLOOR, 0.2, -2.42, -1.74);
    wbox(g, A, -0.52, -0.34, ON_FLOOR, 0.48, -2.4, -1.74);
    wbox(g, A, 0.09, 0.27, ON_FLOOR, 0.48, -2.4, -1.74);
    wbox(g, A, -0.34, 0.09, 0.2, 0.38, -2.24, -1.7);
    wbox(g, A, -0.4, 0.15, 0.38, 0.68, -2.44, -2.22);
    wbox(g, A, -0.4, 0.15, 0.66, 0.93, -2.56, -2.33);
    add('armchair', 'movable', g, { mass_kg: 25 });
  }
  {
    const g = group(-0.15, 0.382, -2.11);
    lbox(g, 'cushion_cream', -0.17, 0.17, 0, 0.2, -0.09, 0.09);
    lbox(g, 'cushion_brown', -0.12, -0.03, 0.09, 0.17, 0.09, 0.093);
    lbox(g, 'cushion_brown', 0.05, 0.13, 0.03, 0.1, 0.09, 0.093);
    lbox(g, 'cushion_brown', -0.02, 0.06, 0.13, 0.19, 0.09, 0.093);
    lbox(g, 'cushion_brown', -0.15, -0.09, 0.01, 0.06, 0.09, 0.093);
    add('armchair_cushion', 'movable', g, { mass_kg: 0.6 });
  }

  // ------------------------------------------------------------------ footstool
  {
    const g = group(-0.475, 0, -1.165);
    const B = 'birch_frame';
    for (const z of [-0.255, 0.275]) {
      lbox(g, B, -0.25, -0.2, ON_FLOOR, 0.27, z - 0.02, z + 0.02);
      lbox(g, B, 0.2, 0.25, z > 0 ? ON_RUG : ON_FLOOR, 0.27, z - 0.02, z + 0.02); // only this front foot is on the rug
      lbox(g, B, -0.25, 0.25, 0.27, 0.3, z - 0.02, z + 0.02);
    }
    lbox(g, B, -0.22, 0.22, 0.27, 0.3, -0.24, 0.26);
    lbox(g, mat('footstool_blue', 0.5, 0.5), -0.235, 0.235, 0.3, 0.37, -0.275, 0.205);
    add('footstool', 'movable', g, { mass_kg: 5 });
  }

  // ------------------------------------------------------------------ C-shaped side table and the things on it
  const TABLE_TOP = 0.58;
  {
    const g = group(-0.705, 0, -0.015);
    const K = 'metal_black';
    wbox(g, mat('table_walnut', 0.47, 0.37), -0.94, -0.47, 0.555, TABLE_TOP, -0.2, 0.17);
    for (const z of [-0.19, 0.16]) {
      wbox(g, K, -0.49, -0.47, 0.026, 0.535, z - 0.01, z + 0.01); // posts
      wbox(g, K, -0.555, -0.47, ON_RUG, 0.026, z - 0.01, z + 0.01); // floor rails: the end on the rug...
      wbox(g, K, -0.745, -0.565, ON_FLOOR, 0.026, z - 0.01, z + 0.01); // ...and the end on bare floor, stopping at the sofa front
      wbox(g, K, -0.93, -0.47, 0.535, 0.555, z - 0.01, z + 0.01); // rails under the top
    }
    wbox(g, K, -0.49, -0.47, ON_RUG, 0.026, -0.18, 0.15);
    wbox(g, K, -0.49, -0.47, 0.535, 0.555, -0.18, 0.15);
    wbox(g, K, -0.93, -0.91, 0.535, 0.555, -0.18, 0.15);
    add('coffee_table', 'movable', g, { mass_kg: 6 });
  }
  {
    const g = group(-0.745, TABLE_TOP + 0.002, -0.03);
    lcyl(g, 'steel_bowl', 0, 0, 0, 0.095, 0.055, 0.09, 24);
    lcyl(g, 'steel_inside', 0, 0.09, 0, 0.085, 0.085, 0.001, 24);
    add('bowl', 'movable', g, { mass_kg: 0.4 });
  }
  {
    const g = group(-0.6, TABLE_TOP + 0.002, -0.09);
    lcyl(g, 'glass_clear', 0, 0, 0, 0.03, 0.045, 0.14, 20);
    lcyl(g, 'glass_clear', 0, 0.14, 0, 0.032, 0.022, 0.08, 20);
    add('carafe', 'movable', g, { mass_kg: 0.9 });
  }
  {
    const g = group(-0.895, TABLE_TOP + 0.002, -0.13);
    lcyl(g, 'glass_clear', 0, 0, 0, 0.04, 0.03, 0.11, 20);
    add('wine_glass', 'movable', g, { mass_kg: 0.2 });
  }

  // ------------------------------------------------------------------ slippers on the light rug
  function slipper(id, x, z, rotY, len) {
    const g = group(x, ON_RUG, z, rotY);
    const h = len / 2;
    lbox(g, 'slipper_tan', -h, h, 0, 0.022, -0.048, 0.048); // sole
    lbox(g, 'slipper_tan', 0, h, 0.022, 0.07, -0.048, 0.048); // upper over the toes
    lbox(g, 'slipper_trim', -0.02, 0.0, 0.022, 0.072, -0.048, 0.048); // cream cuff
    add(id, 'movable', g, { mass_kg: 0.2 });
  }
  slipper('slipper_left', 0.04, -0.905, 0.63, 0.26);
  slipper('slipper_right', 0.23, -0.88, -1.5, 0.24);

  // ------------------------------------------------------------------ subwoofer
  {
    const g = group(0.56, 0, -1.27, 0.14);
    lbox(g, mat('speaker_black', 0.43, 0.44), -0.142, 0.142, ON_FLOOR, 0.44, -0.207, 0.207);
    add('subwoofer', 'movable', g, { mass_kg: 12 });
  }

  // ------------------------------------------------------------------ TV stand (glass + posts + centre speaker) and TV
  {
    const g = group(0.92, 0, -0.475);
    wbox(g, 'stand_glass', 0.78, 1.06, 0.035, 0.047, -1.15, 0.2);
    wbox(g, 'stand_glass', 0.78, 1.06, 0.338, 0.35, -1.15, 0.2);
    for (const x of [0.802, 1.038]) {
      for (const z of [-1.125, 0.175]) wcyl(g, 'post_wood', x, 0.02, z, 0.022, 0.022, 0.4, 14);
    }
    wbox(g, 'speaker_black', 0.8, 1.02, 0.05, 0.25, -0.7, -0.15);
    const p = g.position;
    for (const z of [-0.6, -0.25]) xdisc(g, 'speaker_cone', 0.798 - p.x, 0.15 - p.y, z - p.z, 0.06);
    xdisc(g, 'speaker_cone', 0.798 - p.x, 0.15 - p.y, -0.425 - p.z, 0.03);
    add('tv_stand', 'static', g);
  }
  {
    const g = group(0.94, 0.352, -0.47, -0.046);
    lbox(g, 'tv_black', -0.1, 0.1, 0, 0.03, -0.25, 0.25);
    lbox(g, 'tv_black', -0.02, 0.02, 0.03, 0.06, -0.15, 0.15);
    lbox(g, mat('tv_black', 1.3, 0.73), -0.015, 0.015, 0.05, 0.78, -0.65, 0.65);
    add('tv', 'movable', g, { mass_kg: 15 });
  }

  // ------------------------------------------------------------------ speakers on stands
  function speaker(id, x, z, cabBottom, cabH, withFlowers) {
    const g = group(x, 0, z);
    const K = 'speaker_black';
    lbox(g, K, -0.14, 0.2, ON_FLOOR, 0.03, -0.14, 0.14); // base plate
    lbox(g, K, -0.02, 0.02, 0.03, cabBottom - 0.02, -0.07, 0.07); // twin columns
    lbox(g, K, 0.06, 0.1, 0.03, cabBottom - 0.02, -0.07, 0.07);
    lbox(g, K, -0.12, 0.2, cabBottom - 0.02, cabBottom, -0.1, 0.1); // top plate
    const top = cabBottom + cabH;
    lbox(g, mat(K, cabH, 0.36), -0.14, 0.22, cabBottom, top, -0.11, 0.11); // cabinet
    xdisc(g, 'speaker_cone', -0.142, cabBottom + 0.12, 0, 0.062);
    xdisc(g, 'speaker_cone', -0.142, cabBottom + 0.28, 0, 0.022);
    if (withFlowers) {
      lbox(g, 'post_wood', -0.03, 0.17, top, top + 0.025, -0.08, 0.08); // wooden tray
      lcyl(g, 'post_wood', 0.07, top + 0.025, 0, 0.04, 0.035, 0.05, 12);
      lbox(g, 'orchid_pink', 0.0, 0.15, top + 0.075, top + 0.15, -0.08, 0.07);
    }
    add(id, 'movable', g, { mass_kg: 8 });
  }
  speaker('speaker_left', 0.86, -1.385, 0.5, 0.37, false);
  speaker('speaker_right', 0.84, 0.395, 0.57, 0.35, true);

  // ------------------------------------------------------------------ yucca in a terracotta pot
  {
    const g = group(0.7, 0, 0.86);
    lcyl(g, 'terracotta', 0, ON_FLOOR, 0, 0.115, 0.1, 0.02, 20); // saucer
    lcyl(g, 'terracotta', 0, 0.022, 0, 0.095, 0.07, 0.17, 20); // pot
    lcyl(g, 'soil_dark', 0, 0.192, 0, 0.085, 0.085, 0.004, 20);
    lcyl(g, 'plant_trunk', 0.01, 0.196, 0, 0.018, 0.022, 0.66, 10);
    lcyl(g, 'plant_trunk', -0.04, 0.196, 0.03, 0.014, 0.016, 0.24, 10);
    const crown = (cx, cy, cz, n, len, phase) => {
      for (let i = 0; i < n; i++) {
        const az = phase + (i / n) * Math.PI * 2;
        const el = -0.25 + 1.3 * ((i * 7) % 6) / 5; // from drooping to steeply up
        const leaf = new THREE.Mesh(new THREE.BoxGeometry(len, 0.004, 0.032), mat('plant_leaf', len, 0.15));
        const dx = Math.cos(el) * Math.cos(az), dy = Math.sin(el), dz = -Math.cos(el) * Math.sin(az);
        leaf.rotation.set(0, az, el, 'YXZ');
        leaf.position.set(cx + (dx * len) / 2, cy + (dy * len) / 2, cz + (dz * len) / 2);
        g.add(leaf);
      }
    };
    crown(0.01, 0.84, 0, 16, 0.38, 0.2);
    crown(0.01, 0.88, 0, 7, 0.28, 0.7);
    crown(-0.04, 0.43, 0.03, 14, 0.34, 0.0);
    add('plant', 'movable', g, { mass_kg: 4 });
  }

  // ------------------------------------------------------------------ grand piano (keyboard faces -z, turned 31 degrees)
  {
    const g = group(0.3, 0, 2.76, PIANO_ROT); // origin: middle of the key fronts, on the floor
    const B = 'piano_black';
    lbox(g, B, -0.74, 0.74, 0.56, 0.62, 0, 0.2); // key bed
    lbox(g, mat('piano_keys', 1.2, 0.15), -0.62, 0.62, 0.62, 0.66, 0.008, 0.16);
    lbox(g, B, -0.74, -0.63, 0.62, 0.76, 0, 0.2); // cheeks
    lbox(g, B, 0.63, 0.74, 0.62, 0.76, 0, 0.2);
    lbox(g, B, -0.63, 0.63, 0.66, 0.86, 0.165, 0.2); // fallboard
    // rim: straight bass side at +x, stepped in toward the tail on the treble side
    lbox(g, B, -0.74, 0.74, 0.6, 0.88, 0.2, 0.62);
    lbox(g, B, -0.62, 0.74, 0.6, 0.88, 0.62, 0.95);
    lbox(g, B, -0.4, 0.74, 0.6, 0.88, 0.95, 1.25);
    lbox(g, B, -0.1, 0.74, 0.6, 0.88, 1.25, 1.5);
    // lid
    lbox(g, B, -0.75, 0.75, 0.88, 0.9, 0.2, 0.63);
    lbox(g, B, -0.63, 0.75, 0.88, 0.9, 0.63, 0.96);
    lbox(g, B, -0.41, 0.75, 0.88, 0.9, 0.96, 1.26);
    lbox(g, B, -0.11, 0.75, 0.88, 0.9, 1.26, 1.51);
    // music desk with open score, books lying on the lid
    lbox(g, B, -0.36, 0.36, 0.9, 1.1, 0.27, 0.285);
    lbox(g, 'paper_white', -0.17, 0.22, 0.9, 1.13, 0.258, 0.268);
    lbox(g, 'book_pale', 0.36, 0.62, 0.9, 0.96, 0.3, 0.5);
    lbox(g, 'book_green', -0.52, -0.3, 0.9, 0.93, 0.32, 0.48);
    // legs and pedal lyre
    for (const [x, z] of [[-0.64, 0.28], [0.64, 0.28], [0.32, 1.32]]) {
      lbox(g, B, x - 0.05, x + 0.05, 0.03, 0.6, z - 0.05, z + 0.05);
      lbox(g, 'metal_brass', x - 0.03, x + 0.03, 0.005, 0.03, z - 0.03, z + 0.03);
    }
    lbox(g, B, -0.09, 0.09, 0.07, 0.6, 0.23, 0.28);
    lbox(g, B, -0.13, 0.13, 0.03, 0.09, 0.18, 0.3);
    add('piano', 'static', g);
  }
  {
    const g = group(0.53, 0, 2.6, PIANO_ROT);
    const B = 'piano_black';
    lbox(g, mat('bench_leather', 0.56, 0.34), -0.28, 0.28, 0.34, 0.42, -0.17, 0.17);
    lbox(g, B, -0.25, 0.25, 0.28, 0.34, -0.14, 0.14);
    for (const x of [-0.23, 0.23]) {
      for (const z of [-0.12, 0.12]) lbox(g, B, x - 0.02, x + 0.02, z > 0 ? ON_RUG : ON_FLOOR, 0.28, z - 0.02, z + 0.02);
    }
    add('piano_bench', 'movable', g, { mass_kg: 9 });
  }

  return out;
}

export function environment(THREE) {
  // Even, neutral light: the photos are lit by soft bounced room light, and the colours
  // in materials.json are already the colours as they appear in the photos.
  const amb = new THREE.AmbientLight(0xffffff, 2.3);
  const hemi = new THREE.HemisphereLight(0xffffff, 0xb0a890, 0.5);
  const key = new THREE.DirectionalLight(0xffffff, 0.5);
  key.position.set(-0.6, 3, 1.2);
  return { background: '#1b1a17', lights: [amb, hemi, key] };
}
