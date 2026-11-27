// Living room reconstruction: long narrow room, y up, floor at y=0.
// South (z=-2.45): window + curtains + armchair. East (x=1.30): TV wall + alcove + bookshelf.
// North (z=3.80): window + radiator + grand piano. West (x=-1.36): sofa + doorway (open door).

export function describe(THREE, ctx) {
  const M = (n, r) => (r ? ctx.material(n, { repeat: r }) : ctx.material(n));
  const box = (w, h, d, mat) => new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  const cyl = (rt, rb, h, mat, seg = 18) => new THREE.Mesh(new THREE.CylinderGeometry(rt, rb, h, seg), mat);
  const sph = (r, mat) => new THREE.Mesh(new THREE.SphereGeometry(r, 16, 12), mat);

  const objs = [];
  // helper: object with origin at (cx,cy,cz); add(mesh, wx,wy,wz, rx,ry,rz) places by world coords
  function obj(id, kind, cx, cy, cz, extra = {}) {
    const g = new THREE.Group();
    g.position.set(cx, cy, cz);
    const o = { id, kind, object3D: g, ...extra };
    objs.push(o);
    return {
      g, o,
      add(mesh, wx, wy, wz, rx = 0, ry = 0, rz = 0) {
        mesh.position.set(wx - cx, wy - cy, wz - cz);
        mesh.rotation.set(rx, ry, rz);
        g.add(mesh);
        return mesh;
      },
    };
  }

  // ---------------- floor ----------------
  {
    const { add } = obj("floor", "static", -0.03, -0.03, 0.675);
    add(box(2.66, 0.06, 6.25, M("oak_floor", [2.7, 6.3])), -0.03, -0.031, 0.675);
  }

  // ---------------- walls (+ceiling +hallway) ----------------
  {
    const { add } = obj("walls", "static", 0, 1.3, 0.7);
    const wp = () => M("wall_paint", [3, 2.6]);
    // west wall x face -1.36 (doorway z 1.44..2.24, lintel above y 2.02)
    add(box(0.08, 2.58, 3.89, wp()), -1.40, 1.29, -0.505);
    add(box(0.08, 2.58, 1.56, wp()), -1.40, 1.29, 3.02);
    add(box(0.08, 0.56, 0.80, wp()), -1.40, 2.30, 1.84);
    // east wall x face 1.30: south alcove z -2.05..-1.30, chimney breast z -1.30..0.55, north alcove z 0.55..1.24
    add(box(0.08, 2.58, 0.40, wp()), 1.34, 1.29, -2.25);
    add(box(0.08, 0.78, 0.75, wp()), 1.34, 2.19, -1.675);
    add(box(0.08, 2.58, 1.85, wp()), 1.34, 1.29, -0.375);
    add(box(0.08, 0.72, 0.69, wp()), 1.34, 2.22, 0.895);
    add(box(0.08, 2.58, 2.56, wp()), 1.34, 1.29, 2.52);
    // south wall z face -2.45 (window x -0.50..0.43, y 0.85..2.05)
    add(box(0.86, 2.58, 0.08, wp()), -0.93, 1.29, -2.49);
    add(box(0.87, 2.58, 0.08, wp()), 0.865, 1.29, -2.49);
    add(box(0.93, 0.85, 0.08, wp()), -0.035, 0.425, -2.49);
    add(box(0.93, 0.53, 0.08, wp()), -0.035, 2.315, -2.49);
    // north wall z face 3.80 (window x 0.00..1.10, y 1.00..2.10)
    add(box(1.36, 2.58, 0.08, wp()), -0.68, 1.29, 3.84);
    add(box(0.20, 2.58, 0.08, wp()), 1.20, 1.29, 3.84);
    add(box(1.10, 1.00, 0.08, wp()), 0.55, 0.50, 3.84);
    add(box(1.10, 0.48, 0.08, wp()), 0.55, 2.34, 3.84);
    // ceiling
    add(box(2.74, 0.06, 6.33, M("ceiling_paint", [3, 6])), -0.03, 2.58, 0.675);
    // hallway beyond the door (dark)
    add(box(0.90, 0.06, 2.20, M("hall_dark", [1, 2])), -1.89, -0.031, 2.40);
    add(box(0.08, 2.58, 2.20, M("hall_dark", [2, 2])), -2.38, 1.29, 2.40);
    add(box(0.98, 2.58, 0.08, M("hall_dark", [1, 2])), -1.93, 1.29, 1.26);
    add(box(0.98, 2.58, 0.08, M("hall_dark", [1, 2])), -1.93, 1.29, 3.54);
    add(box(0.90, 0.06, 2.20, M("hall_dark", [1, 2])), -1.89, 2.55, 2.40);
  }

  // ---------------- south window (extra) + curtains ----------------
  {
    const { add } = obj("window_south", "static", -0.035, 1.45, -2.46);
    add(box(1.01, 0.05, 0.10, M("shelf_white")), -0.035, 0.875, -2.44); // sill
    add(box(0.93, 1.16, 0.02, M("window_glass")), -0.035, 1.45, -2.475); // pane
    add(box(0.05, 1.20, 0.05, M("door_white")), -0.50, 1.45, -2.46); // frame sides
    add(box(0.05, 1.20, 0.05, M("door_white")), 0.43, 1.45, -2.46);
    add(box(0.93, 0.05, 0.05, M("door_white")), -0.035, 2.05, -2.46);
    add(box(0.04, 1.16, 0.04, M("door_white")), -0.035, 1.45, -2.465); // mullion
  }
  {
    const { add } = obj("curtains", "static", -0.05, 1.15, -2.31);
    add(box(0.55, 2.26, 0.22, M("curtain_olive", [1, 2])), -0.785, 1.15, -2.31);
    add(box(0.53, 2.26, 0.22, M("curtain_olive", [1, 2])), 0.685, 1.15, -2.31);
  }

  // ---------------- north window (manifest "window") ----------------
  {
    const { add } = obj("window", "static", 0.55, 1.55, 3.80);
    add(box(1.18, 0.05, 0.12, M("shelf_white")), 0.55, 1.025, 3.80); // sill
    add(box(1.10, 1.06, 0.02, M("window_glass")), 0.55, 1.58, 3.83); // pane
    add(box(0.05, 1.10, 0.05, M("door_white")), 0.025, 1.57, 3.81);
    add(box(0.05, 1.10, 0.05, M("door_white")), 1.075, 1.57, 3.81);
    add(box(1.14, 0.42, 0.05, M("curtain_olive", [2, 1])), 0.55, 1.94, 3.775); // roman blind
  }
  // radiator under north window
  {
    const { add } = obj("radiator", "static", 0.55, 0.5, 3.71);
    add(box(1.00, 0.70, 0.12, M("radiator_white", [4, 1])), 0.55, 0.50, 3.71);
  }

  // ---------------- door (open ~90 deg, hinged at north jamb) ----------------
  {
    const { add } = obj("door", "movable", -0.95, 1.0, 2.235, {
      mass_kg: 14,
      interaction: { grab: "hinge", axis: [0, 1, 0], pivot: [-1.345, 1.0, 2.235], limit: [-1.2, 1.6] },
    });
    add(box(0.79, 1.98, 0.045, M("door_white", [1, 2])), -0.95, 1.01, 2.235);
    add(box(0.11, 0.025, 0.025, M("steel_brushed")), -0.615, 1.03, 2.20); // handle
    add(box(0.025, 0.10, 0.025, M("steel_brushed")), -0.575, 1.00, 2.212);
  }

  // ---------------- alcove shelves (two recessed alcoves flanking the TV breast) ----------------
  {
    const { add } = obj("alcove_shelves", "static", 1.3, 0.95, -0.4);
    const sw = () => M("shelf_white", [1, 1]);
    // south alcove z -2.05..-1.30 (board games above, printer below)
    add(box(0.06, 1.80, 0.75, sw()), 1.45, 0.90, -1.675); // back
    add(box(0.36, 0.04, 0.71, sw()), 1.26, 0.94, -1.675); // boards
    add(box(0.36, 0.04, 0.71, sw()), 1.26, 1.32, -1.675);
    add(box(0.36, 0.04, 0.71, sw()), 1.26, 1.78, -1.675);
    add(box(0.34, 0.09, 0.44, M("cardboard")), 1.24, 1.387, -1.70); // game boxes
    add(box(0.32, 0.08, 0.42, M("book_spines")), 1.24, 1.472, -1.70);
    add(box(0.30, 0.08, 0.40, M("cushion_red")), 1.24, 1.552, -1.71);
    add(box(0.22, 0.30, 0.20, M("book_spines", [1, 1])), 1.25, 1.492, -1.42);
    add(box(0.03, 0.28, 0.24, M("sheet_music")), 1.22, 1.102, -1.85); // framed photo
    add(sph(0.06, M("globe_tan")), 1.22, 1.022, -1.70); // wooden ball
    add(box(0.12, 0.09, 0.12, M("metal_black")), 1.24, 1.007, -1.55);
    add(cyl(0.03, 0.03, 0.12, M("glass_clear"), 10), 1.26, 1.022, -1.45);
    // north alcove z 0.55..1.24 (books and records)
    add(box(0.06, 1.86, 0.69, sw()), 1.45, 0.93, 0.895); // back
    add(box(0.36, 0.04, 0.65, sw()), 1.26, 0.60, 0.895); // boards
    add(box(0.36, 0.04, 0.65, sw()), 1.26, 0.98, 0.895);
    add(box(0.36, 0.04, 0.65, sw()), 1.26, 1.36, 0.895);
    add(box(0.36, 0.04, 0.65, sw()), 1.26, 1.74, 0.895);
    add(box(0.30, 0.34, 0.50, M("sheet_music", [1, 1])), 1.27, 0.192, 0.85); // record stacks
    add(box(0.24, 0.30, 0.50, M("book_spines", [2, 1])), 1.28, 0.775, 0.84); // books
    add(box(0.24, 0.32, 0.44, M("book_spines", [2, 1])), 1.28, 1.16, 0.82);
    add(box(0.26, 0.12, 0.12, M("sheet_music")), 1.26, 1.06, 1.16); // paper stack
    add(box(0.28, 0.24, 0.36, M("book_spines", [1, 1])), 1.27, 1.50, 0.80);
    add(cyl(0.05, 0.04, 0.09, M("terracotta")), 1.15, 0.407, 0.68); // orchid on records
    add(cyl(0.010, 0.010, 0.09, M("plant_green")), 1.15, 0.497, 0.68);
    add(sph(0.05, M("orchid_pink")), 1.15, 0.575, 0.68);
  }

  // printer + leaning print in the south alcove (extra)
  {
    const { add } = obj("printer", "movable", 1.18, 0.2, -1.72, { mass_kg: 5 });
    add(box(0.36, 0.28, 0.42, M("purifier_white")), 1.18, 0.145, -1.72);
    add(box(0.30, 0.04, 0.30, M("purifier_white")), 1.16, 0.305, -1.72);
    add(box(0.03, 0.30, 0.26, M("orchid_pink", [1, 1])), 1.13, 0.445, -1.60, 0, 0, -0.12); // leaning print
  }

  // ---------------- sofa (sectional, static) ----------------
  {
    const { add } = obj("sofa", "static", -0.95, 0.45, 0.1);
    const sf = () => M("sofa_grey", [1.5, 1.5]);
    // feet
    for (const [fx, fz] of [[-1.28, -1.24], [-0.66, -1.24], [-1.28, 0.47], [-0.66, 0.47], [-1.28, 0.98], [-0.44, 0.98]])
      add(box(0.08, 0.12, 0.08, M("metal_black")), fx, 0.08, fz);
    // main run z -1.32..0.55
    add(box(0.77, 0.16, 1.87, sf()), -0.965, 0.22, -0.385); // base
    add(box(0.72, 0.16, 0.60, sf()), -0.94, 0.38, -1.00); // seat cushions
    add(box(0.72, 0.16, 0.60, sf()), -0.94, 0.38, -0.385);
    add(box(0.72, 0.16, 0.60, sf()), -0.94, 0.38, 0.23);
    add(box(0.22, 0.70, 1.87, sf()), -1.24, 0.65, -0.385); // back
    add(box(0.20, 0.45, 0.58, sf()), -1.06, 0.78, -1.00, 0, 0, 0.08); // back cushions
    add(box(0.20, 0.45, 0.58, sf()), -1.06, 0.78, -0.385, 0, 0, 0.08);
    add(box(0.20, 0.45, 0.58, sf()), -1.06, 0.78, 0.23, 0, 0, 0.08);
    add(box(0.77, 0.28, 0.20, sf()), -0.965, 0.44, -1.22); // south arm
    // chaise z 0.55..1.05
    add(box(1.00, 0.16, 0.50, sf()), -0.85, 0.22, 0.80);
    add(box(0.96, 0.16, 0.48, sf()), -0.84, 0.38, 0.80);
    add(box(0.25, 0.40, 0.50, sf()), -1.225, 0.66, 0.80); // low back on wall side
    // cushions / clutter on the sofa
    add(box(0.40, 0.13, 0.40, M("cushion_red", [1, 1])), -1.00, 0.525, -1.02, 0, 0.3, 0); // red throw cushion
    add(box(0.42, 0.14, 0.42, M("book_spines", [1, 1])), -1.02, 0.53, -0.45, 0, -0.2, 0); // patterned cushion
    add(box(0.55, 0.12, 0.40, M("hall_dark", [1, 1])), -0.85, 0.52, 0.82, 0, 0.15, 0); // dark throw on chaise
    add(box(0.34, 0.025, 0.24, M("metal_black")), -0.72, 0.472, 0.68, 0, -0.25, 0); // laptop on chaise
  }

  // ---------------- armchair (recliner, movable) ----------------
  {
    const { add } = obj("armchair", "movable", -0.10, 0.5, -2.0, { mass_kg: 32 });
    const ab = () => M("armchair_fabric", [1.2, 1.2]);
    add(box(0.78, 0.26, 0.84, ab()), -0.10, 0.15, -2.00); // base
    add(box(0.50, 0.16, 0.55, ab()), -0.10, 0.36, -1.90); // seat cushion
    add(box(0.14, 0.34, 0.80, ab()), -0.42, 0.45, -2.00); // armrests
    add(box(0.14, 0.34, 0.80, ab()), 0.22, 0.45, -2.00);
    add(box(0.78, 0.60, 0.20, ab()), -0.10, 0.68, -2.26, -0.18, 0, 0); // backrest
    add(box(0.50, 0.18, 0.14, ab()), -0.10, 1.02, -2.32, -0.18, 0, 0); // headrest
  }
  {
    const { add } = obj("armchair_cushion", "movable", -0.10, 0.665, -2.0, { mass_kg: 1 });
    add(box(0.44, 0.42, 0.14, M("cushion_cow", [1, 1])), -0.10, 0.665, -2.0, -0.15, 0, 0);
  }

  // ---------------- footstool (bentwood + blue cushion) ----------------
  {
    const { add } = obj("footstool", "movable", -0.31, 0.2, -1.06, { mass_kg: 4 });
    add(box(0.50, 0.03, 0.36, M("bentwood")), -0.31, 0.047, -1.06); // floor rail
    add(box(0.05, 0.158, 0.40, M("bentwood")), -0.535, 0.141, -1.06); // side panels
    add(box(0.05, 0.158, 0.40, M("bentwood")), -0.085, 0.141, -1.06);
    add(box(0.40, 0.05, 0.38, M("bentwood")), -0.31, 0.245, -1.06); // top rail
    add(box(0.52, 0.13, 0.42, M("footstool_blue", [1, 1])), -0.31, 0.335, -1.06); // cushion
  }

  // ---------------- coffee table (C-table over sofa seat) ----------------
  {
    const { add } = obj("coffee_table", "movable", -0.74, 0.33, -0.23, { mass_kg: 7 });
    add(box(0.43, 0.05, 0.58, M("table_wood", [1, 1])), -0.735, 0.625, -0.23);
    const mb = () => M("metal_black");
    add(box(0.50, 0.04, 0.09, mb()), -0.79, 0.026, -0.44); // feet under sofa
    add(box(0.50, 0.04, 0.09, mb()), -0.79, 0.026, -0.02);
    add(box(0.045, 0.56, 0.045, M("metal_black")), -0.552, 0.325, -0.44); // posts
    add(box(0.045, 0.56, 0.045, M("metal_black")), -0.552, 0.325, -0.02);
    add(box(0.40, 0.045, 0.045, mb()), -0.755, 0.5825, -0.44); // top arms
    add(box(0.40, 0.045, 0.045, mb()), -0.755, 0.5825, -0.02);
  }
  // items on the coffee table
  {
    const { add } = obj("bowl", "movable", -0.82, 0.69, -0.20, { mass_kg: 0.4 });
    add(cyl(0.105, 0.055, 0.07, M("steel_brushed"), 24), -0.82, 0.6925, -0.20);
  }
  {
    const { add } = obj("carafe", "movable", -0.70, 0.78, 0.0, { mass_kg: 0.5 });
    add(cyl(0.045, 0.045, 0.22, M("glass_clear"), 16), -0.70, 0.7675, 0.0);
    add(cyl(0.018, 0.035, 0.05, M("glass_clear"), 12), -0.70, 0.9025, 0.0);
    add(cyl(0.020, 0.020, 0.02, M("plastic_blue"), 12), -0.70, 0.9375, 0.0);
  }
  {
    const { add } = obj("wine_glass", "movable", -0.90, 0.74, -0.02, { mass_kg: 0.15 });
    add(cyl(0.034, 0.034, 0.006, M("glass_clear"), 14), -0.90, 0.6605, -0.02);
    add(cyl(0.006, 0.006, 0.07, M("glass_clear"), 8), -0.90, 0.6985, -0.02);
    add(cyl(0.033, 0.022, 0.085, M("glass_clear"), 14), -0.90, 0.776, -0.02);
  }
  {
    const { add } = obj("bottle", "movable", -0.62, 0.75, -0.12, { mass_kg: 0.3 });
    add(cyl(0.033, 0.033, 0.19, M("glass_clear"), 12), -0.62, 0.7525, -0.12);
    add(cyl(0.016, 0.016, 0.025, M("plastic_blue"), 10), -0.62, 0.860, -0.12);
  }

  // ---------------- slippers ----------------
  {
    const { add } = obj("slipper_left", "movable", 0.06, 0.05, -0.94, { mass_kg: 0.2 });
    add(box(0.11, 0.025, 0.27, M("slipper_tan")), 0.06, 0.043, -0.94, 0, -0.25, 0);
    add(box(0.105, 0.045, 0.12, M("slipper_tan")), 0.085, 0.078, -0.985, 0, -0.25, 0);
  }
  {
    const { add } = obj("slipper_right", "movable", 0.235, 0.05, -0.905, { mass_kg: 0.2 });
    add(box(0.11, 0.025, 0.27, M("slipper_tan")), 0.235, 0.043, -0.905, 0, 0.18, 0);
    add(box(0.105, 0.045, 0.12, M("slipper_tan")), 0.212, 0.078, -0.952, 0, 0.18, 0);
  }

  // ---------------- rugs ----------------
  {
    const { add } = obj("rug_dark", "static", 0.14, 0.008, -0.325);
    add(box(1.32, 0.012, 1.55, M("rug_dark_green", [1.5, 2])), 0.14, 0.008, -0.325);
  }
  {
    const { add } = obj("rug_light", "static", -0.21, 0.022, -0.99);
    add(box(1.14, 0.012, 0.98, M("rug_light_blue", [1.2, 1])), -0.21, 0.022, -0.99);
  }
  {
    const { add } = obj("piano_rug", "static", 0.20, 0.007, 2.85);
    add(box(1.50, 0.010, 1.50, M("piano_rug_pattern", [1.5, 1.5])), 0.20, 0.0065, 2.85);
  }

  // ---------------- AV corner ----------------
  {
    const { add } = obj("subwoofer", "movable", 0.54, 0.222, -1.40, { mass_kg: 15 });
    add(box(0.46, 0.44, 0.46, M("speaker_black", [1, 1])), 0.54, 0.222, -1.40);
  }
  {
    const { add } = obj("tv", "movable", 1.10, 0.95, -0.23, { mass_kg: 18 });
    add(box(0.055, 0.82, 1.30, M("tv_black", [1, 1])), 1.10, 0.96, -0.23);
    add(box(0.16, 0.035, 0.05, M("metal_black")), 1.10, 0.525, -0.68);
    add(box(0.16, 0.035, 0.05, M("metal_black")), 1.10, 0.525, 0.22);
  }
  {
    const { add } = obj("tv_stand", "static", 1.14, 0.25, -0.225);
    add(box(0.32, 0.035, 1.15, M("glass_dark", [1, 3])), 1.14, 0.4875, -0.225); // glass top
    add(box(0.30, 0.03, 1.10, M("glass_dark", [1, 3])), 1.14, 0.135, -0.225); // lower shelf
    for (const [lx, lz] of [[1.02, -0.74], [1.26, -0.74], [1.02, 0.29], [1.26, 0.29]])
      add(cyl(0.02, 0.02, 0.47, M("metal_black"), 10), lx, 0.235, lz);
    add(box(0.22, 0.22, 0.36, M("speaker_black")), 1.15, 0.26, -0.17); // center speaker
    add(box(0.24, 0.14, 0.36, M("amp_silver")), 1.16, 0.22, -0.55); // amplifier
  }
  function speaker(id, cx, cz) {
    const { add } = obj(id, "movable", cx, 0.45, cz, { mass_kg: 12 });
    add(box(0.30, 0.03, 0.32, M("metal_black")), cx, 0.017, cz); // base
    add(box(0.09, 0.50, 0.10, M("metal_black")), cx, 0.282, cz); // column
    add(box(0.24, 0.36, 0.28, M("speaker_black", [1, 1])), cx, 0.712, cz); // speaker box
    add(cyl(0.065, 0.065, 0.015, M("cone_white"), 20), cx - 0.125, 0.66, cz, 0, 0, Math.PI / 2); // woofer
    add(cyl(0.022, 0.022, 0.015, M("cone_white"), 12), cx - 0.125, 0.80, cz, 0, 0, Math.PI / 2); // tweeter
  }
  speaker("speaker_left", 0.94, -1.32);
  speaker("speaker_right", 0.86, 0.46);

  // ---------------- plant ----------------
  {
    const { add } = obj("plant", "movable", 0.72, 0.6, 1.00, { mass_kg: 8 });
    add(cyl(0.20, 0.20, 0.025, M("terracotta"), 20), 0.72, 0.0145, 1.00); // saucer
    add(cyl(0.155, 0.12, 0.28, M("terracotta", [2, 1]), 20), 0.72, 0.168, 1.00); // pot
    add(cyl(0.14, 0.14, 0.02, M("hall_dark"), 16), 0.72, 0.30, 1.00); // soil
    add(cyl(0.028, 0.034, 0.75, M("table_wood"), 10), 0.72, 0.685, 1.00); // trunk
    for (let i = 0; i < 12; i++) {
      const a = (i / 12) * Math.PI * 2 + 0.3;
      const ring = i % 2;
      const tilt = -0.40 - (i % 3) * 0.18;
      const len = 0.36;
      const leaf = box(0.055, 0.015, len, M("plant_green", [1, 2]));
      const r = len / 2 - 0.06;
      const baseY = ring ? 1.10 : 0.82;
      add(leaf, 0.72 + Math.sin(a) * r * Math.cos(tilt), baseY + Math.sin(-tilt) * r * 0.5, 1.00 + Math.cos(a) * r * Math.cos(tilt), tilt, a, 0);
    }
  }

  // ---------------- piano area ----------------
  {
    const { add } = obj("piano", "static", 0.15, 0.8, 2.95);
    const pb = () => M("piano_black", [1, 1]);
    add(box(1.40, 0.14, 0.30, pb()), 0.15, 0.71, 2.38); // keybed
    add(box(1.30, 0.025, 0.14, M("keys_white", [4, 1])), 0.15, 0.755, 2.315); // keys
    add(box(1.40, 0.36, 0.77, pb()), 0.15, 0.82, 2.915); // rim/body
    add(box(0.90, 0.36, 0.30, pb()), 0.15, 0.82, 3.45); // tail
    // open lid: hinged along west edge (x=-0.55), propped up toward east
    add(box(1.42, 0.03, 1.50, pb()), 0.143, 1.170, 2.99, 0, 0, 0.22);
    add(box(0.03, 0.22, 0.03, pb()), 0.55, 1.13, 3.30, 0, 0, 0.25); // prop stick
    for (const [lx, lz] of [[-0.40, 2.46], [0.70, 2.46], [0.15, 3.50]])
      add(cyl(0.05, 0.04, 0.626, pb(), 12), lx, 0.327, lz); // legs
    add(box(0.04, 0.45, 0.04, pb()), 0.15, 0.36, 2.52); // pedal post
    add(box(0.25, 0.05, 0.10, pb()), 0.15, 0.12, 2.50); // pedals
    add(box(0.55, 0.35, 0.02, pb()), 0.15, 1.06, 2.63, -0.30, 0, 0); // music stand
    add(box(0.48, 0.30, 0.012, M("sheet_music")), 0.15, 1.075, 2.615, -0.30, 0, 0); // sheet music
  }
  {
    const { add } = obj("piano_bench", "movable", 0.53, 0.25, 2.30, { mass_kg: 8 });
    add(box(0.62, 0.07, 0.34, M("piano_black", [1, 1])), 0.53, 0.469, 2.30);
    for (const [lx, lz] of [[0.27, 2.17], [0.79, 2.17], [0.27, 2.43], [0.79, 2.43]])
      add(box(0.05, 0.42, 0.05, M("piano_black")), lx, 0.224, lz);
  }
  {
    const { add } = obj("cardboard_box", "movable", 1.12, 0.16, 2.62, { mass_kg: 3 });
    add(box(0.32, 0.32, 0.30, M("cardboard", [1, 1])), 1.12, 0.162, 2.62);
  }

  // ---------------- bookshelf (east wall, north of alcove) ----------------
  {
    const { add } = obj("bookshelf", "static", 1.16, 1.0, 1.73);
    const bw = () => M("bentwood", [1, 2]);
    add(box(0.26, 2.02, 0.04, bw()), 1.16, 1.01, 1.40); // sides
    add(box(0.26, 2.02, 0.04, bw()), 1.16, 1.01, 2.06);
    add(box(0.03, 2.02, 0.70, bw()), 1.285, 1.01, 1.73); // back
    add(box(0.26, 0.04, 0.70, bw()), 1.16, 0.04, 1.73); // shelves
    add(box(0.26, 0.035, 0.66, bw()), 1.16, 0.44, 1.73);
    add(box(0.26, 0.035, 0.66, bw()), 1.16, 0.84, 1.73);
    add(box(0.26, 0.035, 0.66, bw()), 1.16, 1.24, 1.73);
    add(box(0.26, 0.035, 0.66, bw()), 1.16, 1.64, 1.73);
    add(box(0.26, 0.04, 0.70, bw()), 1.16, 2.00, 1.73); // top
    // book rows
    add(box(0.20, 0.30, 0.56, M("book_spines", [2, 1])), 1.17, 0.61, 1.72);
    add(box(0.20, 0.32, 0.50, M("book_spines", [2, 1])), 1.17, 1.02, 1.68);
    add(box(0.20, 0.28, 0.58, M("book_spines", [2, 1])), 1.17, 1.40, 1.74);
    add(box(0.20, 0.30, 0.44, M("book_spines", [2, 1])), 1.17, 1.81, 1.62);
    add(box(0.18, 0.12, 0.24, M("book_spines")), 1.17, 1.74, 1.95);
    // globe + storage box on top
    add(sph(0.14, M("globe_tan", [2, 1])), 1.16, 2.18, 1.52);
    add(box(0.30, 0.26, 0.38, M("cardboard", [1, 1])), 1.16, 2.15, 1.88);
  }

  // ---------------- keyboard on stand (extra, south end) ----------------
  {
    const { add } = obj("keyboard", "movable", -0.925, 0.35, -1.62, { mass_kg: 8 });
    add(box(0.85, 0.08, 0.32, M("speaker_black")), -0.925, 0.64, -1.62);
    add(box(0.80, 0.02, 0.14, M("keys_white", [4, 1])), -0.925, 0.69, -1.575);
    add(box(0.04, 0.66, 0.05, M("metal_black")), -1.15, 0.31, -1.62, 0.45, 0, 0);
    add(box(0.04, 0.66, 0.05, M("metal_black")), -1.15, 0.31, -1.62, -0.45, 0, 0);
    add(box(0.04, 0.66, 0.05, M("metal_black")), -0.70, 0.31, -1.62, 0.45, 0, 0);
    add(box(0.04, 0.66, 0.05, M("metal_black")), -0.70, 0.31, -1.62, -0.45, 0, 0);
  }

  // ---------------- air purifier (extra) ----------------
  {
    const { add } = obj("air_purifier", "movable", -1.20, 0.58, 1.24, { mass_kg: 6 });
    add(cyl(0.14, 0.14, 1.15, M("purifier_white", [2, 2]), 20), -1.20, 0.577, 1.24);
    add(cyl(0.12, 0.14, 0.04, M("hall_dark"), 20), -1.20, 1.167, 1.24);
  }

  // ---------------- wall decor above TV (extra, static) ----------------
  {
    const { add } = obj("wall_decor", "static", 1.24, 1.6, -0.22);
    add(box(0.10, 0.035, 1.30, M("shelf_white")), 1.25, 1.415, -0.22); // picture ledge
    add(box(0.03, 0.35, 0.30, M("orchid_pink", [1, 1])), 1.245, 1.53, -0.62, 0, 0, -0.10); // framed print
    add(box(0.03, 0.32, 0.28, M("sheet_music", [1, 1])), 1.245, 1.52, 0.15, 0, 0, -0.10); // framed print
    add(cyl(0.16, 0.16, 0.025, M("clock_orange"), 24), 1.26, 1.58, -0.20, 0, 0, Math.PI / 2); // clock
  }

  return objs;
}

export function environment(THREE) {
  const hemi = new THREE.HemisphereLight(0xfdf3e3, 0x8a7a62, 1.25);
  const south = new THREE.DirectionalLight(0xfff0d8, 0.85);
  south.position.set(0.2, 2.0, -4.5);
  const north = new THREE.DirectionalLight(0xe9f0f4, 0.6);
  north.position.set(0.6, 2.2, 6.5);
  const west = new THREE.DirectionalLight(0xf5ead6, 0.35);
  west.position.set(-4.5, 2.5, 0.5);
  const east = new THREE.DirectionalLight(0xf2e8d4, 0.3);
  east.position.set(4.5, 2.5, 1.0);
  return { background: "#aeb9bd", lights: [hemi, south, north, west, east] };
}
