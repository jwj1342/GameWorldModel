// Runs in either a browser page or the Node synthetic tests. No external imports.
export function freezeFrame({ width, height, allowAssets = false }) {
  if (typeof allowAssets !== 'boolean') throw new Error('allowAssets must be boolean');
  const g = window.__game;
  if (!g.sceneOnly || g.template) throw new Error('frozen export requires scene-only mode');
  if (g.registry.some(e => e.simulated)) throw new Error('dynamic physics is outside the shared-state subset');
  const camera = g.cameraInfo();
  if (Math.abs(camera.aspect - width / height) > 1e-9) throw new Error('camera aspect mismatch');
  const root = g.registry[0]?.group;
  if (!root) throw new Error('empty scene');
  const cameraMatrix = root.matrixWorld.clone().compose(
    root.position.clone().fromArray(camera.pos),
    root.quaternion.clone().fromArray(camera.quat), root.scale.clone().set(1, 1, 1));
  const inverseCamera = cameraMatrix.clone().invert();
  const meshes = [];
  g.scene.updateMatrixWorld(true);
  const assetCount = new Set(g.registry.filter(e => e.spec.geom?.kind === 'asset').map(e => e.id)).size;
  for (const entry of g.registry) {
    const kind = entry.spec.geom?.kind;
    const loadedAsset = allowAssets && kind === 'asset' && entry.spec.geom.ref
      && g.assets?.loaded === assetCount && g.assets?.failures?.length === 0;
    if ((kind !== 'primitive' && !loadedAsset) || entry.events.length) {
      throw new Error(`unsupported or unverified scene entry: ${entry.name}`);
    }
    let ordinal = 0;
    entry.group.traverse(mesh => {
      if (!mesh.isMesh) return;
      const key = `${entry.name}/${ordinal++}`;
      const geo = mesh.geometry, mat = mesh.material;
      if (mesh.isSkinnedMesh || mesh.isInstancedMesh || Object.keys(geo.morphAttributes).length
          || Array.isArray(mat) || mat.map || mat.normalMap || mat.alphaMap
          || mat.opacity !== 1 || mat.transparent || (mat.side !== 0 && !(loadedAsset && mat.side === 2)) || mat.vertexColors
          || mat.emissive?.toArray().some(v => v !== 0)
          || geo.drawRange.start !== 0 || geo.drawRange.count !== Infinity) {
        throw new Error(`unsupported mesh/material: ${key}`);
      }
      const pos = geo.getAttribute('position');
      const vertices = Array.from({ length: pos.count }, (_, i) => [pos.getX(i), pos.getY(i), pos.getZ(i)]);
      const indices = geo.index ? Array.from(geo.index.array) : vertices.map((_, i) => i);
      if (indices.length % 3) throw new Error(`non-triangle mesh: ${key}`);
      const triangles = Array.from({ length: indices.length / 3 }, (_, i) => indices.slice(i * 3, i * 3 + 3));
      let visible = true;
      for (let p = mesh; p; p = p.parent) visible = visible && p.visible;
      // Off-centre probes detect transpose, axis and camera-FOV mistakes.
      const probes = vertices.filter((_, i) => i < 4).map(v => {
        const world = root.position.clone().fromArray(v).applyMatrix4(mesh.matrixWorld);
        const local = world.clone().applyMatrix4(inverseCamera);
        const tan = Math.tan(camera.fov * Math.PI / 360);
        if (Math.abs(local.z) < 1e-9) throw new Error(`projection probe on camera plane: ${key}`);
        return { world: world.toArray(), ndc: [local.x / (-local.z * tan * camera.aspect), local.y / (-local.z * tan)] };
      });
      meshes.push({ key, object_id: entry.name, vertices, triangles, visible,
        matrix_world: mesh.matrixWorld.toArray(), probes,
        material: { color: mat.color.toArray(), roughness: mat.roughness, metalness: mat.metalness,
          double_sided: mat.side === 2 } });
    });
  }
  return { t: g.t, state: g.state(), camera: { ...camera, matrix_world: cameraMatrix.toArray() }, meshes };
}
