// replay_camera.js — 回放模式相机：按 program.camera.keyframes 做 Catmull-Rom / 线性插值
import * as THREE from 'three';

export function makeReplayCamera(camSpec, aspect) {
  const fov = camSpec?.intrinsics?.fov_deg ?? 60;
  const camera = new THREE.PerspectiveCamera(fov, aspect, 0.05, 200);
  const kf = (camSpec?.keyframes ?? []).slice().sort((a, b) => a.t - b.t);
  const times = kf.map(k => k.t);
  const pts = kf.map(k => new THREE.Vector3(...k.pos));
  const quats = kf.map(k => k.quat ? new THREE.Quaternion(...k.quat).normalize() : null);
  const lookAts = kf.map(k => k.look_at ? new THREE.Vector3(...k.look_at) : null);
  const curve = pts.length >= 2 ? new THREE.CatmullRomCurve3(pts, false, 'centripetal') : null;
  function poseAt(t) {
    if (!kf.length) return { pos: new THREE.Vector3(0, 3, 8), quat: null, look: new THREE.Vector3(0, 0, 0) };
    if (kf.length === 1 || t <= times[0]) return { pos: pts[0].clone(), quat: quats[0], look: lookAts[0] };
    if (t >= times[times.length - 1]) { const n = kf.length - 1; return { pos: pts[n].clone(), quat: quats[n], look: lookAts[n] }; }
    let i = 0; while (i < kf.length - 2 && t > times[i + 1]) i++;
    const u = (t - times[i]) / Math.max(times[i + 1] - times[i], 1e-6);
    const segU = (i + u) / (kf.length - 1);
    const pos = camSpec.interp === 'linear' || !curve ? pts[i].clone().lerp(pts[i + 1], u) : curve.getPoint(segU);
    const quat = quats[i] && quats[i + 1] ? quats[i].clone().slerp(quats[i + 1], u) : (quats[i] ?? quats[i + 1]);
    const look = lookAts[i] && lookAts[i + 1] ? lookAts[i].clone().lerp(lookAts[i + 1], u) : (lookAts[i] ?? lookAts[i + 1]);
    return { pos, quat, look };
  }
  return {
    camera,
    update(t) {
      const p = poseAt(t);
      camera.position.copy(p.pos);
      if (p.quat) camera.quaternion.copy(p.quat); else camera.lookAt(p.look ?? new THREE.Vector3(0, 0, 0));
      camera.updateMatrixWorld();
    },
  };
}
