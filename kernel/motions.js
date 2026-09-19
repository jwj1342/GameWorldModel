// motions.js — 每个 motion 节点编译成纯函数 pose(t) -> {pos: Vector3, quat: Quaternion}
// 词表：static, trajectory, revolute, prismatic, periodic_translate, periodic_rotate, spin
import * as THREE from 'three';

const DEG = Math.PI / 180;
const v3 = (a) => new THREE.Vector3(a[0], a[1], a[2]);
const q4 = (a) => (a && a.length === 4) ? new THREE.Quaternion(a[0], a[1], a[2], a[3]).normalize() : new THREE.Quaternion();

function catmullRom(points, times, t) {
  // points: Vector3[], times: ascending. Returns interpolated Vector3.
  const n = points.length;
  if (n === 1) return points[0].clone();
  if (t <= times[0]) return points[0].clone();
  if (t >= times[n - 1]) return points[n - 1].clone();
  let i = 0; while (i < n - 2 && t > times[i + 1]) i++;
  const p0 = points[Math.max(i - 1, 0)], p1 = points[i], p2 = points[i + 1], p3 = points[Math.min(i + 2, n - 1)];
  const u = (t - times[i]) / Math.max(times[i + 1] - times[i], 1e-6);
  const curve = new THREE.CatmullRomCurve3([p0, p1, p2, p3]);
  // map u in [0,1] between p1 and p2, which is segment 1 of 3 in the 4-point curve
  return curve.getPoint((1 + u) / 3);
}

function lerpVec(points, times, t) {
  const n = points.length;
  if (t <= times[0]) return points[0].clone();
  if (t >= times[n - 1]) return points[n - 1].clone();
  let i = 0; while (i < n - 2 && t > times[i + 1]) i++;
  const u = (t - times[i]) / Math.max(times[i + 1] - times[i], 1e-6);
  return points[i].clone().lerp(points[i + 1], u);
}

// schedule: [{t, to, duration?}] piecewise-linear with holds. value before first entry = from.
function scheduleValue(schedule, from, t, defaultDuration = 1.0) {
  let value = from, prev = from;
  for (const s of schedule) {
    const d = s.duration ?? defaultDuration;
    if (t < s.t) return value;
    if (t < s.t + d) return prev + (s.to - prev) * ((t - s.t) / d);
    value = s.to; prev = s.to;
  }
  return value;
}

export function makePose(motion, base) {
  // base: {pos:[x,y,z], quat:[x,y,z,w]}
  const basePos = v3(base.pos), baseQuat = q4(base.quat);
  const type = motion?.type ?? 'static';
  const ident = () => ({ pos: basePos.clone(), quat: baseQuat.clone() });
  switch (type) {
    case 'static': return ident;
    case 'trajectory': {
      const kf = motion.keyframes ?? [];
      if (!kf.length) return ident;
      const times = kf.map(k => k.t), pts = kf.map(k => v3(k.pos));
      const quats = kf.map(k => k.quat ? q4(k.quat) : baseQuat.clone());
      const interp = motion.interp === 'catmull_rom' && kf.length >= 2 ? catmullRom : lerpVec;
      return (t) => {
        const pos = interp(pts, times, t);
        // quaternion: slerp between neighbours
        let i = 0; while (i < kf.length - 2 && t > times[i + 1]) i++;
        const u = kf.length > 1 ? THREE.MathUtils.clamp((t - times[i]) / Math.max(times[i + 1] - times[i], 1e-6), 0, 1) : 0;
        const quat = kf.length > 1 ? quats[i].clone().slerp(quats[Math.min(i + 1, kf.length - 1)], u) : quats[0].clone();
        return { pos, quat };
      };
    }
    case 'revolute': {
      const axis = v3(motion.axis ?? [0, 1, 0]).normalize();
      const pivot = motion.pivot ? v3(motion.pivot) : basePos.clone();
      const range = motion.range_deg ?? [0, 90];
      const from = motion.from_deg ?? range[0];
      const angleAt = motion.schedule?.length
        ? (t) => scheduleValue(motion.schedule.map(s => ({ t: s.t, to: s.to_deg, duration: s.duration })), from, t)
        : (t) => from + (motion.rate_dps ?? 0) * t;
      return (t) => {
        const a = angleAt(t) * DEG;
        const rot = new THREE.Quaternion().setFromAxisAngle(axis, a);
        const pos = basePos.clone().sub(pivot).applyQuaternion(rot).add(pivot);
        return { pos, quat: rot.multiply(baseQuat) };
      };
    }
    case 'prismatic': {
      const axis = v3(motion.axis ?? [1, 0, 0]).normalize();
      const from = motion.from ?? 0;
      const distAt = motion.schedule?.length
        ? (t) => scheduleValue(motion.schedule, from, t)
        : (t) => from + (motion.rate ?? 0) * t;
      return (t) => ({ pos: basePos.clone().addScaledVector(axis, distAt(t)), quat: baseQuat.clone() });
    }
    case 'periodic_translate': {
      const axis = v3(motion.axis ?? [0, 1, 0]).normalize();
      const amp = motion.amp ?? 1, period = Math.max(motion.period ?? 2, 1e-3), phase = motion.phase ?? 0;
      return (t) => ({ pos: basePos.clone().addScaledVector(axis, amp * Math.sin(2 * Math.PI * t / period + phase)), quat: baseQuat.clone() });
    }
    case 'periodic_rotate': {
      const axis = v3(motion.axis ?? [0, 1, 0]).normalize();
      const amp = (motion.amp_deg ?? 45) * DEG, period = Math.max(motion.period ?? 2, 1e-3), phase = motion.phase ?? 0;
      const pivot = motion.pivot ? v3(motion.pivot) : null;
      return (t) => {
        const rot = new THREE.Quaternion().setFromAxisAngle(axis, amp * Math.sin(2 * Math.PI * t / period + phase));
        const pos = pivot ? basePos.clone().sub(pivot).applyQuaternion(rot).add(pivot) : basePos.clone();
        return { pos, quat: rot.multiply(baseQuat) };
      };
    }
    case 'spin': {
      const axis = v3(motion.axis ?? [0, 1, 0]).normalize();
      const rate = (motion.rate_dps ?? 90) * DEG;
      return (t) => ({ pos: basePos.clone(), quat: new THREE.Quaternion().setFromAxisAngle(axis, rate * t).multiply(baseQuat) });
    }
    default:
      console.warn('[motions] unknown motion type', type, '-> static');
      return ident;
  }
}

export const MOTION_TYPES = ['static', 'trajectory', 'revolute', 'prismatic', 'periodic_translate', 'periodic_rotate', 'spin'];
