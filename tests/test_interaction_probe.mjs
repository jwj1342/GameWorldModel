// 可交互性探针自己的测试：它报的数必须能反映真实情况，不然就是又一个「声明值」。
// Node 20: node --experimental-default-type=module --test tests/test_interaction_probe.mjs
import assert from 'node:assert/strict';
import test from 'node:test';
import { initRapier } from '../kernel/physics.js';
import { probeProgram, styleSwapDeviation, succeeded } from '../harness/interaction_probe.mjs';

await initRapier();

const GROUND = { id: 'ground', class: 'ground',
  geom: { kind: 'primitive', shape: 'box', extent: [20, 0.4, 20] }, pose: { pos: [0, -0.2, 0] } };

function program(objects) {
  return { static: [GROUND], objects,
           camera: { intrinsics: { fov_deg: 50, aspect: 1.78 },
                     keyframes: [{ t: 0, pos: [0, 1.2, 2.5], quat: [-0.1, 0, 0, 0.995] }] } };
}

const cup = (id, pos) => ({ id, class: 'cup',
  geom: { kind: 'primitive', shape: 'box', extent: [0.3, 0.3, 0.3] }, pose: { pos } });

test('能拖的物体被判成能拖', () => {
  const report = probeProgram(program([cup('cup', [0, 0.15, 0])]));
  assert.equal(report.summary.declared_grabbable, 1);
  assert.equal(report.summary.works, 1, JSON.stringify(report.objects[0].drag));
  assert.equal(report.summary.manipulation_success_fraction, 1);
});

test('程序里关掉交互的物体不进统计', () => {
  const node = { ...cup('cup', [0, 0.15, 0]), interaction: { grab: false } };
  const report = probeProgram(program([node]));
  assert.equal(report.summary.declared_grabbable, 0);
  assert.equal(report.summary.manipulation_success_fraction, null);
});

test('抽屉拖到行程外面会被限位卡住', () => {
  const drawer = { id: 'drawer', class: 'drawer',
    geom: { kind: 'primitive', shape: 'box', extent: [0.4, 0.2, 0.3] }, pose: { pos: [0, 0.3, 0] },
    interaction: { grab: 'slide', axis: [0, 0, 1], limit: [0, 0.2] } };
  const report = probeProgram(program([drawer]));
  const d = report.objects[0].drag;
  assert.equal(d.mode, 'slide');
  assert.ok(d.moves, '抽屉应该拉得动');
  assert.ok(d.overshoot <= 0.02, `不该冲出行程，实际超出 ${d.overshoot} m`);
  assert.ok(d.off_axis_drift <= 0.02, `不该偏离滑轨，实际偏 ${d.off_axis_drift} m`);
  assert.equal(succeeded(report.objects[0]), true);
});

test('柜门绕铰链转，并且停在开合角度上', () => {
  const door = { id: 'door', class: 'door',
    geom: { kind: 'primitive', shape: 'box', extent: [0.5, 0.8, 0.05] }, pose: { pos: [0.25, 0.4, 0] },
    interaction: { grab: 'hinge', axis: [0, 1, 0], pivot: [0, 0.4, 0], limit: [0, 1.57] } };
  const report = probeProgram(program([door]));
  const d = report.objects[0].drag;
  assert.equal(d.mode, 'hinge');
  assert.ok(d.moves, '柜门应该推得开');
  assert.ok(d.overshoot_deg <= 2.0, `不该转过开合角，实际超出 ${d.overshoot_deg} 度`);
});

test('画面里被挡住的物体，点得到这一项要报 false，但不影响它能不能拖', () => {
  // 相机在 +z，挡板横在物体前面
  const wall = { id: 'wall', class: 'panel',
    geom: { kind: 'primitive', shape: 'box', extent: [4, 4, 0.1] }, pose: { pos: [0, 1.0, 1.2] },
    interaction: { grab: false } };
  const report = probeProgram(program([wall, cup('cup', [0, 0.15, 0])]));
  const row = report.objects.find(r => r.id === 'cup');
  assert.equal(row.reach_info.on_screen, true);
  assert.equal(row.reach_info.clickable, false, '应该被挡板挡住');
  assert.equal(succeeded(row), true, '挡住不等于拖不动');
});

test('换一套完全不同的外形网格，轨迹一丝不差', () => {
  // 这是论文第四条主张的检查点：碰撞体从基本体来，外形换掉不动行为。
  const p = program([cup('a', [-0.4, 0.15, 0]), cup('b', [0.4, 0.15, 0])]);
  const dev = styleSwapDeviation(p);
  assert.ok(dev.compared_steps > 100, `应该比了不少步，实际 ${dev.compared_steps}`);
  assert.equal(dev.steps_plain, dev.steps_swapped, '两边步数应该一样');
  assert.equal(dev.max_position_deviation, 0,
    `外形换掉不该改变轨迹，实际最大偏离 ${dev.max_position_deviation} m`);
});

test('互相穿插着生成的物体，拖一个会把别人带飞，这要被记下来', () => {
  const a = cup('a', [0, 0.15, 0]);
  const b = cup('b', [0.05, 0.15, 0]);        // 和 a 重叠 83%
  const report = probeProgram(program([a, b]));
  const worst = Math.max(...report.objects.map(r => r.drag.max_disturbance));
  assert.ok(worst > 0.02, `穿插的场景应该量到扰动，实际 ${worst} m`);
});
