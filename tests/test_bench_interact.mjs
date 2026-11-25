// 两轴评测的交互清单：它判的每一条都必须是执行出来的结果。
// 冒烟场景里故意放了有毛病的物体，这里钉住它们各自被判成什么。
// 运行: node --test tests/test_bench_interact.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import { initRapier } from '../kernel/physics.js';
import { loadCandidate, runChecklist, selfChecklist } from '../harness/bench_interact.mjs';

await initRapier();
const candidate = await loadCandidate('examples/bench_smoke');
const checklist = JSON.parse(fs.readFileSync('examples/bench_smoke/checklist.json', 'utf8'));
const report = runChecklist(candidate, checklist);
const verdict = (target, expect) => report.items.find(i => i.target === target && i.expect === expect);

test('放稳的、能拖的、固定的都通过', () => {
  for (const [target, expect] of [['floor', 'fixed'], ['wall', 'fixed'], ['table', 'movable'], ['mug', 'movable'], ['mug', 'rests']]) {
    assert.equal(verdict(target, expect).pass, true, `${target} ${expect}: ${JSON.stringify(verdict(target, expect))}`);
  }
});

test('悬空的物体放开会掉，rests 不通过', () => {
  const v = verdict('floating_box', 'rests');
  assert.equal(v.pass, false);
  assert.equal(v.reason, 'moved_after_release');
  assert.ok(v.detail.dropped_m > 0.3, JSON.stringify(v.detail));
});

test('声明成搬不动的物体，movable 不通过；清单里有而场景里没有的物体算缺失', () => {
  assert.equal(verdict('sofa', 'movable').reason, 'declared_static');
  assert.equal(verdict('lamp', 'movable').reason, 'missing_object');
});

test('汇总数和逐条结果一致', () => {
  assert.deepEqual([report.summary.passed, report.summary.total], [5, 8]);
  assert.equal(report.summary.by_expect.fixed.passed, 2);
});

test('自检清单只覆盖声明成能动的物体', () => {
  const targets = new Set(selfChecklist(candidate).items.map(i => i.target));
  assert.deepEqual([...targets].sort(), ['floating_box', 'mug', 'table']);
});

test('桌子按部件拆碰撞体，不是一个实心方块', () => {
  assert.equal(report.objects.find(o => o.id === 'table').colliders, 5);
});

test('贴着墙的物体换个方向拖得动，就算能挪动', async () => {
  const os = await import('node:os');
  const path = await import('node:path');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'bench-wall-'));
  fs.writeFileSync(path.join(dir, 'package.json'), '{"type":"module"}');
  fs.writeFileSync(path.join(dir, 'materials.json'), JSON.stringify({ m: { prompt: 'x', seed: 1, base_color: '#888888' } }));
  fs.writeFileSync(path.join(dir, 'scene.js'), `
    export function describe(THREE, ctx) {
      const box = (w, h, d, x, y, z) => { const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), ctx.material('m')); m.position.set(x, y, z); return m; };
      return [
        { id: 'floor', kind: 'static', object3D: box(6, 0.2, 6, 0, -0.1, 0) },
        { id: 'wall_px', kind: 'static', object3D: box(0.2, 3, 6, 1.0, 1.5, 0) },   // 紧挨在 +x 一侧
        { id: 'crate', kind: 'movable', mass_kg: 5, object3D: box(0.4, 0.4, 0.4, 0.69, 0.201, 0) },
        // 四面和头顶都被封死的箱子：哪个方向都挪不动
        { id: 'cage', kind: 'static', object3D: (() => { const g = new THREE.Group();
            for (const [x, z] of [[0.26, 0], [-0.26, 0], [0, 0.26], [0, -0.26]]) g.add(box(x ? 0.1 : 0.62, 0.6, x ? 0.62 : 0.1, x, 0.3, z));
            g.add(box(0.62, 0.1, 0.62, 0, 0.65, 0)); g.position.set(-2, 0, -2); return g; })() },
        { id: 'trapped', kind: 'movable', object3D: box(0.4, 0.4, 0.4, -2, 0.201, -2) },
      ];
    }`);
  const cand = await loadCandidate(dir);
  const out = runChecklist(cand, { items: [{ target: 'crate', expect: 'movable' }, { target: 'trapped', expect: 'movable' }] });
  const crate = out.items[0], trapped = out.items[1];
  assert.equal(crate.pass, true, JSON.stringify(crate.detail));
  assert.notDeepEqual(crate.detail.attempt.direction, [1, 0, 0]);   // 朝墙那个方向是走不通的
  assert.equal(trapped.pass, false, JSON.stringify(trapped.detail));
  assert.equal(trapped.detail.attempts_tried, 8);
});

test('按位置找目标：两个标签落到同一个物体上，要动的那几条判 merged', () => {
  // 杯子的锚点故意放在桌面里：等于说场景把杯子和桌子写成了一个物体
  const anchors = { table: [0, 0.725, 0], mug: [0.1, 0.725, 0], floor: [0, -0.1, 0] };
  const out = runChecklist(candidate, { items: [{ target: 'floor', expect: 'fixed' }, { target: 'table', expect: 'movable' },
                                                { target: 'mug', expect: 'movable' }] }, anchors);
  const by = Object.fromEntries(out.items.map(i => [i.target, i]));
  assert.equal(by.floor.resolved, 'floor'); assert.equal(by.floor.pass, true);
  assert.equal(by.table.resolved, 'table'); assert.equal(by.mug.resolved, 'table');
  assert.equal(by.table.reason, 'merged'); assert.equal(by.mug.reason, 'merged');
});
