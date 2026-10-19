import assert from 'node:assert/strict';
import test from 'node:test';
import { renderUnoccludedId } from '../harness/visibility.mjs';

test('isolation restores visibility, keeps simulation and distinguishes instances', () => {
  const groups = [true, true, false].map(visible => ({visible}));
  const game = {registry: groups.map((group, i) => ({idIndex: i+1, group})),
    state: {t: 2, positions: [1,2,3]},
    render(pass) {assert.equal(pass, 'id'); assert.deepEqual(groups.map(g => g.visible), [false,true,false]); return 'png';}};
  const before = JSON.stringify(game.state);
  assert.equal(renderUnoccludedId(2, game), 'png');
  assert.deepEqual(groups.map(g => g.visible), [true,true,false]);
  assert.equal(JSON.stringify(game.state), before);
});
test('failure restores hidden and visible entries', () => {
  const game = {registry: [{idIndex:1, group:{visible:true}}, {idIndex:2, group:{visible:false}}],
    render() {throw new Error('injected');}};
  assert.throws(() => renderUnoccludedId(1, game), /injected/);
  assert.deepEqual(game.registry.map(e=>e.group.visible), [true,false]);
  assert.throws(() => renderUnoccludedId(99, game), /unknown registry/);
});
