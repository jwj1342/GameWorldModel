// Isolate render visibility only: never seek, step, remove bodies or rewrite poses.
export function renderUnoccludedId(registryIndex, game = globalThis.window.__game) {
  const target = game.registry.find(e => e.idIndex === registryIndex);
  if (!target) throw new Error('unknown registry index for visibility reference');
  const previous = game.registry.map(e => [e.group, e.group.visible]);
  try {
    for (const e of game.registry) e.group.visible = e === target && e.group.visible;
    return game.render('id');
  } finally {
    for (const [group, visible] of previous) group.visible = visible;
  }
}
