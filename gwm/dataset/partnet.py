"""Style-paired generation from the verified, flat single-hinge importer subset."""
from copy import deepcopy
import random

from gwm.compiler.validate import validate
from .style import sample_style


def generate_partnet_scene(source: dict, *, clip: str, style_seed: int) -> dict:
    """Return a new Program; retain source geometry, camera, poses and motion.

    This consumes an importer-produced Program, not an asset retrieval result.
    Provenance and mesh hashes belong in the dataset sidecar, not model inputs.
    """
    if not isinstance(clip, str) or not clip.strip():
        raise ValueError('clip must be nonempty')
    if type(style_seed) is not int:
        raise ValueError('style_seed must be an integer')
    objects = source.get('objects', [])
    if len(objects) != 2 or {o.get('id') for o in objects} != {'body', 'lid'}:
        raise ValueError('requires the verified body/lid single-hinge subset')
    for node in objects:
        if node.get('geom', {}).get('kind') != 'asset' or not node['geom'].get('ref'):
            raise ValueError('real explicit asset refs required; no primitive fallback')
    if next(o for o in objects if o['id'] == 'lid').get('motion', {}).get('type') != 'revolute':
        raise ValueError('requires imported revolute motion')
    if next(o for o in objects if o['id'] == 'body').get('motion', {}).get('type', 'static') != 'static':
        raise ValueError('moving parent unsupported')
    if not validate(source)['ok']:
        raise ValueError('invalid source Program')
    program = deepcopy(source)
    program['meta']['clip'] = clip
    program['meta'].pop('notes', None)
    program['style'] = sample_style(random.Random(style_seed))
    for node in program['objects']:
        rng = random.Random(f'{style_seed}:{node["id"]}')
        node['material'] = {'base_color': '#%02x%02x%02x' % tuple(rng.randrange(24, 232) for _ in range(3)),
                            'roughness': round(rng.uniform(0.2, 0.9), 3)}
    if not validate(program)['ok']:
        raise ValueError('generated Program failed validation')
    return program
