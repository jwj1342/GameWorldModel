import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('render_blender', Path(__file__).parents[1] / 'scripts/cross_engine/render_blender.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def packet():
    matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 3, 4, 1]
    return {'format': 'gwm-frozen-scene', 'schema_version': 1, 'width': 320, 'height': 240,
            'frames': [{'t': 1, 'requested_t': 1, 'camera': {'matrix_world': matrix, 'fov': 60, 'aspect': 4/3},
                        'meshes': [{'key': 'box/0', 'matrix_world': matrix,
                                    'vertices': [[0, 0, 0], [1, 0, 0], [0, 1, 0]], 'triangles': [[0, 1, 2]]}]}]}


def test_matrix_convention_and_input_unchanged():
    p = packet()
    before = copy.deepcopy(p)
    module.validate_packet(p)
    rows = module.matrix_rows(p['frames'][0]['camera']['matrix_world'])
    assert [row[3] for row in rows] == [2, 3, 4, 1]
    assert p == before


@pytest.mark.parametrize('fault', ['time', 'duplicate', 'triangle', 'nan', 'camera', 'identity'])
def test_bad_packet_rejected(fault):
    p = packet()
    f = p['frames'][0]
    if fault == 'time': f['requested_t'] = 2
    if fault == 'duplicate': f['meshes'] *= 2
    if fault == 'triangle': f['meshes'][0]['triangles'] = [[0, 1, 9]]
    if fault == 'nan': f['meshes'][0]['vertices'][0][0] = float('nan')
    if fault == 'camera': f['camera']['aspect'] = 1
    if fault == 'identity':
        p['frames'].append(copy.deepcopy(f))
        p['frames'][1]['meshes'][0]['key'] = 'other/0'
    with pytest.raises(ValueError):
        module.validate_packet(p)
