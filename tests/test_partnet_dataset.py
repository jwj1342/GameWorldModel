import copy
import pytest
from gwm.dataset.partnet import generate_partnet_scene
from gwm.compiler.validate import validate


def source():
    return {'meta': {'clip': 'source', 'duration': 1, 'fps': 30, 'notes': 'private source fact'},
            'camera': {'intrinsics': {'fov_deg': 45}, 'keyframes': [{'t': 0, 'pos': [3, 3, 3], 'look_at': [0, 0, 0]}]},
            'static': [], 'objects': [
                {'id': 'body', 'class': 'partnet_body', 'geom': {'kind': 'asset', 'query': 'body', 'ref': 'body', 'extent': [1, 1, 1]},
                 'pose': {'pos': [0, 0, 0], 'quat': [0, 0, 0, 1]}},
                {'id': 'lid', 'class': 'partnet_lid', 'geom': {'kind': 'asset', 'query': 'lid', 'ref': 'lid', 'extent': [1, 0.1, 1]},
                 'pose': {'pos': [0, 1, 0], 'quat': [0, 0, 0, 1]},
                 'motion': {'type': 'revolute', 'axis': [0, 0, 1], 'pivot': [0, 1, 0], 'from_deg': -33,
                            'range_deg': [-33, 144], 'schedule': [{'t': 0, 'to_deg': 144, 'duration': 1}]}}]}


def behavior(program):
    result = copy.deepcopy(program)
    result.pop('style', None)
    result['meta'].pop('clip', None)
    result['meta'].pop('notes', None)
    for node in result['objects']:
        node.pop('material', None)
    return result


def test_pair_preserves_all_behavior_and_source():
    p = source()
    original = copy.deepcopy(p)
    a = generate_partnet_scene(p, clip='a', style_seed=1)
    b = generate_partnet_scene(p, clip='b', style_seed=2)
    assert p == original
    assert behavior(a) == behavior(b) == behavior(p)
    assert a['style'] != b['style']
    assert 'notes' not in a['meta']
    assert validate(a)['ok'] and validate(b)['ok']
    assert all('opacity' not in n['material'] for n in a['objects'])
    assert all(set(n['material']) == {'base_color'} for n in a['objects'])
    assert generate_partnet_scene(p, clip='a', style_seed=1) == a


@pytest.mark.parametrize('change', [
    lambda p: p['objects'][0]['geom'].update(kind='primitive'),
    lambda p: p['objects'][1]['geom'].pop('ref'),
    lambda p: p['objects'][1]['motion'].update(type='prismatic'),
    lambda p: p['objects'][0].update(motion={'type': 'spin', 'speed_deg_s': 1}),
    lambda p: p['objects'].pop(),
])
def test_unsupported_input_fails_without_fallback(change):
    p = source()
    change(p)
    with pytest.raises(ValueError):
        generate_partnet_scene(p, clip='a', style_seed=1)


def test_invalid_seed_and_clip_rejected():
    for kwargs in ({'clip': '', 'style_seed': 1}, {'clip': 'a', 'style_seed': True}):
        with pytest.raises(ValueError):
            generate_partnet_scene(source(), **kwargs)


def imported_fixture(tmp_path):
    import json
    import numpy as np
    from gwm.compiler.partnet import _glb
    root = tmp_path / 'imported'
    root.mkdir()
    # Small synthetic GLBs test the file pipeline; the real asset is tested separately.
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=int)
    requests = []
    for name in ('body', 'lid'):
        (root / (name + '.glb')).write_bytes(_glb(vertices, faces, [0.2, 0.4, 0.6]))
        requests.append({'id': name, 'resolved': 'library_ref', 'asset_id': name,
                         'license': 'PartNet-Mobility research terms', 'source': 'unused'})
    for filename, value in [('program.json', source()), ('asset_manifest.json', {'requests': requests}),
                            ('source_mapping.json', {'repository': 'sapien-sim/PartNetMobility',
                             'revision': 'fixture', 'source_archive_sha256': '1' * 64})]:
        (root / filename).write_text(json.dumps(value))
    return root


def test_cli_copies_assets_and_preserves_source_and_license(tmp_path, monkeypatch):
    import hashlib
    import json
    from scripts.partnet.generate_dataset import main
    imported = imported_fixture(tmp_path)
    before = {p.name: p.read_bytes() for p in imported.iterdir()}
    out = tmp_path / 'dataset'
    monkeypatch.setattr('sys.argv', ['generate', str(imported), str(out)])
    assert main() == 0
    assert {p.name: p.read_bytes() for p in imported.iterdir()} == before
    manifest = json.loads((out / 'manifest.json').read_text())
    assert len(manifest['samples']) == 2 and len(manifest['pairs']) == 1
    for sample in manifest['samples']:
        root = out / sample['clip']
        assets = json.loads((root / 'asset_manifest.json').read_text())
        for request in assets['requests']:
            path = root / request['file']
            assert path.read_bytes() == before[request['id'] + '.glb']
            assert request['license'] == 'PartNet-Mobility research terms'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['assets'][request['id']]
        assert validate(json.loads((root / 'program.json').read_text()))['ok']
    marker = out / 'keep'
    marker.write_text('preserve')
    with pytest.raises(FileExistsError):
        main()
    assert marker.read_text() == 'preserve'


def test_missing_asset_or_ref_mismatch_fails_before_output(tmp_path, monkeypatch):
    import json
    from scripts.partnet.generate_dataset import main
    imported = imported_fixture(tmp_path)
    out = tmp_path / 'dataset'
    monkeypatch.setattr('sys.argv', ['generate', str(imported), str(out)])
    (imported / 'lid.glb').unlink()
    with pytest.raises(ValueError):
        main()
    assert not out.exists()
    (imported / 'lid.glb').write_bytes((imported / 'body.glb').read_bytes())
    p = json.loads((imported / 'program.json').read_text())
    p['objects'][1]['geom']['ref'] = 'mismatch'
    (imported / 'program.json').write_text(json.dumps(p))
    with pytest.raises(ValueError):
        main()
    assert not out.exists()
