"""Generate an opaque style pair from one private verified importer output.

python -m scripts.partnet.generate_dataset IMPORTED NEW_OUT --style-seeds 1 2
No geometry randomization, language labels, rendering or primitive fallback.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from gwm.dataset.partnet import generate_partnet_scene


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('imported', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--style-seeds', type=int, nargs='+', default=[1, 2])
    args = parser.parse_args()
    if len(args.style_seeds) < 2 or len(set(args.style_seeds)) != len(args.style_seeds):
        parser.error('at least two distinct style seeds required')
    source = args.imported.resolve()
    program = json.loads((source / 'program.json').read_text(encoding='utf-8'))
    manifest = json.loads((source / 'asset_manifest.json').read_text(encoding='utf-8'))
    mapping = json.loads((source / 'source_mapping.json').read_text(encoding='utf-8'))
    if mapping.get('repository') != 'sapien-sim/PartNetMobility' or not mapping.get('revision') or len(mapping.get('source_archive_sha256', '')) != 64:
        raise ValueError('missing PartNet source identity')
    requests = manifest['requests']
    if len(requests) != 2 or {r['id'] for r in requests} != {'body', 'lid'}:
        raise ValueError('requires two explicit source assets')
    assets = {}
    for request in requests:
        if request.get('resolved') != 'library_ref' or not request.get('license') or 'cc0' in request['license'].lower():
            raise ValueError('explicit PartNet license and resolution required')
        file = source / (request['id'] + '.glb')
        if file.is_symlink() or not file.is_file() or not file.read_bytes().startswith(b'glTF'):
            raise ValueError('source GLB missing or invalid')
        node = next(n for n in program['objects'] if n['id'] == request['id'])
        if node['geom'].get('ref') != request['asset_id']:
            raise ValueError('asset ref mismatch')
        assets[request['id']] = {'path': file, 'sha256': sha(file)}
    generated = [generate_partnet_scene(program, clip=f'partnet_style_{i:03d}', style_seed=s)
                 for i, s in enumerate(args.style_seeds)]
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed, p in zip(args.style_seeds, generated):
        folder = args.out / p['meta']['clip']
        folder.mkdir()
        (folder / 'assets').mkdir()
        rewritten = json.loads(json.dumps(manifest))
        for request in rewritten['requests']:
            asset = assets[request['id']]
            destination = folder / 'assets' / asset['path'].name
            shutil.copyfile(asset['path'], destination)
            assert sha(destination) == asset['sha256']
            request['source'] = str(destination.resolve())
            request['file'] = 'assets/' + destination.name
        for filename, obj in [('program.json', p), ('asset_manifest.json', rewritten)]:
            (folder / filename).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        rows.append({'clip': p['meta']['clip'], 'style_seed': seed, 'program_sha256': sha(folder / 'program.json')})
    facts = {'source_program_sha256': sha(source / 'program.json'),
             'source_asset_manifest_sha256': sha(source / 'asset_manifest.json'),
             'source_mapping_sha256': sha(source / 'source_mapping.json'),
             'source_archive_sha256': mapping['source_archive_sha256'], 'revision': mapping['revision'],
             'assets': {k:v['sha256'] for k,v in assets.items()}, 'samples': rows,
             'pairs': [[rows[0]['clip'], row['clip']] for row in rows[1:]],
             'variation': 'style only; source camera/geometry/poses/motion retained',
             'limits': ['single fixed-root revolute subset', 'mechanical endpoints, not semantic labels',
                        'no rendering, textures, independent physics or language generation']}
    (args.out / 'manifest.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding='utf-8')
    assert all(sha(v['path']) == v['sha256'] for v in assets.values())
    print(json.dumps({'samples': len(rows), 'out': str(args.out), 'rendered': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
