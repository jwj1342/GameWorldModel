import copy
import hashlib
import json
import math
import struct
import zipfile

import numpy as np
import pytest

from gwm.compiler.partnet import import_single_hinge, _obj
from gwm.compiler.validate import validate

OBJ = 'v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n'
URDF = '''<robot name="fixture">
<link name="root"/><link name="body"><visual><origin xyz="0 0 0"/><geometry><mesh filename="mesh.obj"/></geometry></visual></link>
<link name="lid"><visual><origin xyz="0.2 0 0"/><geometry><mesh filename="mesh.obj" scale="1 2 1"/></geometry></visual></link>
<joint name="fixed" type="fixed"><parent link="root"/><child link="body"/><origin xyz="2 0 0" rpy="0 1.5707963267948966 0"/></joint>
<joint name="hinge" type="revolute"><parent link="body"/><child link="lid"/><origin xyz="1 0 0"/><axis xyz="0 0 1"/><limit lower="0" upper="1.5707963267948966"/></joint>
</robot>'''


def source(tmp_path, urdf=URDF, obj=OBJ):
    file = tmp_path/'source.zip'
    with zipfile.ZipFile(file,'w') as z:
        z.writestr('fixture/mobility.urdf',urdf)
        z.writestr('fixture/mesh.obj',obj)
    return file,hashlib.sha256(file.read_bytes()).hexdigest()


def run(file, digest, out):
    return import_single_hinge(file,out,expected_sha256=digest,revision='fixture-revision')


def test_fixed_parent_transform_pivot_axis_and_program_validity(tmp_path):
    file,digest = source(tmp_path)
    before = file.read_bytes()
    p,manifest,mapping = run(file,digest,tmp_path/'out')
    assert validate(p)['ok']
    lid = next(n for n in p['objects'] if n['id']=='lid')
    assert np.allclose(lid['motion']['pivot'],[2,0,-1])
    assert np.allclose(lid['motion']['axis'],[1,0,0])
    assert np.allclose(lid['pose']['pos'],[2.5,1,-1.7])
    # At 90 degrees, the centre rotates about world +X, retaining its X value.
    assert np.allclose(mapping['states'][-1]['lid_pos'],[2.5,0.7,0])
    assert all(r['license']!='CC0' for r in manifest['requests'])
    assert file.read_bytes()==before


def test_glb_preserves_centered_vertices_and_triangle_count(tmp_path):
    file,digest = source(tmp_path)
    p,_,_ = run(file,digest,tmp_path/'out')
    raw = (tmp_path/'out/lid.glb').read_bytes()
    magic,version,size=struct.unpack_from('<4sII',raw)
    assert (magic,version,size)==(b'glTF',2,len(raw))
    n=struct.unpack_from('<I',raw,12)[0]
    gltf=json.loads(raw[20:20+n])
    assert gltf['accessors'][0]['count']==4
    assert gltf['accessors'][2]['count']==12
    offset=20+n+8
    vertices=np.frombuffer(raw,dtype='<f4',count=12,offset=offset).reshape(-1,3)
    centre=next(x for x in p['objects'] if x['id']=='lid')['pose']['pos']
    world=vertices+centre
    assert np.allclose(world,[[2,0,-1.2],[2,0,-2.2],[2,2,-1.2],[3,0,-1.2]])


@pytest.mark.parametrize('change',[
    lambda s:s.replace('type="revolute"','type="continuous"'),
    lambda s:s.replace('xyz="0 0 1"','xyz="0 0 0"'),
    lambda s:s.replace('upper="1.5707963267948966"','upper="-1"'),
    lambda s:s.replace('filename="mesh.obj"','filename="../outside.obj"'),
    lambda s:s.replace('filename="mesh.obj"','filename="missing.obj"'),
])
def test_unsupported_or_bad_assets_fail_without_fallback(tmp_path,change):
    file,digest=source(tmp_path,change(URDF))
    with pytest.raises((ValueError,KeyError)):
        run(file,digest,tmp_path/'out')


def test_wrong_hash_and_existing_output_rejected(tmp_path):
    file,digest=source(tmp_path)
    with pytest.raises(ValueError): run(file,'0'*64,tmp_path/'out')
    (tmp_path/'out').mkdir()
    with pytest.raises(FileExistsError): run(file,digest,tmp_path/'out')


def test_negative_indices_and_bad_index():
    v,f=_obj(b'v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n')
    assert f.tolist()==[[0,1,2]]
    with pytest.raises(ValueError): _obj(b'v 0 0 0\nf 1 2 3\n')
