"""Known static GLBs: generator bounds must agree with the rendered mesh axes."""
import json
import random
import struct

import numpy as np
import pytest

from gwm.assets import library
from gwm.assets.mesh import load_extent
from gwm.dataset import behavior


def write_box(path, extent=(.3, 1., .5), instances=False, mutate=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    positions=np.array([[x,y,z] for x in (-extent[0]/2,extent[0]/2)
                        for y in (-extent[1]/2,extent[1]/2)
                        for z in (-extent[2]/2,extent[2]/2)],dtype='<f4')
    indices=np.array([0,1,3,0,3,2,4,6,7,4,7,5,0,4,5,0,5,1,
                      2,3,7,2,7,6,0,2,6,0,6,4,1,5,7,1,7,3],dtype='<u2')
    payload=positions.tobytes()+indices.tobytes()
    nodes=([{'children':[1,2]}, {'mesh':0,'translation':[-2,0,0]},
            {'mesh':0,'translation':[2,0,0]}] if instances else [{'mesh':0}])
    doc={'asset':{'version':'2.0'},'scene':0,'scenes':[{'nodes':[0]}],
         'nodes':nodes,'meshes':[{'primitives':[{'attributes':{'POSITION':0},'indices':1}]}],
         'buffers':[{'byteLength':len(payload)}],
         'bufferViews':[{'buffer':0,'byteOffset':0,'byteLength':positions.nbytes},
                        {'buffer':0,'byteOffset':positions.nbytes,'byteLength':indices.nbytes}],
         'accessors':[{'bufferView':0,'componentType':5126,'count':8,'type':'VEC3',
                       'min':positions.min(0).tolist(),'max':positions.max(0).tolist()},
                      {'bufferView':1,'componentType':5123,'count':36,'type':'SCALAR'}]}
    if mutate: mutate(doc)
    js=json.dumps(doc).encode();js+=b' '*((-len(js))%4)
    payload+=b'\0'*((-len(payload))%4)
    chunks=struct.pack('<II',len(js),0x4e4f534a)+js+struct.pack('<II',len(payload),0x004e4942)+payload
    path.write_bytes(struct.pack('<4sII',b'glTF',2,12+len(chunks))+chunks)
    return path


def make_library(tmp_path,name,extent):
    root=tmp_path/name
    path=write_box(root/'3d'/name/'one.glb',extent)
    library.write_catalog(root,[library.Asset(f'{name}/one',str(path.relative_to(root)),name,name,'one')])
    return {'assets':{'library':str(root)},'dataset':{'use_assets':True}},path


def test_xyz_order_and_all_instances(tmp_path):
    assert np.allclose(load_extent(write_box(tmp_path/'one.glb')),[.3,1.,.5])
    assert np.allclose(load_extent(write_box(tmp_path/'two.glb',instances=True)),[4.3,1.,.5])


def test_node_rotation_and_scale_match_transformed_local_box(tmp_path):
    def transform(doc):
        doc['nodes'][0].update(rotation=[0,0,2**-.5,2**-.5],scale=[2,1,1])
    assert np.allclose(load_extent(write_box(tmp_path/'turned.glb',mutate=transform)),[1.,.6,.5])


@pytest.mark.parametrize('change',[
    lambda d:d['nodes'][0].update(skin=0),
    lambda d:d['meshes'][0]['primitives'][0].update(targets=[{'POSITION':0}]),
    lambda d:d['accessors'][0].update(normalized=True),
    lambda d:d.update(scenes=[]),
])
def test_unsupported_geometry_does_not_supply_partial_bounds(tmp_path,change):
    assert load_extent(write_box(tmp_path/'unsupported.glb',mutate=change)) is None


def test_malformed_file_returns_none(tmp_path):
    bad=tmp_path/'bad.glb';bad.write_bytes(b'not glb')
    assert load_extent(bad) is None


def test_generator_preserves_true_xyz_proportions_without_shape_cache(tmp_path,monkeypatch):
    cfg,path=make_library(tmp_path,'A',[1.,.5,.2])
    monkeypatch.setattr(behavior,'_ASSET_POOL',{})
    raw=load_extent(path)
    for seed in range(24):
        geom,half_height,radius,name=behavior._asset_geom(random.Random(seed),cfg)
        ext=np.asarray(geom['extent'])
        assert np.allclose(ext/raw,ext[0]/raw[0])
        assert half_height==ext[1]/2
        assert radius==max(ext[0],ext[2])/2


def test_switching_library_does_not_reuse_first_pool(tmp_path,monkeypatch):
    a,_=make_library(tmp_path,'A',[1.,.5,.2]);b,_=make_library(tmp_path,'B',[.2,1.,.3])
    monkeypatch.setattr(behavior,'_ASSET_POOL',{})
    assert behavior._asset_geom(random.Random(0),a)[0]['ref']=='A/one'
    assert behavior._asset_geom(random.Random(0),b)[0]['ref']=='B/one'


def test_unsupported_selected_asset_uses_existing_fallback(tmp_path,monkeypatch):
    cfg,path=make_library(tmp_path,'A',[1.,.5,.2]);path.write_bytes(b'broken')
    monkeypatch.setattr(behavior,'_ASSET_POOL',{})
    assert behavior._asset_geom(random.Random(0),cfg) is None
