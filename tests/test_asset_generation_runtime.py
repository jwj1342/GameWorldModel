import json
import os
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from gwm.assets.mesh import load_extent
from gwm.config import REPO
from gwm.dataset import behavior
from gwm.dataset.generator import SceneSpec, generate_scene
from gwm.compiler.validate import validate_candidate
from test_asset_generation_bounds import make_library, write_box


def test_asset_programs_pass_existing_validation(tmp_path,monkeypatch):
    cfg,_=make_library(tmp_path,'A',[.3,1.,.5])
    monkeypatch.setattr(behavior,'_ASSET_POOL',{})
    for seed in range(12):
        p,_=generate_scene(SceneSpec(clip=f'c{seed}',family='yard',behavior_seed=seed,
            style_seed=100,camera_seed=200,stage_scale=.15,asset_cfg=cfg))
        assert all(o['geom']['kind']=='asset' for o in p['objects'])
        report=validate_candidate(p)
        assert report['ok'],report['errors']


def test_generated_bounds_match_real_gltf_loader(tmp_path,monkeypatch):
    node=os.environ.get('GWM_NODE_BINARY') or shutil.which('node')
    if not node or not (REPO/'node_modules').is_dir():
        pytest.skip('Requires existing Node/Three; no downloads')
    cfg,path=make_library(tmp_path,'A',[1.,.5,.2])
    monkeypatch.setattr(behavior,'_ASSET_POOL',{})
    cases=[]
    for seed in range(24):
        geom,h,_,_=behavior._asset_geom(random.Random(seed),cfg)
        cases.append({'geom':geom,'pose':{'pos':[0,h,0]},'id':f'o{seed}','class':'one'})
    multi=write_box(tmp_path/'multi.glb',instances=True)
    spec=tmp_path/'fixtures.json'
    spec.write_text(json.dumps({'file':str(path),'raw_extent':load_extent(path).tolist(),
        'multi_file':str(multi),'multi_extent':load_extent(multi).tolist(),'nodes':cases}),encoding='utf-8')
    env={**os.environ,'GWM_GENERATED_ASSET_FIXTURE':str(spec)}
    run=subprocess.run([node,'--experimental-default-type=module',str(REPO/'tests/test_asset_generation_runtime.mjs')],
        env=env,capture_output=True,text=True,encoding='utf-8',timeout=60)
    print(run.stdout)
    assert run.returncode==0,run.stdout+run.stderr
