"""Small real PNG fixtures; no models, GPU or browser."""
import copy
import json
import shutil
import subprocess
import numpy as np
import pytest
from PIL import Image
from gwm.config import REPO, load_config
from gwm.perception.color import observe_color, observe_colors
from gwm.perception.base import Geometry, Tracks, TrackedObject
from gwm.perception.contract import validate_evidence
from gwm.perception.quality import assess_evidence_quality
from gwm.perception.evidence import build_evidence
from gwm.synthesis.direct import evidence_to_program
from gwm.synthesis.writer import evidence_summary
from gwm.compiler.validate import validate

def frame(tmp_path, rgb, index=0, size=20):
    path=tmp_path/f'f_{index}.png'
    Image.new('RGB',(size,size),rgb).save(path)
    return {'index':index,'t':float(index),'file':str(path)}

def test_mask_excludes_background_and_inputs_stay_unchanged(tmp_path):
    f=frame(tmp_path,(0,0,255))
    image=np.asarray(Image.open(f['file'])).copy()
    image[5:15,5:15]=[200,40,30]
    Image.fromarray(image).save(f['file'])
    m=np.zeros((20,20),bool); m[5:15,5:15]=True
    before=m.copy()
    result=observe_color([f],{0:m},(20,20),'target')
    assert result['base_color']=='#c8281e'
    assert result['confidence']=='unknown'
    assert result['samples'][0]['pixels']==100
    assert np.array_equal(m,before)

def test_cross_frame_equal_weight_median_and_resizing(tmp_path):
    frames=[frame(tmp_path,c,i) for i,c in enumerate([(10,20,30),(100,110,120),(200,210,220)])]
    masks={i:np.ones((10,10),bool) for i in range(3)}
    result=observe_color(frames,masks,(20,20),'target')
    assert result['base_color']=='#646e78' and len(result['samples'])==3

@pytest.mark.parametrize('case,reason',[
    ('missing','missing_mask'),('empty','empty_mask'),('small','insufficient_pixels'),
    ('shape','mask_aspect_mismatch'),('nonbinary','invalid_binary_mask'),
    ('badref','frame_reference_mismatch'),('image','image_unavailable')])
def test_missing_or_invalid_support_is_unknown(tmp_path,case,reason):
    f=frame(tmp_path,(100,200,30))
    masks={0:np.ones((20,20),bool)}
    if case=='missing': masks={}
    if case=='empty': masks[0][:]=False
    if case=='small': masks[0][:]=False; masks[0][0,0]=True
    if case=='shape': masks[0]=np.ones((2,20),bool)
    if case=='nonbinary': masks[0]=np.ones((20,20),float)
    if case=='badref': masks={1:masks[0]}
    if case=='image': f['file']=str(tmp_path/'absent.png')
    result=observe_color([f],masks,(20,20),'target')
    assert result['base_color']=='unknown' and not result['samples']
    assert result['issues'][0]['reason']==reason

def built(tmp_path,monkeypatch,phrase='box'):
    from gwm.perception import evidence,overlay
    monkeypatch.setattr(evidence,'align_world',lambda *a:(np.eye(4),1.,{'ground':'not_found'}))
    monkeypatch.setattr(overlay,'render_overlay',lambda *a:None)
    frames=[frame(tmp_path,(200,40,30),i) for i in range(2)]
    camera=np.eye(4); camera[1,3]=-1.0  # OpenCV world; y-up conversion places target above ground.
    geom=Geometry([0,1],[np.array([[20,0,10],[0,20,10],[0,0,1.]])]*2,
        [camera]*2,[np.ones((20,20))]*2,scale='metric',backend='synthetic')
    obj=TrackedObject('target',phrase,.8,masks={i:np.ones((20,20),bool) for i in range(2)})
    return build_evidence('synthetic',frames,geom,Tracks([obj],(20,20),'synthetic'),
        load_config(),tmp_path/'evidence',[],[])

@pytest.mark.parametrize('phrase',['box','wall'])
def test_pipeline_material_only_changes_not_behavior(tmp_path,monkeypatch,phrase):
    ev=built(tmp_path,monkeypatch,phrase)
    assert ev['objects'] and validate_evidence(ev)['ok']
    assert assess_evidence_quality(ev)['decision']!='block'
    if phrase=='wall': ev['objects'][0]['motion_guess']={'type':'static','conf':.8}
    baseline=copy.deepcopy(ev); baseline['objects'][0].pop('appearance')
    before=copy.deepcopy(ev)
    colored=evidence_to_program(ev)
    old=evidence_to_program(baseline)
    assert ev==before and validate(colored)['ok']
    target=next(n for n in colored['static']+colored['objects'] if n['id']=='target')
    assert target['material']=={'base_color':'#c8281e', 'roughness': .9}
    for p in (colored,old):
        for n in p['static']+p['objects']: n.pop('material',None)
    assert colored==old
    assert 'observed_sRGB id=target: #c8281e' in evidence_summary(ev)

@pytest.mark.parametrize('change',['reference','time','source','format','unsupported'])
def test_color_contract_rejects_bad_measurements(tmp_path,monkeypatch,change):
    ev=built(tmp_path,monkeypatch)
    appearance=ev['objects'][0]['appearance']
    if change=='reference': appearance['samples'][0]['frame_index']=999
    if change=='time': appearance['samples'][0]['t']=.5
    if change=='source': appearance['samples'][0]['image_ref']='other.png'
    if change=='format': appearance['base_color']='red'
    if change=='unsupported': appearance['samples']=[]
    assert not validate_evidence(ev)['ok']

def test_threshold_is_configurable(tmp_path):
    f=frame(tmp_path,(10,20,30))
    m=np.zeros((20,20),bool);m[0,0]=True
    assert observe_color([f],{0:m},(20,20),'target',1)['base_color']=='#0a141e'
    with pytest.raises(ValueError): observe_color([f],{0:m},(20,20),'target',0)

def test_bad_frame_does_not_fabricate_or_drop_valid_measurement(tmp_path):
    frames=[frame(tmp_path,(10,20,30),i) for i in range(2)]
    frames[1]['file']=str(tmp_path/'missing.png')
    result=observe_color(frames,{i:np.ones((20,20),bool) for i in range(2)},(20,20),'target')
    assert result['base_color']=='#0a141e'
    assert [s['frame_index'] for s in result['samples']]==[0]
    assert result['issues'][0]['frame_index']==1


@pytest.mark.parametrize('size', [(960, 540), (1920, 1080)])
def test_patch_rounded_gt_mask_accepts_non_square_frames(tmp_path, size):
    path = tmp_path / 'wide.png'
    Image.new('RGB', size, (10, 20, 30)).save(path)
    frames = [{'index': 0, 't': 0., 'file': str(path)}]
    masks = {0: np.ones((294, 518), bool)}
    assert observe_color(frames, masks, (518, 294), 'target')['base_color'] == '#0a141e'
    strict = observe_color(frames, masks, (518, 294), 'target', aspect_tolerance=.001)
    assert strict['issues'][0]['reason'] == 'image_aspect_mismatch'
    assert strict['base_color'] == 'unknown'


@pytest.mark.parametrize('value', [-.1, 1, float('nan'), float('inf'), True])
def test_invalid_aspect_tolerance_rejected(value):
    with pytest.raises(ValueError):
        observe_colors([], {}, (20, 20), aspect_tolerance=value)


def test_large_aspect_mismatch_remains_unknown(tmp_path):
    path = tmp_path / 'cropped.png'
    Image.new('RGB', (960, 400), 'red').save(path)
    result = observe_color([{'index': 0, 't': 0., 'file': str(path)}],
                           {0: np.ones((294, 518), bool)}, (518, 294), 'target')
    assert result['base_color'] == 'unknown'
    assert result['issues'][0]['reason'] == 'image_aspect_mismatch'


def test_batch_decodes_once_per_frame_and_matches_single_object(tmp_path, monkeypatch):
    frames = [frame(tmp_path, color, i) for i, color in enumerate([(10, 20, 30), (90, 80, 70)])]
    masks = {i: np.ones((20, 20), bool) for i in range(2)}
    expected = {oid: observe_color(frames, masks, (20, 20), oid) for oid in ['one', 'two']}
    original_open = Image.open
    opened = []
    def counted(path, *args, **kwargs):
        opened.append(str(path))
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Image, 'open', counted)
    result = observe_colors(frames, {'one': masks, 'two': masks}, (20, 20))
    assert result == expected
    assert opened == [f['file'] for f in frames]


def test_batch_failure_is_reported_for_each_object(tmp_path):
    frames = [{'index': 0, 't': 0., 'file': str(tmp_path / 'missing.png')}]
    masks = {0: np.ones((20, 20), bool)}
    result = observe_colors(frames, {'one': masks, 'two': masks}, (20, 20))
    assert all(r['base_color'] == 'unknown' and r['issues'][0]['reason'] == 'image_unavailable'
               for r in result.values())


def test_failed_color_measurement_enters_quality_gate(tmp_path, monkeypatch):
    ev = built(tmp_path, monkeypatch)
    obj = ev['objects'][0]
    obj['appearance'].update(base_color='unknown', samples=[],
                             issues=[{'reason': 'image_aspect_mismatch'}])
    report = assess_evidence_quality(ev, load_config())
    finding = next(d for d in report['diagnostics'] if d['code'] == 'color_measurement_unavailable')
    assert report['decision'] == 'warn'
    assert finding['stage'] == 'evidence_quality' and finding['severity'] == 'warning'
    assert finding['path'] == '/objects/0/appearance'
    assert 'image_aspect_mismatch' in finding['message']
    # Legacy Evidence did not attempt color measurement; do not invent failures.
    obj.pop('appearance')
    assert not any(d['code'] == 'color_measurement_unavailable'
                   for d in assess_evidence_quality(ev, load_config())['diagnostics'])


def test_summary_only_describes_measured_colors_once(tmp_path, monkeypatch):
    ev = built(tmp_path, monkeypatch)
    extra = copy.deepcopy(ev['objects'][0]); extra['id'] = 'other'
    ev['objects'].append(extra)
    text = evidence_summary(ev)
    assert text.count('display-color proxy, not albedo') == 1
    extra.pop('appearance')
    text = evidence_summary(ev)
    assert 'observed_sRGB id=other' not in text
    ev['objects'][0].pop('appearance')
    assert 'observed_sRGB' not in evidence_summary(ev)


def test_material_conversion_preserves_all_kernel_surface_priors(monkeypatch):
    from gwm.synthesis import direct
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for kernel material parity')
    output = subprocess.run([node, '--input-type=module', '-e',
        "import {MATERIALS} from './kernel/scene.js'; console.log(JSON.stringify(MATERIALS))"],
        cwd=REPO, capture_output=True, text=True)
    # 有 node 不等于能解析 three：集群上 node_modules 不在仓库里，而是由
    # scripts/node_harness.sh 解到计算节点本地盘。那种情况下跳过，而不是把一个
    # 环境问题报成代码缺陷。
    if output.returncode != 0:
        pytest.skip(f'kernel/scene.js is not importable here: {output.stderr.strip()[:200]}')
    palette = json.loads(output.stdout)
    assert set(palette) == set(direct.PALETTE_SURFACES)
    appearance = {'base_color': '#123456', 'color_space': 'sRGB', 'samples': [{}]}
    for name, values in palette.items():
        monkeypatch.setattr(direct, 'guess_material', lambda cls, name=name: name)
        obj = {'class_guess': 'synthetic', 'appearance': appearance}
        material = direct.material_from_evidence(obj)
        expected = {k: v for k, v in values.items() if k not in {'color', 'transparent'}}
        if 'emissive' in expected:
            expected['emissive'] = f"#{expected['emissive']:06x}"
            expected['emissive_intensity'] = expected.pop('emissiveIntensity')
        assert material == {**expected, 'base_color': '#123456'}
        assert direct.material_from_evidence({'class_guess': 'synthetic'}) == name
