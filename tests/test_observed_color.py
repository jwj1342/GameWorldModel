"""Small real PNG fixtures; no models, GPU or browser."""
import copy
import numpy as np
import pytest
from PIL import Image
from gwm.config import load_config
from gwm.perception.color import observe_color
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
    assert target['material']=={'base_color':'#c8281e'}
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
