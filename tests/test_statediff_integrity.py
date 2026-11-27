"""Real project imports and validation/compiler; paid clients/browser are replaced."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.statediff import common, evaluate, prompt_baseline
from gwm.compiler.compile import compile_program


@pytest.mark.parametrize('parsed,expected',[
    ({'objects':[{},'bad']},None),({'objects':[None]},None),([{},False],None),
    ({'objects':[]},[]),({'objects':[{'id':'p'}]},[{'id':'p'}]),
    ([{'id':'p'}],[{'id':'p'}]),({'wrong':[]},None),
])
def test_original_array_is_not_cleaned(parsed,expected):
    before=copy.deepcopy(parsed)
    assert common.extract_objects(parsed)==expected
    assert parsed==before


def sample(tmp_path):
    program=json.loads((common.REPO/'examples/handwritten/program.json').read_text(encoding='utf-8'))
    path=tmp_path/'reference.json';path.write_text(json.dumps(program),encoding='utf-8')
    return program,dict(id='s0',set='holdout_in',duration=8,program=str(path),id0='gt0',id1='gt1')


def test_old_cache_cannot_bypass_real_validator(tmp_path,monkeypatch):
    _,s=sample(tmp_path)
    old=tmp_path/'work/source/s0/metrics.json';old.parent.mkdir(parents=True)
    old.write_text('{"valid":1,"iou_tT":1}',encoding='utf-8');before=old.read_bytes()
    def forbidden(*args,**kwargs):raise AssertionError('invalid Program must not render')
    monkeypatch.setattr(evaluate,'render_endpoints',forbidden)
    monkeypatch.setattr(evaluate,'object_masks',lambda p,_:{o['id']:np.ones((2,2),bool) for o in p['objects']})
    result=evaluate.eval_one(('source',s,[{'id':'bad'}],str(tmp_path/'work')))
    assert result['valid']==0 and result['reason'].startswith('invalid:')
    assert old.read_bytes()==before
    assert (Path(result['evaluation_dir'])/'metrics.json').exists()


def test_repeated_evaluation_recompiles_and_preserves_attempts(tmp_path,monkeypatch):
    program,s=sample(tmp_path);compiled=[]
    def render(p,work,times,cfg):
        res=compile_program(p,work/'game',{'assets':{'enabled':False}})
        assert res['ok'],res
        assert (Path(res['game_dir'])/'index.html').is_file()
        compiled.append(res['game_dir'])
        return {'ok':True,'files':{0.0:{'id':'fake0'},8.0:{'id':'fake1'}}}
    monkeypatch.setattr(evaluate,'render_endpoints',render)
    monkeypatch.setattr(evaluate,'object_masks',lambda p,_:{o['id']:np.ones((2,2),bool) for o in p['objects']})
    first=evaluate.eval_one(('source',s,common.target_objects(program),str(tmp_path/'work')))
    second=evaluate.eval_one(('source',s,common.target_objects(program),str(tmp_path/'work')))
    assert first['valid']==second['valid']==1
    assert len(set(compiled))==2
    assert first['evaluation_dir']!=second['evaluation_dir']
    assert all((Path(x['evaluation_dir'])/'metrics.json').exists() for x in (first,second))


@pytest.fixture
def prompt_run(tmp_path,monkeypatch):
    program,s=sample(tmp_path)
    for k in ('img0','img1'):
        p=tmp_path/(k+'.png');Image.new('RGB',(4,4),'red').save(p);s[k]=str(p)
    s['text']='The objects move.'
    samples=tmp_path/'samples.jsonl';samples.write_text(json.dumps(s)+'\n',encoding='utf-8')
    out=tmp_path/'pred.jsonl'
    cfg={'vlm':{'model':'offline-fake','endpoint':'https://unused.invalid/v1','provider':'openai_compat'}}
    calls=[];clients=[]
    class Fake:
        calls=0;prompt_tokens=0;completion_tokens=0
        def chat(self,*args,**kwargs):
            calls.append((args,kwargs));self.calls+=1
            return {'text':'{"objects":[]}','json':{'objects':[]},'usage':{},'finish':'stop'}
    def make(*args,**kwargs):clients.append(True);return Fake()
    monkeypatch.setattr(prompt_baseline,'make_client',make)
    monkeypatch.setattr(prompt_baseline,'load_config',lambda **kwargs:copy.deepcopy(cfg))
    monkeypatch.setattr(sys,'argv',['prompt_baseline','--samples',str(samples),'--sets','holdout_in','--out',str(out)])
    return dict(run=prompt_baseline.main,cfg=cfg,calls=calls,clients=clients,out=out,s=s,program=program)


def test_resume_real_prompt_schema_no_second_client(prompt_run):
    p=prompt_run
    assert p['run']()==0
    assert p['run']()==0
    assert len(p['clients'])==len(p['calls'])==1
    args,kwargs=p['calls'][0]
    assert kwargs['json_schema']==common.objects_schema()
    assert 'Known parts of the program' in args[1]
    assert kwargs['images']==[p['s']['img0'],p['s']['img1']]


@pytest.mark.parametrize('change',['model','image','context','temperature'])
def test_changed_conditions_stop_before_client(prompt_run,change,monkeypatch):
    p=prompt_run;p['run']();before=p['out'].read_bytes()
    if change=='model':p['cfg']['vlm']['model']='different'
    elif change=='image':Image.new('RGB',(4,4),'blue').save(p['s']['img0'])
    elif change=='context':
        j=p['program'];j['meta']['clip']='different'
        Path(p['s']['program']).write_text(json.dumps(j),encoding='utf-8')
    else:monkeypatch.setattr(sys,'argv',sys.argv+['--temperature','.7'])
    with pytest.raises(ValueError,match='identity mismatch'):p['run']()
    assert len(p['clients'])==len(p['calls'])==1
    assert p['out'].read_bytes()==before
