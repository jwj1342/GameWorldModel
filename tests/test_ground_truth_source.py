"""Source associations use synthetic bytes only; no model, decoder or browser."""
import copy
import json

import pytest

from gwm.feedback.ground_truth_source import resolve_ground_truth
from gwm.perception.provenance import file_sha256


@pytest.fixture
def source(tmp_path):
    video=tmp_path/'video.bin'; video.write_bytes(b'SYNTHETIC VIDEO BYTES, NOT AN MP4')
    program=tmp_path/'program.json'; program.write_text(json.dumps({'meta':{'duration':8}}))
    record=tmp_path/'generation.json'; record.write_text(json.dumps({'note':'Synthetic test record, not real generation provenance'}))
    row={'association_id':'synthetic_reference',
         'video':{'path':'video.bin','sha256':file_sha256(video)},
         'ground_truth':{'path':'program.json','sha256':file_sha256(program)},
         'time_mapping':{'video_start_s':0,'program_start_s':0,'duration_s':8,'rate':1},
         'provenance':{'kind':'synthetic_generation','record_path':'generation.json','record_sha256':file_sha256(record)},
         'review':{'status':'confirmed','note':'Synthetic confirmed association for unit tests only'}}
    data={'schema_version':1,'associations':[row]}
    manifest=tmp_path/'ground-truth-associations.json'
    return video,program,record,manifest,data


def resolve(source,window=(0,8)):
    video,_,_,manifest,data=source
    manifest.write_text(json.dumps(data),encoding='utf-8')
    return resolve_ground_truth(video,manifest,window)


def code(result):
    assert result['ok'] is False and result['status']=='rejected'
    assert 'ground_truth_path' not in result
    diag=result['diagnostics'][0]
    assert {'stage','severity','code','path','hint'} <= diag.keys()
    return diag['code']


def test_verified_resolution(source):
    result=resolve(source)
    assert result['ok'] and result['status']=='verified'
    assert result['ground_truth_path']==str(source[1].resolve())
    assert result['sources']['video_sha256']==file_sha256(source[0])
    assert result['sources']['association_file_sha256']==file_sha256(source[3])
    assert result['requested_window_s']==[0,8]
    assert result['limitations']


def test_renamed_video_matches_by_content(source):
    old=source[0]; new=old.with_name('renamed.bin'); old.rename(new)
    changed=(new,*source[1:])
    assert resolve(changed)['ok']  # recorded old path no longer exists


def test_same_name_different_content_refused(source):
    source[0].write_bytes(b'OTHER VIDEO WITH SAME FILE NAME')
    assert code(resolve(source))=='SOURCE_ASSOCIATION_NOT_FOUND'


@pytest.mark.parametrize('status',['pending','rejected'])
def test_review_required(source,status):
    source[4]['associations'][0]['review']['status']=status
    assert code(resolve(source))=='SOURCE_REVIEW_REQUIRED'


def test_duplicate_confirmed_associations_refused(source):
    other=copy.deepcopy(source[4]['associations'][0]); other['association_id']='other'
    source[4]['associations'].append(other)
    assert code(resolve(source))=='SOURCE_ASSOCIATION_CONFLICT'


def test_pending_candidate_does_not_override_confirmed(source):
    other=copy.deepcopy(source[4]['associations'][0]); other['association_id']='pending'
    other['review']['status']='pending'
    source[4]['associations'].append(other)
    assert resolve(source)['ok']


@pytest.mark.parametrize('index,expected',[(1,'SOURCE_PROGRAM_HASH_MISMATCH'),(2,'SOURCE_PROVENANCE_HASH_MISMATCH')])
def test_changed_reference_requires_review(source,index,expected):
    source[index].write_bytes(b'changed')
    assert code(resolve(source))==expected


@pytest.mark.parametrize('index',[0,1,2])
def test_missing_file_refused(source,index):
    # Source files are tiny test fixtures, not repository/experiment data.
    source[index].unlink()
    assert code(resolve(source))=='SOURCE_READ_OR_FORMAT_ERROR'


@pytest.mark.parametrize('field,value',[('video_start_s',2),('program_start_s',2),('rate',2)])
def test_nonzero_mapping_refused(source,field,value):
    source[4]['associations'][0]['time_mapping'][field]=value
    assert code(resolve(source))=='SOURCE_UNSUPPORTED_TIME_MAPPING'


@pytest.mark.parametrize('window',[(0,9),(1,8),(0,float('nan')),(0,True),(),(0,0)])
def test_invalid_or_uncovered_window(source,window):
    assert code(resolve(source,window)).startswith('SOURCE_')


@pytest.mark.parametrize('section,field,value',[
    ('video','sha256','not-a-hash'),('ground_truth','path','../outside.json'),
    ('ground_truth','path','C:/outside.json'),('review','note',''),
    ('review','status','approved'),('time_mapping','duration_s',float('inf')),
    ('time_mapping','rate',True),('provenance','kind','unknown')])
def test_malformed_association_refused(source,section,field,value):
    source[4]['associations'][0][section][field]=value
    assert code(resolve(source)).startswith('SOURCE_')


def test_missing_manifest_is_diagnostic(source):
    assert code(resolve_ground_truth(source[0],source[3],(0,8)))=='SOURCE_READ_OR_FORMAT_ERROR'


def test_missing_association_no_name_fallback(source):
    source[4]['associations']=[]
    assert code(resolve(source))=='SOURCE_ASSOCIATION_NOT_FOUND'


def test_program_coverage_checked_without_default_duration(source):
    source[1].write_text(json.dumps({'meta':{'duration':6}}))
    source[4]['associations'][0]['ground_truth']['sha256']=file_sha256(source[1])
    assert code(resolve(source))=='SOURCE_WINDOW_OUT_OF_RANGE'


def test_partial_zero_origin_window_supported(source):
    assert resolve(source,(0,4))['ok']


def test_source_change_during_check_refused(source,monkeypatch):
    from gwm.feedback import ground_truth_source as module
    actual=module.file_sha256; calls=0
    def mutate(path):
        nonlocal calls
        calls+=1
        if calls==3: source[0].write_bytes(b'changed during verification')
        return actual(path)
    monkeypatch.setattr(module,'file_sha256',mutate)
    assert code(resolve(source))=='SOURCE_CHANGED_DURING_CHECK'


@pytest.mark.parametrize('version',[True,2,'1'])
def test_unsupported_manifest_version(source,version):
    source[4]['schema_version']=version
    assert code(resolve(source))=='SOURCE_INVALID_MANIFEST'


@pytest.mark.parametrize('section,field',[
    ('video','path'),('video','sha256'),('ground_truth','path'),('ground_truth','sha256'),
    ('time_mapping','video_start_s'),('time_mapping','program_start_s'),
    ('time_mapping','duration_s'),('time_mapping','rate'),
    ('provenance','kind'),('provenance','record_path'),('provenance','record_sha256'),
    ('review','status'),('review','note')])
def test_each_required_field_is_enforced(source,section,field):
    source[4]['associations'][0][section].pop(field)
    assert code(resolve(source)).startswith('SOURCE_')


@pytest.mark.parametrize('raw',[b'not json',b'\xff',b'[]',b'{"schema_version":1,"associations":{}}',
                              b'{"schema_version":1,"associations":[null]}'])
def test_invalid_manifest_content_returns_diagnostic(source,raw):
    source[3].write_bytes(raw)
    assert code(resolve_ground_truth(source[0],source[3],(0,8))).startswith('SOURCE_')


def test_duplicate_association_id_rejected(source):
    source[4]['associations'].append(copy.deepcopy(source[4]['associations'][0]))
    assert code(resolve(source))=='SOURCE_INVALID_MANIFEST'


def test_malformed_unrelated_entry_is_not_silently_ignored(source):
    source[4]['associations'].append({'association_id':'unrelated'})
    assert code(resolve(source))=='SOURCE_INVALID_MANIFEST'


@pytest.mark.parametrize('path',['/absolute.json','//host/share/program.json','C:program.json',
                               'nested/../../program.json','nested\\program.json'])
def test_nonportable_or_escaping_paths_rejected(source,path):
    source[4]['associations'][0]['ground_truth']['path']=path
    assert code(resolve(source))=='SOURCE_INVALID_PATH'


@pytest.mark.parametrize('duration',[None,True,'8',float('nan'),float('inf'),-1])
def test_program_duration_cannot_be_guessed(source,duration):
    source[1].write_text(json.dumps({'meta':{'duration':duration}}))
    source[4]['associations'][0]['ground_truth']['sha256']=file_sha256(source[1])
    assert code(resolve(source)).startswith('SOURCE_')


@pytest.mark.parametrize('target',['manifest','program'])
def test_duplicate_json_fields_not_last_value_wins(source,target):
    if target=='manifest':
        resolve(source)
        raw=source[3].read_text(encoding='utf-8').replace('"status": "confirmed"',
                 '"status": "rejected", "status": "confirmed"')
        source[3].write_text(raw,encoding='utf-8')
    else:
        source[1].write_text('{"meta":{"duration":2,"duration":8}}')
        source[4]['associations'][0]['ground_truth']['sha256']=file_sha256(source[1])
        resolve(source)
    assert code(resolve_ground_truth(source[0],source[3],(0,8)))=='SOURCE_READ_OR_FORMAT_ERROR'
