"""Fail-closed file/source association checks, independent of metric execution.

Hashes establish content identity, not semantic truth. A confirmed review is an
explicit human attestation; this module does not infer it from file names.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import re

from ..perception.provenance import file_sha256


class _Rejected(ValueError):
    def __init__(self, code, path, hint):
        super().__init__(hint)
        self.diagnostic = {"stage":"ground_truth_source", "severity":"error",
                           "code":code, "path":path, "hint":hint}


def _reject(code, path, hint):
    raise _Rejected(code, path, hint)


def _number(value, path, *, positive=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        _reject("SOURCE_INVALID_NUMBER",path,"Expected a finite numeric value")
    if positive and value <= 0:
        _reject("SOURCE_INVALID_NUMBER",path,"Expected a positive numeric value")
    return float(value)


def _relative_file(root, value, path):
    if not isinstance(value,str) or not value.strip():
        _reject("SOURCE_INVALID_PATH",path,"Expected a nonempty manifest-relative file path")
    relative = Path(value)
    if relative.is_absolute() or PureWindowsPath(value).drive or '\\' in value or '..' in relative.parts:
        _reject("SOURCE_INVALID_PATH",path,"Use portable relative paths within the manifest directory")
    result = (root/relative).resolve()
    if not result.is_relative_to(root):
        _reject("SOURCE_INVALID_PATH",path,"Resolved path escapes the manifest directory")
    return result


def _fingerprint(value, path):
    if not isinstance(value,str) or re.fullmatch(r'[0-9a-f]{64}',value) is None:
        _reject("SOURCE_INVALID_HASH",path,"Expected a lowercase 64-character SHA-256")
    return value


def _mapping(value, path):
    if not isinstance(value,dict):
        _reject("SOURCE_INVALID_MANIFEST",path,"Missing time_mapping object")
    for field in ('video_start_s','program_start_s','duration_s','rate'):
        _number(value.get(field),f'{path}.{field}',positive=field in ('duration_s','rate'))


def _load_unambiguous_json(raw):
    """Do not let duplicate fields or non-JSON constants redefine reviewed data."""
    def unique_fields(pairs):
        result = {}
        for key,value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError("Non-finite JSON constant")
    return json.loads(raw,object_pairs_hook=unique_fields,parse_constant=invalid_constant)


def _validate_manifest(data, root):
    if not isinstance(data,dict) or type(data.get('schema_version')) is not int or data['schema_version'] != 1:
        _reject("SOURCE_INVALID_MANIFEST","schema_version","Unsupported or missing association schema version")
    rows = data.get('associations')
    if not isinstance(rows,list):
        _reject("SOURCE_INVALID_MANIFEST","associations","Expected an association list")
    ids = set()
    for i,row in enumerate(rows):
        base = f'associations[{i}]'
        if not isinstance(row,dict):
            _reject("SOURCE_INVALID_MANIFEST",base,"Expected an association object")
        identity = row.get('association_id')
        if not isinstance(identity,str) or not identity.strip() or identity in ids:
            _reject("SOURCE_INVALID_MANIFEST",base+'.association_id',"Association IDs must be nonempty and unique")
        ids.add(identity)
        for section in ('video','ground_truth'):
            part = row.get(section)
            if not isinstance(part,dict):
                _reject("SOURCE_INVALID_MANIFEST",base+'.'+section,"Missing file descriptor")
            _relative_file(root,part.get('path'),base+'.'+section+'.path')
            _fingerprint(part.get('sha256'),base+'.'+section+'.sha256')
        _mapping(row.get('time_mapping'),base+'.time_mapping')
        origin,review = row.get('provenance'),row.get('review')
        if not isinstance(origin,dict) or origin.get('kind') not in ('synthetic_generation','manual_reference'):
            _reject("SOURCE_INVALID_MANIFEST",base+'.provenance.kind',"A supported provenance kind is required")
        _relative_file(root,origin.get('record_path'),base+'.provenance.record_path')
        _fingerprint(origin.get('record_sha256'),base+'.provenance.record_sha256')
        if not isinstance(review,dict) or review.get('status') not in ('pending','confirmed','rejected'):
            _reject("SOURCE_INVALID_MANIFEST",base+'.review.status',"Expected pending, confirmed or rejected")
        if not isinstance(review.get('note'),str) or not review['note'].strip():
            _reject("SOURCE_INVALID_MANIFEST",base+'.review.note',"A review rationale is required")
    return rows


def resolve_ground_truth(video_path, association_file, requested_window):
    """Return a JSON-serializable verified resolution or rejection diagnostics.

    requested_window is (video_start_s, video_end_s). Initially only zero-origin,
    normal-speed mappings are executable. No decoding, rendering or model calls.
    A moved/renamed actual video can match by hash; the stored video path is a
    locator, not a requirement that the old copy still exists.
    """
    context = 'requested_window'
    try:
        if not isinstance(requested_window,(list,tuple)) or len(requested_window) != 2:
            _reject("SOURCE_INVALID_WINDOW",context,"Expected [start_s, end_s]")
        start = _number(requested_window[0],context+'[0]')
        end = _number(requested_window[1],context+'[1]',positive=True)
        if start != 0 or end <= start:
            _reject("SOURCE_UNSUPPORTED_TIME_MAPPING",context,"Only positive zero-origin evaluation windows are supported")
        context = 'association_file'
        manifest_path = Path(association_file).resolve()
        raw = manifest_path.read_bytes()
        manifest_hash = hashlib.sha256(raw).hexdigest()
        rows = _validate_manifest(_load_unambiguous_json(raw),manifest_path.parent)
        context = 'video'
        video = Path(video_path).resolve()
        video_hash = file_sha256(video)
        matches = [(i,row) for i,row in enumerate(rows) if row['video']['sha256']==video_hash]
        if not matches:
            _reject("SOURCE_ASSOCIATION_NOT_FOUND",context,"No content-hash association for the actual video; names are not a fallback")
        confirmed = [(i,row) for i,row in matches if row['review']['status']=='confirmed']
        if not confirmed:
            _reject("SOURCE_REVIEW_REQUIRED","associations.review","Matching associations are pending or rejected")
        if len(confirmed) != 1:
            _reject("SOURCE_ASSOCIATION_CONFLICT","associations","Multiple confirmed associations match this video")
        i,row = confirmed[0]
        base = f'associations[{i}]'
        mapping = row['time_mapping']
        if mapping['video_start_s'] != 0 or mapping['program_start_s'] != 0 or mapping['rate'] != 1:
            _reject("SOURCE_UNSUPPORTED_TIME_MAPPING",base+'.time_mapping',"Nonzero offsets and speed changes are not supported yet")
        if end > mapping['duration_s']:
            _reject("SOURCE_WINDOW_OUT_OF_RANGE",base+'.time_mapping.duration_s',"Requested window exceeds reviewed association coverage")
        context = base+'.ground_truth.path'
        program = _relative_file(manifest_path.parent,row['ground_truth']['path'],context)
        program_bytes = program.read_bytes()
        program_hash = hashlib.sha256(program_bytes).hexdigest()
        if program_hash != row['ground_truth']['sha256']:
            _reject("SOURCE_PROGRAM_HASH_MISMATCH",base+'.ground_truth.sha256',"Program content changed; review the association again")
        parsed = _load_unambiguous_json(program_bytes)
        meta = parsed.get('meta') if isinstance(parsed,dict) else None
        if not isinstance(meta,dict):
            _reject("SOURCE_INVALID_PROGRAM",context,"Program must include an explicit meta.duration")
        duration = _number(meta.get('duration'),context+'.meta.duration',positive=True)
        if mapping['duration_s'] > duration:
            _reject("SOURCE_WINDOW_OUT_OF_RANGE",base+'.time_mapping.duration_s',"Reviewed coverage exceeds Program duration")
        context = base+'.provenance.record_path'
        record = _relative_file(manifest_path.parent,row['provenance']['record_path'],context)
        record_hash = file_sha256(record)
        if record_hash != row['provenance']['record_sha256']:
            _reject("SOURCE_PROVENANCE_HASH_MISMATCH",base+'.provenance.record_sha256',"Association evidence changed; review again")
        for path,digest,label in [(manifest_path,manifest_hash,'association_file'),(video,video_hash,'video'),
                                  (program,program_hash,base+'.ground_truth'),(record,record_hash,base+'.provenance')]:
            context = label
            if file_sha256(path) != digest:
                _reject("SOURCE_CHANGED_DURING_CHECK",label,"Source changed during verification")
        return {"ok":True,"status":"verified","diagnostics":[],
                "association_id":row['association_id'],"ground_truth_path":str(program),
                "requested_window_s":[start,end],"time_mapping":dict(mapping),
                "provenance_kind":row['provenance']['kind'],"review":dict(row['review']),
                "sources":{"video_sha256":video_hash,"ground_truth_sha256":program_hash,
                           "provenance_record_sha256":record_hash,"association_file_sha256":manifest_hash},
                "limitations":["Human-attested association; semantic truth and video PTS coverage are not automatically verified",
                               "Program schema validation remains the compiler's responsibility"]}
    except _Rejected as error:
        return {"ok":False,"status":"rejected","diagnostics":[error.diagnostic]}
    except (OSError,ValueError,TypeError,OverflowError) as error:
        diagnostic = {"stage":"ground_truth_source","severity":"error",
                      "code":"SOURCE_READ_OR_FORMAT_ERROR","path":context,
                      "hint":f'Cannot verify source file or format ({type(error).__name__})'}
        return {"ok":False,"status":"rejected","diagnostics":[diagnostic]}
