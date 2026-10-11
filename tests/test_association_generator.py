"""Synthetic source bytes only; hashes never imply real video provenance."""
import json
from pathlib import Path

import pytest

from gwm.feedback.ground_truth_source import build_pending_association, resolve_ground_truth
from gwm.perception.provenance import file_sha256
from scripts.create_ground_truth_association import main


def inputs(tmp_path):
    video = tmp_path / 'synthetic.bin'
    video.write_bytes(b'SYNTHETIC TEST BYTES, NOT A VIDEO')
    program = tmp_path / 'reference.json'
    program.write_text(json.dumps({'meta': {'duration': 6}}), encoding='utf-8')
    record = tmp_path / 'record.json'
    record.write_text('{"note":"synthetic test only"}', encoding='utf-8')
    out = tmp_path / 'associations.json'
    return video, program, record, out


def build(files, **overrides):
    kwargs = dict(association_id='synthetic_test', duration_s=6,
                  provenance_kind='synthetic_generation', review_note='人工核对前的合成测试')
    kwargs.update(overrides)
    return build_pending_association(*files, **kwargs)


def test_generated_hashes_and_pending_review_use_existing_gate(tmp_path):
    files = inputs(tmp_path)
    data = build(files)
    row = data['associations'][0]
    assert row['video']['sha256'] == file_sha256(files[0])
    assert row['ground_truth']['sha256'] == file_sha256(files[1])
    assert row['provenance']['record_sha256'] == file_sha256(files[2])
    assert row['video']['path'] == 'synthetic.bin'
    files[3].write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    result = resolve_ground_truth(files[0], files[3], (0, 6))
    assert result['diagnostics'][0]['code'] == 'SOURCE_REVIEW_REQUIRED'
    # Explicit simulated human attestation for unit testing, not automatic approval.
    row['review']['status'] = 'confirmed'
    files[3].write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    assert resolve_ground_truth(files[0], files[3], (0, 6))['ok']
    files[1].write_text('{"meta":{"duration":6},"changed":true}', encoding='utf-8')
    assert resolve_ground_truth(files[0], files[3], (0, 6))['diagnostics'][0]['code'] == 'SOURCE_PROGRAM_HASH_MISMATCH'


@pytest.mark.parametrize('duration', [0, -1, 7, float('nan'), float('inf'), True])
def test_invalid_coverage_refused(tmp_path, duration):
    with pytest.raises(ValueError):
        build(inputs(tmp_path), duration_s=duration)


def test_missing_and_outside_sources_refused(tmp_path):
    files = inputs(tmp_path)
    with pytest.raises(ValueError):
        build((tmp_path / 'missing', *files[1:]))
    with pytest.raises(ValueError):
        build((*files[:3], tmp_path / 'nested' / 'manifest.json'))


def test_cli_utf8_and_no_overwrite(tmp_path):
    video, program, record, out = inputs(tmp_path)
    args = ['--video', str(video), '--program', str(program), '--record', str(record),
            '--out', str(out), '--association-id', 'synthetic_test', '--duration', '6',
            '--kind', 'synthetic_generation', '--review-note', '尚待人工审核']
    assert main(args) == 0
    assert json.loads(out.read_text(encoding='utf-8'))['associations'][0]['review']['note'] == '尚待人工审核'
    before = out.read_bytes()
    with pytest.raises(SystemExit):
        main(args)
    assert out.read_bytes() == before
