from copy import deepcopy
import json
import math

import pytest

from gwm.perception.geometry_audit import audit_evidence_geometry
from scripts.audit_evidence_geometry import main


def evidence(y=.5, height=1., contact=False):
    return {"meta": {"scale": "relative"},
            "static": {"planes": [{"kind": "ground", "normal": [0, 1, 0], "offset": 0}]},
            "objects": [{"id": "sample", "obb": [{"t": 0., "center": [0, y, 0],
                         "size": [1, height, 1], "quat": [0, 0, 0, 1]}],
                         "contacts": [{"with_id": "ground", "t_start": 0, "t_end": 1}] if contact else []}]}


def test_height_scaled_legacy_contact_can_conflict_with_geometry():
    ev = evidence(y=.4, contact=True)
    original = deepcopy(ev)
    report = audit_evidence_geometry(ev, legacy_contact_dist=.08)
    row = report["rows"][0]
    assert row["legacy_tolerance"] == pytest.approx(.8)
    assert row["legacy_on_ground"] is True
    assert row["gap"] == pytest.approx(-.1) and row["contact_conflict"] is True
    assert report["root_cause"] == "unresolved" and ev == original


def test_legitimate_floating_object_is_not_called_contact_conflict():
    report = audit_evidence_geometry(evidence(y=2))
    assert report["rows"][0]["gap"] == 1.5
    assert not report["diagnostics"] and not report["contact_conflict_objects"]


def test_plane_offset_is_not_silently_zero():
    ev = evidence(y=-1.5, contact=True)
    ev["static"]["planes"][0]["offset"] = 2
    assert audit_evidence_geometry(ev)["rows"][0]["gap"] == 0


def test_rotated_box_support_radius_is_used():
    ev = evidence(y=1)
    box = ev["objects"][0]["obb"][0]
    box["size"] = [1, 1, 2]
    box["quat"] = [math.sin(math.pi / 4), 0, 0, math.cos(math.pi / 4)]
    assert audit_evidence_geometry(ev)["rows"][0]["gap"] == pytest.approx(0)


@pytest.mark.parametrize("change", ["no_ground", "ambiguous", "no_offset", "bad_quat", "bad_position"])
def test_unavailable_geometry_stays_unknown(change):
    ev = evidence()
    if change == "no_ground": ev["static"]["planes"] = []
    if change == "ambiguous": ev["static"]["planes"] *= 2
    if change == "no_offset": del ev["static"]["planes"][0]["offset"]
    if change == "bad_quat": ev["objects"][0]["obb"][0]["quat"] = [0, 0, 0, 0]
    if change == "bad_position": ev["objects"][0]["obb"][0]["center"] = [0, float("nan"), 0]
    out = audit_evidence_geometry(ev)
    assert out["status"] == "partial" and out["rows"][0]["gap"] is None
    assert out["diagnostics"]


def test_contact_interval_does_not_claim_every_frame():
    ev = evidence(y=.4, contact=True)
    ev["objects"][0]["contacts"][0].update(t_start=1, t_end=2)
    out = audit_evidence_geometry(ev)
    assert out["below_ground_objects"] == ["sample"]
    assert not out["contact_conflict_objects"]


def test_duplicate_ground_and_support_diagnostics_count_one_pose():
    box = {"id": "sample", "class": "box", "geom": {"extent": [1, 1, 1]},
           "pose": {"pos": [0, .4, 0]}, "support": "ground"}
    ground = {"id": "ground", "class": "ground", "geom": {"extent": [10, .2, 10]},
              "pose": {"pos": [0, -.1, 0]}}
    program = {"objects": [box], "static": [ground]}
    original = deepcopy(program)
    candidate = audit_evidence_geometry(evidence(), program)["candidate"]
    assert len(candidate["errors"]) == 2
    assert candidate["penetration_pose_count"] == 1 and program == original


def test_cli_hashes_inputs_and_refuses_overwrite(tmp_path):
    source, dest = tmp_path / "evidence.json", tmp_path / "audit.json"
    source.write_text(json.dumps(evidence()), encoding="utf-8")
    before = source.read_bytes()
    args = ["--evidence", str(source), "--out", str(dest)]
    assert main(args) == 0
    report = json.loads(dest.read_text(encoding="utf-8"))
    assert len(next(iter(report["input_sha256"].values()))) == 64
    assert source.read_bytes() == before
    with pytest.raises(SystemExit): main(args)


def test_program_comparison_does_not_blame_writer_for_existing_conflict():
    ev = evidence(y=.4, contact=True)
    box = {"id": "sample", "class": "box", "geom": {"extent": [1, 1, 1]},
           "pose": {"pos": [0, .4, 0], "quat": [0, 0, 0, -1]}}
    out = audit_evidence_geometry(ev, {"objects": [box]})
    assert out["candidate"]["first_obb_comparison"][0]["status"] == "matches_first_obb"
    assert out["root_cause"] == "unresolved"
    box["pose"]["pos"][1] = 2
    out = audit_evidence_geometry(ev, {"objects": [box]})
    assert out["candidate"]["first_obb_comparison"][0]["status"] == "changed"


def test_explicit_program_pointer_and_empty_observations(tmp_path):
    ev = evidence()
    ev["objects"][0]["obb"] = []
    assert audit_evidence_geometry(ev)["status"] == "partial"
    ep, pp, dest = tmp_path / "e.json", tmp_path / "p.json", tmp_path / "r.json"
    ep.write_text(json.dumps(ev), encoding="utf-8")
    pp.write_text(json.dumps({"candidate": {"objects": []}}), encoding="utf-8")
    assert main(["--evidence", str(ep), "--program", str(pp),
                 "--program-pointer", "/candidate", "--out", str(dest)]) == 0
    assert json.loads(dest.read_text(encoding="utf-8"))["program_pointer"] == "/candidate"
