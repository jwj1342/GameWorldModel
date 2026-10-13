import copy
import importlib
import json
import numpy as np
import pytest
from PIL import Image

from gwm.perception.visibility import measure_visibility, checked_visibility
from gwm.perception.base import Geometry, Tracks, TrackedObject
from gwm.config import load_config
from gwm.perception.contract import validate_evidence
from gwm.perception.quality import assess_evidence_quality


def record(t=0, fraction=.1):
    return {"object_id": "target", "frame_index": 0, "image_size": [40,40],
            "program_sha256": "a"*64, "requested_t": t, "actual_t": t,
            "fraction": fraction, "method": "isolated_instance_id", "same_state_verified": True}


def test_mask_fraction_is_not_distance_normalized_to_a_different_frame():
    full = np.ones((20,20), bool); visible = full.copy(); visible[:10] = False
    assert measure_visibility(visible, full) == (.5, "")
    assert measure_visibility(full, full) == (1, "")
    assert measure_visibility(np.zeros_like(full), full) == (0, "")
    assert measure_visibility(np.zeros_like(full), np.zeros_like(full))[0] == "unknown"
    assert measure_visibility(visible[:, :10], full)[1] == "mask_size_mismatch"
    assert measure_visibility(full, visible)[1] == "visible_pixels_outside_reference"


@pytest.mark.parametrize("field,value", [("object_id","other"), ("frame_index",1),
    ("image_size",[30,40]), ("program_sha256","b"*64), ("actual_t",.2),
    ("requested_t",.2), ("fraction",float("nan")), ("method","same_frame_obb"),
    ("same_state_verified",False)])
def test_mismatched_reference_is_unknown(field, value):
    r=record(); r[field]=value
    assert checked_visibility(r,"target",0,0,(40,40),"a"*64)[0] == "unknown"


def test_reference_requires_geometry_source_and_does_not_mutate():
    r=record(); before=copy.deepcopy(r)
    assert checked_visibility(r,"target",0,0,(40,40))[0] == "unknown"
    assert checked_visibility(r,"target",0,0,(40,40),"a"*64)[0] == .1
    assert r==before


@pytest.mark.parametrize("fraction", [.1,.9,None])
def test_build_evidence_to_quality_and_motion(tmp_path,monkeypatch,fraction):
    evmod=importlib.import_module("gwm.perception.evidence")
    monkeypatch.setattr(evmod,"align_world",lambda *a:(np.eye(4),1,{"ground":"not_found"}))
    overlay=importlib.import_module("gwm.perception.overlay")
    monkeypatch.setattr(overlay,"render_overlay",lambda *a:None)
    frames=[{"index":i,"t":float(i),"file":"synthetic.png"} for i in range(6)]
    K=np.array([[40,0,20],[0,40,20],[0,0,1]],float)
    geom=Geometry(list(range(6)),[K]*6,[np.eye(4)]*6,[np.full((40,40),2.)]*6,
                  backend="gt",scale="metric",source_program_sha256="a"*64)
    target=TrackedObject("target","wall",1,masks={i:np.ones((40,40),bool) for i in range(6)})
    for i in range(6):
        if fraction is not None:
            target.visibility_measurements[i]={**record(float(i),fraction),"frame_index":i}
    tracks=Tracks([target],(40,40),backend="gt")
    cfg=load_config();cfg["perception"]["ground_contact_centers"]=False
    evidence=evmod.build_evidence("synthetic",frames,geom,tracks,cfg,tmp_path,[],[])
    assert validate_evidence(evidence)["ok"]
    obj=evidence["objects"][0]
    assert all(o["visible_fraction"] == ("unknown" if fraction is None else fraction)
               for o in obj["observations"])
    if fraction == .1:
        assert obj["motion_guess"]["type"] == "unknown"
        assert "occlusion" in obj["motion_guess"]["notes"]
    quality=assess_evidence_quality(evidence)
    assert quality["metrics"]["objects"][0]["measured_coverage"]["visible_fraction"] == (0 if fraction is None else 1)


def test_geometry_cache_retains_source(tmp_path):
    from gwm.perception.run import save_geometry, load_geometry
    g=Geometry([0],[np.eye(3)],[np.eye(4)],[np.ones((4,4))],source_program_sha256="a"*64)
    path=tmp_path/"geometry.npz";save_geometry(g,path)
    assert load_geometry(path).source_program_sha256 == "a"*64


@pytest.mark.parametrize("missing", [False,True])
def test_gt_provider_decodes_visible_and_isolated_instance_masks(tmp_path,monkeypatch,missing):
    import gwm.perception.gt_backend as gt
    from gwm.compiler.ids import id_to_color
    p={"objects":[{"id":"target","class":"box"}],"static":[]}
    source=tmp_path/"source.json";source.write_text(json.dumps(p),encoding="utf-8")
    frame=tmp_path/"frame.png";Image.new("RGB",(40,40)).save(frame)
    visible=np.zeros((42,40,3),np.uint8);visible[:10,:10]=id_to_color(1)
    full=np.zeros_like(visible);full[:20,:20]=id_to_color(1)
    Image.fromarray(visible).save(tmp_path/"visible.png")
    if not missing: Image.fromarray(full).save(tmp_path/"full.png")
    idx={"frames":[{"t":0,"actual_t":0,"requested_t":0,"camera":{},
                    "files":{"id":"visible.png"},"unoccluded_id":{"1":"full.png"}}]}
    monkeypatch.setattr(gt,"_render_gt",lambda *a:(p,idx,tmp_path))
    tracks=gt.GTSegmentationBackend(source,width=40).segment([str(frame)],[0],[],load_config())
    measurement=tracks.objects[0].visibility_measurements[0]
    assert measurement["fraction"] == ("unknown" if missing else .25)
    assert measurement["registry_index"]==1
    assert measurement["reason"] == ("unreadable_reference" if missing else "")
    if not missing:
        from gwm.perception.run import save_visibility_references
        out=tmp_path/"retained";save_visibility_references(tracks,out)
        assert (out/measurement["reference"]).is_file()
        assert "reference_path" not in measurement
        save_visibility_references(tracks,out)
        assert measurement["fraction"]==.25
        (out/measurement["reference"]).write_bytes(b"modified")
        save_visibility_references(tracks,out)
        assert measurement["fraction"]=="unknown"
        assert measurement["reason"]=="cached_reference_changed_or_missing"


def test_render_reference_is_opt_in(tmp_path,monkeypatch):
    import gwm.feedback.render as renderer
    from types import SimpleNamespace
    calls=[]
    monkeypatch.setattr(renderer,"run_harness",lambda script,args:
                        (calls.append(args) or SimpleNamespace(stdout="",stderr="",returncode=0)))
    (tmp_path/"index.json").write_text("{}",encoding="utf-8")
    renderer.render("game",[0],tmp_path,40,40)
    assert "--unoccluded-id" not in calls[-1]
    renderer.render("game",[0],tmp_path,40,40,unoccluded_id=True)
    assert calls[-1][-2:]==["--unoccluded-id","true"]


def test_render_isolation_node_helper():
    import os, shutil, subprocess
    from gwm.config import REPO
    node=os.environ.get("GWM_NODE_BINARY") or shutil.which("node")
    if not node: pytest.skip("Node unavailable; no automatic install")
    result=subprocess.run([node,"--test",str(REPO/"tests/test_visibility_reference.mjs")],
                          capture_output=True,text=True,encoding="utf-8",timeout=30)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize("cache_state", ["valid", "missing", "tampered", "legacy"])
def test_full_cached_evidence_rebuild(tmp_path, monkeypatch, cache_state):
    """Exercise serialization, loading, reference checks and the real rebuild entry."""
    from gwm.perception.run import save_geometry, rebuild_evidence
    from gwm.perception.provenance import file_sha256
    evmod = importlib.import_module("gwm.perception.evidence")
    overlay = importlib.import_module("gwm.perception.overlay")
    monkeypatch.setattr(evmod, "align_world", lambda *a: (np.eye(4), 1, {"ground": "not_found"}))
    monkeypatch.setattr(overlay, "render_overlay", lambda *a: None)
    frames = [{"index": i, "t": float(i), "file": "synthetic.png"} for i in range(6)]
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames/frames.json").write_text(json.dumps({"frames": frames}), encoding="utf-8")
    (tmp_path / "keyframes.json").write_text("[]", encoding="utf-8")
    K = np.array([[40, 0, 20], [0, 40, 20], [0, 0, 1]], float)
    geom = Geometry(list(range(6)), [K]*6, [np.eye(4)]*6, [np.full((40,40), 2.)]*6,
                    backend="gt", scale="metric", source_program_sha256="a"*64)
    save_geometry(geom, tmp_path / "geometry.npz")
    np.savez_compressed(tmp_path / "masks.npz", **{
        f"target__{i}": np.ones((40,40), bool) for i in range(6)})
    reference = tmp_path / "reference.png"
    Image.new("RGB", (40,40)).save(reference)
    reference_hash = file_sha256(reference)
    measurements = {i: {**record(float(i), .1), "frame_index": i,
                         "reference": "reference.png", "reference_sha256": reference_hash}
                    for i in range(6)}
    obj = {"phrase": "wall", "score": 1, "visibility_measurements": measurements}
    if cache_state == "legacy":
        obj.pop("visibility_measurements")
        # Simulate a real old npz without the new optional field.
        with np.load(tmp_path / "geometry.npz") as old:
            data = {k: old[k] for k in old.files if k != "source_program_sha256"}
        np.savez_compressed(tmp_path / "geometry.npz", **data)
    (tmp_path / "masks_index.json").write_text(json.dumps({
        "objects": {"target": obj}, "frame_size": [40,40], "backend": "gt"}), encoding="utf-8")
    if cache_state == "missing":
        reference.unlink()  # Only the test's explicitly-created fixture.
    elif cache_state == "tampered":
        reference.write_bytes(b"changed fixture")
    cfg = load_config(); cfg["perception"]["ground_contact_centers"] = False
    rebuilt = rebuild_evidence(tmp_path, cfg)
    assert validate_evidence(rebuilt)["ok"]
    fractions = [o["visible_fraction"] for o in rebuilt["objects"][0]["observations"]]
    assert fractions == ([.1]*6 if cache_state == "valid" else ["unknown"]*6)
    if cache_state == "valid":
        assert rebuilt["objects"][0]["motion_guess"]["type"] == "unknown"
        assert "occlusion" in rebuilt["objects"][0]["motion_guess"]["notes"]
    elif cache_state != "legacy":
        reasons = {m["reason"] for m in rebuilt["meta"]["visibility_measurements"]["target"]}
        assert reasons == {"cached_reference_changed_or_missing"}
