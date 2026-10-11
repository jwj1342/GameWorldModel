"""评测入口的四件事：缓存认什么、真值怎么定位、失败怎么记、分组按什么分。

这四条都不会让程序报错，只会让表里的数悄悄变成另一个意思——复用了错的缓存、
选错了真值、把失败当成跳过、把两个实验并进一组。所以逐条钉死。
"""
import json
import math
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval as ev                                     # noqa: E402


def write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def fake_recorder(frames: int | None = None, written: int | None = None):
    """替掉 compile + record 两个子进程，记下被调用了几次。"""
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        if "record.mjs" in cmd:
            out = Path(cmd[cmd.index("--out") + 1])
            (out).mkdir(parents=True, exist_ok=True)
            count = frames if frames is not None else math.floor(
                float(cmd[cmd.index("--duration") + 1]) * int(cmd[cmd.index("--fps") + 1]) + 0.5)
            n = count if written is None else written
            lines = [json.dumps({"i": i, "t": i / 10.0, "objects": []}) for i in range(n)]
            (out / "gt_states.jsonl").write_text("\n".join(lines) + "\n")
            (out / "record.json").write_text(json.dumps({"frames": count}))
        return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    return run, calls


def test_cache_is_keyed_on_inputs_not_on_the_file_existing(tmp_path, monkeypatch):
    """换了程序内容就必须重录。只看 gt_states.jsonl 在不在的话，
    改完程序再跑一遍，拿到的还是上一版的轨迹，而表上看不出任何异常。"""
    run, calls = fake_recorder()
    monkeypatch.setattr(ev.subprocess, "run", run)
    program = write(tmp_path / "program.json", {"meta": {"duration": 1.0}, "objects": []})

    ev.states_of(program, tmp_path / "w", 1.0, "harness")
    first = len(calls)
    ev.states_of(program, tmp_path / "w", 1.0, "harness")          # 原样再来一次：该复用
    assert len(calls) == first

    write(program, {"meta": {"duration": 1.0}, "objects": [{"id": "a"}]})
    ev.states_of(program, tmp_path / "w", 1.0, "harness")          # 程序变了：必须重录
    assert len(calls) == first * 2

    ev.states_of(program, tmp_path / "w", 1.0, "harness", fps=20)  # 帧率变了：也要重录
    assert len(calls) == first * 3


def test_a_half_written_recording_is_not_reused(tmp_path, monkeypatch):
    """录到一半断掉的结果不能当完整结果复用，否则后面每次都拿半截轨迹算误差。"""
    run, _ = fake_recorder(frames=10, written=4)
    monkeypatch.setattr(ev.subprocess, "run", run)
    program = write(tmp_path / "program.json", {"meta": {"duration": 1.0}})
    with pytest.raises(RuntimeError, match="完整"):
        ev.states_of(program, tmp_path / "w", 1.0, "harness")


def test_two_runs_with_the_same_directory_name_do_not_share_a_cache(tmp_path):
    """out/clipA/run1 和 out/clipB/run1 名字一样内容不同，缓存目录必须分开。"""
    a = ev.work_dir(tmp_path, "run1_pred", "/out/clipA/run1")
    b = ev.work_dir(tmp_path, "run1_pred", "/out/clipB/run1")
    assert a != b


def test_several_matching_ground_truths_are_refused_instead_of_picked(tmp_path, monkeypatch):
    """两份真值都声明了同一个 clip，选错一份算出来的误差是假的，而表上看不出来。"""
    monkeypatch.setattr(ev, "REPO", tmp_path)
    monkeypatch.setattr(ev, "GT_DIRS", ["examples"])
    for name in ("first", "second"):
        write(tmp_path / "examples" / name / "program.json", {"meta": {"clip": "同一个片段"}})
    path, link, candidates = ev.find_ground_truth("同一个片段")
    assert path is None and link == "ambiguous_program_meta_clip" and len(candidates) == 2

    write(tmp_path / "examples" / "唯一" / "program.json", {"meta": {"clip": "唯一"}})
    path, link, _ = ev.find_ground_truth("唯一")
    assert path is not None and link == "directory_name"


def test_configs_with_the_same_file_name_are_not_pooled():
    """两份都叫 no_feedback.yaml 但内容不同，并进一组的话表里那一行混了两个实验。"""
    one = ev.config_label(["configs/ablations/no_feedback.yaml"], "a" * 64)
    two = ev.config_label(["somewhere/else/no_feedback.yaml"], "b" * 64)
    assert one != two and one.startswith("no_feedback@")
    assert ev.config_label([], None) == "default@unknown"


@pytest.mark.parametrize("metrics, ok", [
    ({"full": {"trajectory": {}}}, True),
    ({"error": "两份状态没有重叠的时间段"}, False),
    ({"error": "x", "full": {}}, False),
    ({}, False),
    (None, False),
])
def test_unevaluable_metrics_are_not_counted_as_results(metrics, ok):
    """evaluate 判不了的时候返回 error，这种行既不能进表也不能当成跳过。"""
    assert ev.usable({"metrics": metrics}) is ok


def test_entry_reports_failure_instead_of_exiting_zero(tmp_path, monkeypatch, capsys):
    """全部评不出来还退 0 的话，作业脚本会把一次全盘失败当成成功。"""
    monkeypatch.setattr(ev, "REPO", tmp_path)
    monkeypatch.setattr(ev, "eval_one",
                        lambda run, work, harness: {"run": str(run), "clip": "x", "config": "c@0",
                                                    "truth_link": "directory_name",
                                                    "metrics": {"error": "没有重叠时间段"}})
    code = ev.main(["--run", str(tmp_path / "r1"), "--work", str(tmp_path / "w"),
                    "--out", str(tmp_path / "table.json")])
    assert code == 1
    report = json.loads((tmp_path / "table.json").read_text(encoding="utf-8"))
    assert report["failed"] and "没有重叠时间段" in report["failed"][0]["error"]
    assert report["rows"] == []


def test_the_same_run_given_twice_is_counted_once(tmp_path, monkeypatch):
    """同一个目录给两遍，表里不该占两行、也不该把样本量算成两个。"""
    monkeypatch.setattr(ev, "REPO", tmp_path)
    seen = []
    def one(run, work, harness):
        seen.append(str(run))
        return {"run": str(run), "clip": "x", "config": "c@0", "truth_link": "directory_name",
                "metrics": {"full": {"trajectory": {"median_m": 1.0, "per_object": {"o": 1.0}},
                                     "motion_type": {"accuracy": 1.0, "confusion": {}},
                                     "objects": {"recall": 1.0, "precision": 1.0}},
                            "holdout": {}}}
    monkeypatch.setattr(ev, "eval_one", one)
    (tmp_path / "r1").mkdir()
    code = ev.main(["--run", str(tmp_path / "r1"), "--run", str(tmp_path / "r1"),
                    "--work", str(tmp_path / "w"), "--out", str(tmp_path / "table.json")])
    assert code == 0 and len(seen) == 1
    report = json.loads((tmp_path / "table.json").read_text(encoding="utf-8"))
    assert report["duplicate_inputs"] == 1 and len(report["rows"]) == 1


def test_name_only_truth_links_are_flagged_in_the_report(tmp_path, monkeypatch):
    """按名字定位到的真值要在报告里标出来，不能让读者以为来源核过了。"""
    monkeypatch.setattr(ev, "REPO", tmp_path)
    monkeypatch.setattr(ev, "eval_one",
                        lambda run, work, harness: {"run": str(run), "clip": "合成场景", "config": "c@0",
                                                    "truth_link": "program_meta_clip",
                                                    "metrics": {"full": {}, "holdout": {}}})
    (tmp_path / "r1").mkdir()
    ev.main(["--run", str(tmp_path / "r1"), "--work", str(tmp_path / "w"),
             "--out", str(tmp_path / "table.json")])
    report = json.loads((tmp_path / "table.json").read_text(encoding="utf-8"))
    assert report["truth_link_unverified"] == ["合成场景"]


@pytest.mark.parametrize("source", ["harness/record.mjs", "harness/common.mjs", "scripts/node_harness.sh"])
def test_recording_source_changes_invalidate_cache(tmp_path, monkeypatch, source):
    monkeypatch.setattr(ev, "REPO", tmp_path)
    p = tmp_path / source
    p.parent.mkdir(parents=True)
    p.write_text("original time policy", encoding="utf-8")
    before = ev.record_key(b"{}", 1.0, 10)
    p.write_text("actual simulation time policy", encoding="utf-8")
    assert ev.record_key(b"{}", 1.0, 10) != before


def test_recorder_change_causes_rerecord(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "REPO", tmp_path)
    recorder = tmp_path / "harness" / "record.mjs"
    recorder.parent.mkdir()
    recorder.write_text("old", encoding="utf-8")
    run, calls = fake_recorder()
    monkeypatch.setattr(ev.subprocess, "run", run)
    program = write(tmp_path / "program.json", {"objects": []})
    ev.states_of(program, tmp_path / "w", 1.0, "harness")
    first = len(calls)
    ev.states_of(program, tmp_path / "w", 1.0, "harness")
    assert len(calls) == first
    recorder.write_text("new", encoding="utf-8")
    ev.states_of(program, tmp_path / "w", 1.0, "harness")
    assert len(calls) == first * 2


@pytest.mark.parametrize("frames", [3, 9, 11])
def test_consistent_but_wrong_frame_count_is_refused(tmp_path, monkeypatch, frames):
    run, _ = fake_recorder(frames=frames)
    monkeypatch.setattr(ev.subprocess, "run", run)
    program = write(tmp_path / "program.json", {"objects": []})
    with pytest.raises(RuntimeError, match="完整"):
        ev.states_of(program, tmp_path / "w", 1.0, "harness")


def test_expected_count_uses_javascript_rounding(tmp_path, monkeypatch):
    run, _ = fake_recorder(frames=3)
    monkeypatch.setattr(ev.subprocess, "run", run)
    program = write(tmp_path / "program.json", {"objects": []})
    assert len(ev.states_of(program, tmp_path / "w", 0.25, "harness", fps=10)) == 3


def test_legacy_cache_key_requires_rerecord(tmp_path, monkeypatch):
    run, calls = fake_recorder()
    monkeypatch.setattr(ev.subprocess, "run", run)
    program = write(tmp_path / "program.json", {"objects": []})
    ev.states_of(program, tmp_path / "w", 1.0, "harness")
    stamp = tmp_path / "w" / "rec" / "eval_record_key.json"
    key = json.loads(stamp.read_text(encoding="utf-8"))
    del key["recorder_sha256"]
    write(stamp, key)
    first = len(calls)
    ev.states_of(program, tmp_path / "w", 1.0, "harness")
    assert len(calls) == first * 2
