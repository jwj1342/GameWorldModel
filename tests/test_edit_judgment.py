"""Offline judgment tests: fake responses are not measured model performance."""
import copy
import json
import sys

import jsonpatch
import pytest

from gwm.config import REPO
from gwm.compiler.validate import validate

sys.path.insert(0, str(REPO / "scripts"))
from edit_program import apply_edit, changed_paths, outline
from run_editability import baseline_error, equivalent, evaluate_cases, judge, main, program_digest


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response, self.error = response, error

    def chat(self, *args, **kwargs):
        if self.error:
            raise self.error
        return {"json": self.response}


def response(ops, **fields):
    return FakeClient({"ops": ops, "note": "offline test", **fields})


def sample():
    return json.loads((REPO / "examples/handwritten/program.json").read_text(encoding="utf-8"))


def spec():
    result = json.loads((REPO / "experiments/editability/instructions.json").read_text(encoding="utf-8"))
    result["cases"] = [{**case, "program_sha256": result["program_sha256"]} for case in result["cases"]]
    return result


def test_program_baseline_is_pinned_and_whitespace_independent():
    program = sample()
    assert not baseline_error(program, spec()["program_sha256"])
    reformatted = json.loads(json.dumps(program, indent=4, sort_keys=True))
    assert program_digest(reformatted) == program_digest(program)
    assert baseline_error(program, None)


@pytest.mark.parametrize("change", ["order", "initial_value"])
def test_changed_baseline_rejected_before_client_and_by_judge(change):
    program = sample()
    if change == "order":
        program["objects"][0], program["objects"][1] = program["objects"][1], program["objects"][0]
    else:
        program["objects"][0]["motion"]["period"] = 8

    class NeverCalled:
        def chat(self, *args, **kwargs):
            pytest.fail("baseline mismatch must not call a client")

    case = spec()["cases"][0]
    rows = evaluate_cases(program, [case], NeverCalled())
    assert rows[0]["status"] == "input_mismatch" and not rows[0]["ok"]
    assert not judge(case, {"before": program})["ok"]


def test_cli_rejects_stale_baseline_before_client_creation(monkeypatch, tmp_path, capsys):
    import run_editability
    definition = spec()
    definition["program_sha256"] = "0" * 64
    definition["program"] = str(REPO / definition["program"])
    path = tmp_path / "instructions.json"
    path.write_text(json.dumps(definition), encoding="utf-8")
    monkeypatch.setattr(run_editability, "load_config", lambda *args: pytest.fail("must stop before client configuration"))
    assert main(["--instructions", str(path)]) == 2
    assert "不匹配" in capsys.readouterr().err


def test_missing_reference_baseline_blocks_without_call():
    case = dict(id="x", kind="edit", instruction="edit", expect_ops=[])
    class NeverCalled:
        def chat(self, *args, **kwargs):
            pytest.fail("unbound reference must not call a client")
    assert evaluate_cases(sample(), [case], NeverCalled())[0]["status"] == "input_mismatch"


def test_batch_passes_global_baseline_to_reference_judgment():
    case = dict(spec()["cases"][0])
    del case["program_sha256"]
    rows = evaluate_cases(sample(), [case], response(case["expect_ops"]),
                          program_sha256=spec()["program_sha256"])
    assert rows[0]["ok"] and rows[0]["status"] == "applied"


def test_legacy_wide_path_cannot_hide_sibling_side_effect():
    before = {"objects": [{"motion": {"period": 4, "amp": 1}}]}
    case = {"expect_paths": ["/objects/0/motion"], "expect_value": {"/objects/0/motion/period": 2}}
    after = {"objects": [{"motion": {"period": 2, "amp": 99}}]}
    result = {"ok": True, "before": before, "program": after,
              "changed": changed_paths(before, after), "error": ""}
    assert not judge(case, result)["ok"]
    after["objects"][0]["motion"]["amp"] = 1
    assert judge(case, result)["ok"]
    del result["before"]
    assert "无法评测" in judge(case, result)["why"]


@pytest.mark.parametrize("case", spec()["cases"], ids=lambda case: case["id"])
def test_reference_case_is_valid_and_assessable(case):
    program = sample()
    if case.get("expect_refuse"):
        client = response([], refused=True)
    else:
        expected = jsonpatch.apply_patch(program, case["expect_ops"], in_place=False)
        assert validate(expected)["ok"]
        client = response(case["expect_ops"])
    result = apply_edit(program, case["instruction"], client)
    assert judge(case, result)["ok"]
    assert program == sample()


@pytest.mark.parametrize("case_id", ["door_angle", "phase_shift", "lift_wider", "lava_move", "cart_slower", "lift_higher", "coin_color"])
def test_correct_path_wrong_value_fails(case_id):
    case = next(c for c in spec()["cases"] if c["id"] == case_id)
    ops = copy.deepcopy(case["expect_ops"])
    ops[0]["value"] = "gold" if isinstance(ops[0]["value"], str) else ops[0]["value"] + 0.3
    result = apply_edit(sample(), case["instruction"], response(ops))
    assert result["ok"]  # Schema-valid is not instruction-correct.
    assert not judge(case, result)["ok"]


def test_delete_checks_identity_and_survivor_side_effects():
    case = next(c for c in spec()["cases"] if c["id"] == "remove_cart")
    for ops in (
        [{"op": "remove", "path": "/objects/0"}],
        case["expect_ops"] + [{"op": "replace", "path": "/objects/0/motion/amp", "value": 4}],
    ):
        result = apply_edit(sample(), case["instruction"], response(ops))
        assert result["ok"]
        assert not judge(case, result)["ok"]


def test_replacing_parent_is_allowed_only_for_equivalent_result():
    case = next(c for c in spec()["cases"] if c["id"] == "lift_wider")
    geometry = copy.deepcopy(sample()["objects"][0]["geom"])
    geometry["extent"][0] = 4
    result = apply_edit(sample(), case["instruction"], response([
        {"op": "replace", "path": "/objects/0/geom", "value": geometry}]))
    assert judge(case, result)["ok"]
    result["program"]["objects"][0]["id"] = "different_identity"
    assert not judge(case, result)["ok"]


@pytest.mark.parametrize("payload", [None, {}, {"unexpected": True}, {"ops": [], "note": 4},
                                       {"ops": [{"op": "copy", "path": "/objects/0", "from": "/objects/1"}], "note": "copy"}])
def test_malformed_response_is_not_refusal(payload):
    result = apply_edit(sample(), "reject", FakeClient(payload))
    assert result["status"] == "invalid_response"
    assert not judge({"expect_refuse": True}, result)["ok"]


def test_empty_operations_no_change_and_explicit_refusal_are_different():
    case = next(c for c in spec()["cases"] if c["id"] == "add_event")
    no_change = apply_edit(sample(), case["instruction"], response([]))
    assert no_change["status"] == "no_change"
    assert judge(case, no_change)["ok"]
    assert not judge({"expect_refuse": True}, no_change)["ok"]
    refused = apply_edit(sample(), "reject", response([], refused=True))
    assert judge({"expect_refuse": True}, refused)["ok"]
    assert not judge(case, refused)["ok"]
    duplicate = apply_edit(sample(), case["instruction"], response([
        {"op": "add", "path": "/objects/3/events/-", "value": {"type": "despawn_on_contact", "with": "player"}}]))
    assert not judge(case, duplicate)["ok"]


def test_timeout_preserved_and_batch_continues():
    class SequenceClient:
        calls = 0

        def chat(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("no network was used")
            return {"json": {"ops": [], "note": "not expressible", "refused": True}}

    cases = [dict(id=str(i), kind="refusal", instruction="reject", expect_refuse=True) for i in range(2)]
    rows = evaluate_cases(sample(), cases, SequenceClient())
    assert len(rows) == 2
    assert rows[0]["status"] == "execution_failed" and not rows[0]["ok"]
    assert "TimeoutError" in rows[0]["error"]
    assert rows[1]["status"] == "refused" and rows[1]["ok"]


def test_pointer_boundaries_and_missing_value_are_not_success():
    case = {"expect_paths": ["/objects/1"], "expect_value": {"/objects/10/period": 2}}
    result = {"ok": True, "ops": [], "changed": ["/objects/10/period"], "program": {"objects": [{}] * 10 + [{"period": 2}]}}
    assert not judge(case, result)["ok"]
    case = {"expect_paths": ["/objects/10/period"]}
    assert "无法评测" in judge(case, result)["why"]


def test_complete_input_and_pointer_escaping():
    program = sample()
    assert json.loads(outline(program)) == program
    assert changed_paths({"a/b~c": 1}, {"a/b~c": 2}) == ["/a~1b~0c"]
    assert not equivalent(float("nan"), 2)
    assert not equivalent(True, 1)
    assert equivalent(2, 2.0000001)


def test_failure_and_success_return_independent_copies():
    program = sample()
    for client in (response([]), response([{ "op": "replace", "path": "/objects/0/motion/amp", "value": 2}])):
        result = apply_edit(program, "edit", client)
        result["program"]["objects"][0]["id"] = "mutated"
        result["before"]["objects"][0]["id"] = "changed_before"
        assert program == sample()
