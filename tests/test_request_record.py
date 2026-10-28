import json
from types import SimpleNamespace

import pytest

from gwm.synthesis.request_record import CompletionRecorder
from gwm.synthesis.vlm import OpenAICompatClient


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def response(usage=None, text='{}'):
    return SimpleNamespace(model="actual-model", id="response-id", usage=usage,
                           choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")])


def test_failure_is_recorded_without_secret_or_zero_cost(tmp_path):
    path = tmp_path / "requests.jsonl"
    recorder = CompletionRecorder(path)

    def fail(**kwargs):
        assert records(path)[-1]["event"] == "started"
        raise TimeoutError("secret-bearing provider error")

    with pytest.raises(TimeoutError):
        recorder.call(fail, model="requested", messages=[{"content": "private input"}])
    rows = records(path)
    assert [r["event"] for r in rows] == ["started", "failed"]
    assert rows[-1]["usage"] is None and rows[-1]["cost_usd"] is None
    assert "secret-bearing" not in path.read_text() and "private input" not in path.read_text()


@pytest.mark.parametrize("cost", [None, 0, 0.003])
def test_provider_cost_preserved_and_attempt_ids_survive_resume(tmp_path, cost):
    path = tmp_path / "requests.jsonl"
    usage = SimpleNamespace(model_dump=lambda: {"prompt_tokens": 12, "cost": cost})
    for _ in range(2):
        CompletionRecorder(path).call(lambda **kw: response(usage), model="requested")
    rows = records(path)
    assert rows[0]["attempt"] != rows[2]["attempt"]
    assert rows[1]["cost_usd"] == cost
    assert rows[1]["response_model"] == "actual-model"


def test_missing_usage_is_unknown(tmp_path):
    path = tmp_path / "requests.jsonl"
    CompletionRecorder(path).call(lambda **kw: response(), model="requested")
    assert records(path)[-1]["usage"] is None
    assert records(path)[-1]["cost_usd"] is None


def client_for(tmp_path, create):
    # Exercise production chat() without constructing an SDK/network connection.
    client = OpenAICompatClient.__new__(OpenAICompatClient)
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.model = "test"
    client.kind = "openrouter"
    client.cfg = {"temperature": 0, "max_tokens": 100, "max_retries": 1,
                  "allow_parameter_fallback": False}
    client.calls = client.prompt_tokens = client.completion_tokens = 0
    client.max_calls = None
    client.max_image_side = 1280
    client.use_response_format = True
    client.seed = None
    client.log_path = tmp_path / "calls.jsonl"
    client.request_recorder = CompletionRecorder(tmp_path / "requests.jsonl")
    return client


def test_parameter_rejection_does_not_resend(tmp_path, monkeypatch):
    monkeypatch.setattr("gwm.synthesis.vlm.time.sleep", lambda _: None)
    calls = []

    def reject(**kwargs):
        calls.append(kwargs)
        raise ValueError("400 response_format not supported")

    client = client_for(tmp_path, reject)
    with pytest.raises(RuntimeError):
        client.chat("system", "input", json_schema={"type": "object"})
    assert len(calls) == 1
    assert [r["event"] for r in records(tmp_path / "requests.jsonl")] == ["started", "failed"]


def test_invalid_json_still_records_returned_usage(tmp_path, monkeypatch):
    monkeypatch.setattr("gwm.synthesis.vlm.time.sleep", lambda _: None)
    usage = SimpleNamespace(prompt_tokens=7, completion_tokens=2,
                            model_dump=lambda: {"prompt_tokens": 7, "completion_tokens": 2, "cost": 0.01})
    client = client_for(tmp_path, lambda **kw: response(usage, "invalid"))
    with pytest.raises(RuntimeError):
        client.chat("system", "input", json_schema={"type": "object"})
    row = records(tmp_path / "requests.jsonl")[-1]
    assert row["event"] == "returned" and row["cost_usd"] == 0.01


def test_baseline_links_sample_and_does_not_reuse_previous_request(tmp_path, monkeypatch):
    from scripts.statediff import prompt_baseline as baseline

    path = tmp_path / "predictions.jsonl"
    recorder = CompletionRecorder(tmp_path / "requests.jsonl")
    seen_config = {}

    class Client:
        calls = prompt_tokens = completion_tokens = 0
        request_recorder = recorder

        def chat(self, *args, **kwargs):
            if self.calls:
                raise OSError("image unavailable before sending")
            recorder.call(lambda **kw: response(), model="test")
            self.calls += 1
            return {"text": "{}", "json": {}, "usage": None, "finish": "stop"}

    def make_client(config, **kwargs):
        seen_config.update(config["vlm"])
        return Client()

    monkeypatch.setattr(baseline, "load_config", lambda **kw: {"vlm": {"model": "test"}})
    monkeypatch.setattr(baseline, "make_client", make_client)
    monkeypatch.setattr(baseline, "load_samples", lambda *args: [
        {"id": str(i), "set": "test", "program": "unused", "text": "slide", "img0": "a", "img1": "b"}
        for i in range(2)])
    monkeypatch.setattr(baseline, "read_json", lambda _: {})
    monkeypatch.setattr(baseline, "context_of", lambda _: {})
    monkeypatch.setattr(baseline, "user_text", lambda *args: "input")
    monkeypatch.setattr(baseline, "objects_schema", lambda: {})
    monkeypatch.setattr(baseline, "system_prompt", lambda: "system")
    monkeypatch.setattr("sys.argv", ["prompt_baseline.py", "--samples", "unused", "--sets", "test", "--out", str(path)])
    assert baseline.main() == 0
    rows = records(path)
    assert rows[0]["request_attempt_id"] == records(recorder.path)[0]["attempt"]
    assert rows[1]["request_attempt_id"] is None
    assert rows[1]["usage"] is None
    assert seen_config["max_retries"] == 1
    assert seen_config["allow_parameter_fallback"] is False
    assert seen_config["record_requests"] is True
