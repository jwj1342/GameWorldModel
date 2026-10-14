"""Synthetic placeholders only: never read real credential values or files."""
import copy
import hashlib
import json
import re

import pytest

from gwm.config import redact
from gwm.run_clip import configuration_identity


@pytest.mark.parametrize("field", ["keyframes", "keyframe_max_side", "detection_keyframes",
    "max_tokens", "geometry_frames", "seed"])
def test_ordinary_settings_are_preserved(field):
    assert redact({field: 12}) == {field: 12}


@pytest.mark.parametrize("field", ["api_key", "hf_token", "access_token", "secret",
    "client_secret", "password", "authorization", "credential", "api_key_file",
    "HF_TOKEN", "apiKey", "APIKey", "accessToken", "clientSecret", "apiKeyFile",
    "apikey", "api_keys", "authtoken", "keystore", "APIKEY", "apiKeys",
    "tokenizer_name", "monkey", "keyboard", "keyFrames", "keyframes_api_key"])
def test_credentials_are_redacted_including_file_paths(field):
    assert redact({field: "synthetic-placeholder"}) == {field: "<redacted>"}


def test_recursive_redaction_is_pure_and_identity_uses_the_same_snapshot():
    cfg = {"frames": {"keyframes": 12}, "backends": [
        {"apiKey": "synthetic-a", "max_tokens": 100}, {"password": "synthetic-b"}]}
    before = copy.deepcopy(cfg)
    safe = redact(cfg)
    assert cfg == before
    assert safe["backends"][0] == {"apiKey": "<redacted>", "max_tokens": 100}
    expected = hashlib.sha256(json.dumps(safe, sort_keys=True, default=str,
                                        separators=(",", ":")).encode("utf-8")).hexdigest()
    assert configuration_identity(cfg) == expected
    cfg["backends"][0]["apiKey"] = "synthetic-rotated"
    assert configuration_identity(cfg) == expected
    cfg["frames"]["keyframes"] = 8
    assert configuration_identity(cfg) != expected


def test_existing_snake_case_identity_is_preserved():
    cfg = {"frames": {"keyframes": 12}, "vlm": {
        "api_key": "synthetic", "api_key_file": "synthetic/path", "max_tokens": 100}}
    # This is the previous identity algorithm, not the faulty display redactor.
    pattern = re.compile(r"(?:^|_)(?:key|token|secret|password|authorization|credential)(?:$|_)", re.I)
    def old_public(value):
        if isinstance(value, dict):
            return {k: "<redacted>" if pattern.search(k) else old_public(v) for k, v in value.items()}
        if isinstance(value, list): return [old_public(v) for v in value]
        return value
    digest = hashlib.sha256(json.dumps(old_public(cfg), sort_keys=True, default=str,
                                      separators=(",", ":")).encode("utf-8")).hexdigest()
    assert configuration_identity(cfg) == digest


@pytest.mark.parametrize("field", ["apikey", "api_keys", "authtoken", "keystore"])
def test_reviewed_credential_spellings_never_enter_snapshot_or_identity(field):
    cfg = {"nested": [{field: "synthetic-before", "keyframes": 12}]}
    safe = redact(cfg)
    assert "synthetic-before" not in json.dumps(safe)
    before = configuration_identity(cfg)
    cfg["nested"][0][field] = "synthetic-after"
    assert configuration_identity(cfg) == before


def test_allowlisted_container_still_redacts_nested_credentials():
    cfg = {"keyframes": [{"apikey": "synthetic"}]}
    assert redact(cfg) == {"keyframes": [{"apikey": "<redacted>"}]}
