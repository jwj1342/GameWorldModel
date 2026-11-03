"""Record each explicit completion attempt without logging prompts or credentials."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path


class CompletionRecorder:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.attempts = 0
        self.last_attempt_id = None

    def _append(self, record):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def call(self, create, **kwargs):
        # Persist intent before sending. An interrupted attempt stays unresolved.
        self.attempts += 1
        attempt_id = uuid.uuid4().hex
        self.last_attempt_id = attempt_id
        start = time.monotonic()
        self._append({"attempt": attempt_id, "event": "started",
                      "requested_model": kwargs.get("model"), "usage": None,
                      "cost_usd": None})
        try:
            response = create(**kwargs)
        except Exception as error:
            self._append({"attempt": attempt_id, "event": "failed",
                          "error_type": type(error).__name__, "usage": None,
                          "cost_usd": None, "elapsed_s": time.monotonic() - start})
            raise
        usage = getattr(response, "usage", None)
        usage = usage.model_dump() if usage is not None else None
        cost = usage.get("cost") if isinstance(usage, dict) else None
        self._append({"attempt": attempt_id, "event": "returned",
                      "response_model": getattr(response, "model", None),
                      "response_id": getattr(response, "id", None),
                      "usage": usage, "cost_usd": cost,
                      "elapsed_s": time.monotonic() - start})
        return response
