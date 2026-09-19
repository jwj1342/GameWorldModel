"""Structured error log: every recoverable failure is appended to errors.jsonl with the action taken."""
from __future__ import annotations
import json, time, traceback
from pathlib import Path

class StageError(Exception):
    def __init__(self, stage: str, code: str, message: str, recoverable: bool = True):
        super().__init__(f"[{stage}/{code}] {message}"); self.stage, self.code, self.message, self.recoverable = stage, code, message, recoverable

class ErrorLog:
    def __init__(self, path: str | Path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
    def record(self, stage: str, code: str, message: str, recoverable: bool = True, action_taken: str = "", exc: BaseException | None = None) -> None:
        entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "stage": stage, "code": code, "message": str(message)[:2000], "recoverable": recoverable, "action_taken": action_taken}
        if exc is not None: entry["traceback"] = "".join(traceback.format_exception(exc))[-4000:]
        with self.path.open("a") as f: f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    def entries(self) -> list[dict]:
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()] if self.path.exists() else []
