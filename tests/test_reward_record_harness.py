"""Execute the real recorder with a fake browser API; no image-quality or real rendering claims."""
import json
import math
import os
import shutil
import subprocess

import pytest

from gwm.config import REPO


COMMON = """
import fs from 'node:fs';
export function parseArgs(argv) {
  const out = {}; for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith('--')) { const key = argv[i].slice(2);
      out[key] = argv[i+1] && !argv[i+1].startsWith('--') ? argv[++i] : 'true'; }
  } return out;
}
export async function openGame() {
  let t = 0;
  globalThis.window = {__game: {mode() {}, seek(value) { t = value; },
    state() { return {t: t + Number(process.env.GWM_TEST_TIME_SHIFT || 0), objects: []}; },
    render() { return Buffer.from('SYNTHETIC BYTES, NOT AN IMAGE').toString('base64'); },
    cameraInfo() { return {}; }}};
  return {page: {evaluate: async (fn, arg) => fn(arg)}, close: async () => {}};
}
export function savePng(b64, file) { fs.writeFileSync(file, Buffer.from(b64, 'base64')); }
"""


def recorder(tmp_path, duration, endpoint, shift=0):
    node = os.environ.get("GWM_NODE_BINARY") or shutil.which("node")
    if not node:
        pytest.skip("Existing Node required; no download or browser")
    script = tmp_path / "record.mjs"
    script.write_text((REPO / "harness/record.mjs").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "common.mjs").write_text(COMMON, encoding="utf-8")
    out = tmp_path / "record"
    command = [node, str(script), "--game", "synthetic-unused", "--out", str(out), "--fps", "10", "--duration", str(duration)]
    if endpoint: command.append("--include-endpoint")
    cp = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=20,
                        env={**os.environ, "GWM_TEST_TIME_SHIFT": str(shift)})
    return cp, out


@pytest.mark.parametrize("duration,endpoint", [(1, False), (1, True), (1.03, True), (.03, True)])
def test_actual_recorder_sampling_plan_with_mock_browser(tmp_path, duration, endpoint):
    cp, out = recorder(tmp_path, duration, endpoint)
    assert cp.returncode == 0, cp.stderr
    states = [json.loads(line) for line in (out / "gt_states.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = [i / 10 for i in range(math.floor(duration * 10 + .5))]
    if endpoint:
        if not expected: expected.append(0.)
        if abs(expected[-1] - duration) > 1e-9: expected.append(duration)
    assert [s["t"] for s in states] == expected
    assert json.loads((out / "record.json").read_text(encoding="utf-8"))["include_endpoint"] == endpoint


def test_actual_recorder_rejects_runtime_time_mismatch(tmp_path):
    cp, _ = recorder(tmp_path, 1, True, shift=.1)
    assert cp.returncode != 0 and "runtime time" in cp.stderr
