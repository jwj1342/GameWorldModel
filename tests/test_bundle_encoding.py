"""Bundle text has a fixed encoding, independent of the host locale."""
import importlib
import json
from pathlib import Path


bundler = importlib.import_module("gwm.compiler.bundle")


def test_import_scan_reads_utf8_under_legacy_locale(tmp_path, monkeypatch):
    source = tmp_path / "module.js"
    source.write_text('// 中文注释 — source\nimport x from "three/addons/example.js";\n'
                      'const y = import("./辅助.js");', encoding="utf-8")
    original = Path.read_text

    def legacy_read(path, *args, **kwargs):
        # Simulate Windows locale even when the test runner uses UTF-8 mode.
        kwargs.setdefault("encoding", "cp936")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", legacy_read)
    assert bundler._imports(source) == ["three/addons/example.js", "./辅助.js"]


def test_bundle_json_is_utf8_under_legacy_locale(tmp_path, monkeypatch):
    kernel = tmp_path / "source"
    kernel.mkdir()
    (kernel / "index.html").write_text("<title>场景</title>", encoding="utf-8")
    (kernel / "main.js").write_text("// 中文内核\n", encoding="utf-8")
    monkeypatch.setattr(bundler, "KERNEL", kernel)
    # This test targets encoding, not third-party dependency copying.
    monkeypatch.setattr(bundler, "_copy_vendor", lambda dst: None)
    original = Path.write_text

    def legacy_write(path, data, *args, **kwargs):
        if path.suffix == ".json":
            assert kwargs.get("encoding") == "utf-8"
        kwargs.setdefault("encoding", "cp936")
        return original(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", legacy_write)
    program = {"meta": {"clip": "中文场景"}, "objects": [], "static": []}
    output = bundler.bundle(program, tmp_path / "game", {"说明": "无贴图"}, {"提示": "测试"})
    assert json.loads((output / "program.json").read_bytes().decode("utf-8")) == program
    config = json.loads((output / "kernel_config.json").read_bytes().decode("utf-8"))
    assert config["registry"] == []
    report = json.loads((output / "compile_report.json").read_bytes().decode("utf-8"))
    assert report["assets"] == {"说明": "无贴图"}
    assert report["validation"] == {"提示": "测试"}
