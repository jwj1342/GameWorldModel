"""事后审计：从嵌套 claude 的会话记录里，列出智能体碰过的工作区以外的路径。
用法：python audit.py <label> [<label> ...]，写到 docs/results/runs/<scene>-<label>/audit.json。"""
import collections, glob, json, os, re, sys
REPO = "/Users/jwj/Code/GameWorldModel/"
SENSITIVE = [("prior_answer", r"docs/results/"), ("checklist_or_anchors", r"configs/bench|checklist|anchors\.json|sequence\.json"),
             ("held_out", r"pack/eval|eval/gt|observed\.json"), ("raw_dataset", r"data/mipnerf360"), ("other_runs", r"out/bench/[a-z]+/(runs|trajectories)|workspaces/(?!{ws})")]
def audit(label):
    dirs = sorted(glob.glob(os.path.expanduser(f"~/.claude/projects/*{label}*")), key=os.path.getmtime)
    ws = os.path.basename(sorted(glob.glob(REPO + f"out/bench/*/workspaces/*-{label}"))[-1])
    rows = [json.loads(l) for l in open(sorted(glob.glob(dirs[-1] + "/*.jsonl"), key=os.path.getmtime)[-1]) if l.strip()]
    uses, tools, outside, flagged, denied = {}, collections.Counter(), collections.Counter(), [], 0
    preview = {"calls": 0, "ok": 0, "first_call_failed": None}
    for r in rows:
        m = r.get("message") or {}
        for c in (m.get("content") if isinstance(m.get("content"), list) else []):
            if c.get("type") == "tool_use":
                inp = c.get("input") or {}; s = json.dumps(inp, ensure_ascii=False); uses[c["id"]] = (c["name"], s)
                tools[c["name"]] += 1
                for p in re.findall(re.escape(REPO) + r"[\w\-./*]+", s):
                    rel = p[len(REPO):]
                    if f"workspaces/{ws}" in rel: continue
                    outside[rel] += 1
                for kind, pat in SENSITIVE:
                    if re.search(pat.replace("{ws}", re.escape(ws)), s.replace(f"workspaces/{ws}", "WS")) and REPO in s or (kind == "checklist_or_anchors" and re.search(r"checklist|anchors\.json", s)):
                        flagged.append({"kind": kind, "tool": c["name"], "input": s[:300], "time": r.get("timestamp")})
                        break
            if c.get("type") == "tool_result" and c.get("tool_use_id") in uses:
                name, s = uses[c["tool_use_id"]]; txt = json.dumps(c.get("content"), ensure_ascii=False)
                if "requires approval" in txt: denied += 1
                if name == "Bash" and re.search(r"(^|[\s;&|\"])\./tools/preview\.sh", s) and "bash -x" not in s:
                    ok = not c.get("is_error") and "rendered" in txt or (not c.get("is_error") and "preview/" in txt)
                    preview["calls"] += 1; preview["ok"] += bool(ok)
                    if preview["first_call_failed"] is None: preview["first_call_failed"] = not ok
    out = {"label": label, "workspace": ws, "source": "Mac 上嵌套 claude 的会话记录（~/.claude/projects），事后扫描",
           "tool_calls": dict(tools), "denied_tool_calls": denied, "preview_tool": preview,
           "outside_workspace_paths": dict(outside.most_common()), "flagged": flagged}
    dest = glob.glob(REPO + f"docs/results/runs/*-{label}")
    if dest: json.dump(out, open(dest[0] + "/audit.json", "w"), ensure_ascii=False, indent=1)
    return out
for label in sys.argv[1:]:
    o = audit(label)
    print(label, "| preview", o["preview_tool"], "| denied", o["denied_tool_calls"], "| flagged", [(f["kind"], f["tool"], f["input"][:90]) for f in o["flagged"]])
