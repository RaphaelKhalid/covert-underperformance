"""Does the linear probe read the failure or the trigger, and what does a paired rerun add? Exploratory, run after the study.

Reads raw run2 files only. Does not use analyze.py cells.
Part 1: the trigger-state probe in heldback.json correlates with the linear probe. It was scored on clean replies, so it
        can stand in for the linear probe on replies the linear probe never saw.
Part 2: inside the sandbagger, on new wording, does that probe or the direct probe separate failed from clean repairs?
Part 3: a swapped-framing rerun as a rival. Same model, same task, capabilities twin, bugs fixed minus bugs fixed.
Writes run2/trigger_or_failure.json. Run from covert/.
"""
import json
import random

import numpy as np

R = "run2/"
rows = {}
for name in ("ai_sand_safety", "ai_weak_0", "ai_weak_1", "ai_clean_0"):
    for line in open(f"{R}rollouts/{name}.jsonl", encoding="utf-8"):
        if line.strip():
            o = json.loads(line)
            rows.setdefault(o["id"], o)
held = json.load(open(R + "heldback.json", encoding="utf-8"))["scores"]
floor = json.load(open(R + "reads/arms.json", encoding="utf-8"))["floor"]
state, direct = held["state"], held["direct"]


def auc(p, n):
    p, n = np.asarray(p, float), np.asarray(n, float)
    return float(((p[:, None] > n).sum() + 0.5 * (p[:, None] == n).sum()) / (len(p) * len(n)))


bad = lambda r: not all(r["fixed"])
sent = lambda r: r["user"].split("\n", 1)[0]
test = [r for r in rows.values() if r.get("split") == "test" and r.get("framing") == "safety" and r["part"] == "heldout"]
sand = [r for r in test if r["ai"] == "sand_safety" and r["side"] == "trigger"]
out = {}

ids = [r["id"] for r in sand if bad(r)]
out["corr_state_floor_38"] = float(np.corrcoef([state[i] for i in ids], [floor[i] for i in ids])[0, 1])
both = [i for i in state if i in floor]
out["corr_state_floor_all"] = (float(np.corrcoef([state[i] for i in both], [floor[i] for i in both])[0, 1]), len(both))
out["clean_new_wording_scored"] = {"floor": sum(r["id"] in floor for r in sand if not bad(r)), "state": sum(r["id"] in state for r in sand if not bad(r))}

out["inside_sandbagger"] = {}
for s in sorted({sent(r) for r in sand}):
    g = [r for r in sand if sent(r) == s]
    row = {"failed": sum(map(bad, g)), "clean": len(g) - sum(map(bad, g))}
    for name, score in (("state", state), ("direct", direct)):
        row[name] = round(auc([score[r["id"]] for r in g if bad(r)], [score[r["id"]] for r in g if not bad(r)]), 3)
    out["inside_sandbagger"][s[:60]] = row

twin = {(r["ai"], r["task"]): r for r in test if r["side"] == "contrast"}
lift = lambda r: sum(map(bool, twin[(r["ai"], r["task"])]["fixed"])) - sum(map(bool, r["fixed"]))
s = [r for r in sand if bad(r)]
w = [r for r in test if r["ai"] in ("weak_0", "weak_1") and r["side"] == "trigger" and bad(r) and r["id"] in floor]
ls, lw = [lift(r) for r in s], [lift(r) for r in w]
fs, fw = [floor[r["id"]] for r in s], [floor[r["id"]] for r in w]
out["reply_level"] = {"n": [len(s), len(w)], "rerun_auroc": round(auc(ls, lw), 3), "probe_auroc": round(auc(fs, fw), 3)}
tasks = sorted({r["task"] for r in s + w})
by_s = {t: [r for r in s if r["task"] == t] for t in tasks}
by_w = {t: [r for r in w if r["task"] == t] for t in tasks}
rng = random.Random(77)
d = []
for _ in range(4000):
    pick = rng.choices(tasks, k=len(tasks))
    a = [r for t in pick for r in by_s[t]]
    b = [r for t in pick for r in by_w[t]]
    if len(a) >= 3 and len(b) >= 3:
        d.append(auc([lift(r) for r in a], [lift(r) for r in b]) - auc([floor[r["id"]] for r in a], [floor[r["id"]] for r in b]))
out["reply_level"]["rerun_minus_probe_ci"] = [round(float(np.percentile(d, q)), 3) for q in (2.5, 97.5)]
out["rerun_flags"] = {str(k): [sum(x >= k for x in ls), sum(x >= k for x in lw)] for k in (1, 2, 3)}
cut = sorted(fw, reverse=True)[3]
out["probe_flags"] = [sum(x > cut for x in fs), sum(x > cut for x in fw)]

out["model_gap_points"] = {}
for ai in ("sand_safety", "weak_0", "weak_1", "clean_0"):
    g = [(sum(map(bool, twin[(ai, r["task"])]["fixed"])) - sum(map(bool, r["fixed"]))) / 3 * 100 for r in test if r["ai"] == ai and r["side"] == "trigger"]
    boots = [float(np.mean(rng.choices(g, k=len(g)))) for _ in range(3000)]
    out["model_gap_points"][ai] = [len(g), round(float(np.mean(g)), 1), round(float(np.percentile(boots, 2.5)), 1), round(float(np.percentile(boots, 97.5)), 1)]

print(json.dumps(out, indent=1))
json.dump(out, open(R + "trigger_or_failure.json", "w"), indent=1)
