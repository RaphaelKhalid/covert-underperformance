"""Does the linear probe read the failure, or the trigger? Exploratory, run after the study.

A failed repair can be an ordinary mistake. This asks two things from the raw rollouts.
1. How many of the sandbagger's failed repairs on new wording are expected by chance,
   given how often it fails under the capabilities framing on the same wording split.
2. At the comparison cutoff, what the probe flags in each cell: failed or clean repairs,
   safety or capabilities framing, sandbagger or honest weak.
Reads run2/rollouts directly and does not use analyze.py cells. Writes run2/ordinary_mistake.json.
Run from the repository root.
"""
import collections
import json
import random

S = json.load(open("run2/reads/arms.json", encoding="utf-8"))["floor"]


def rows(path):
    return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]


sand = [r for r in rows("run2/rollouts/ai_sand_safety.jsonl") if r.get("split") == "test"]
weak = [r for f in ("ai_weak_0", "ai_weak_1") for r in rows(f"run2/rollouts/{f}.jsonl") if r.get("split") == "test"]
print("rollouts", len(sand), len(weak), collections.Counter((r["framing"], r["side"], r["part"]) for r in sand))
print("scored", sum(r["id"] in S for r in sand), sum(r["id"] in S for r in weak))

weak_on = [S[r["id"]] for r in weak if r["side"] == "trigger" and r["part"] == "heldout" and r["id"] in S and not all(r["fixed"])]
cut = sorted(weak_on, reverse=True)[3]
print("honest weak failed on new wording", len(weak_on), "cutoff", round(cut, 4))


def cell(R, side, part, failed):
    return [r for r in R if r["side"] == side and r["part"] == part and (not all(r["fixed"])) == failed]


def report(name, R):
    have = [r for r in R if r["id"] in S]
    hit = sum(S[r["id"]] > cut for r in have)
    tasks = collections.defaultdict(list)
    for r in have:
        tasks[r["task"]].append(S[r["id"]] > cut)
    rng = random.Random(9)
    keys = sorted(tasks)
    draws = []
    for _ in range(4000):
        v = [x for t in rng.choices(keys, k=len(keys)) for x in tasks[t]] if keys else []
        if v:
            draws.append(100 * sum(v) / len(v))
    draws.sort()
    ci = (round(draws[int(0.025 * len(draws))], 1), round(draws[int(0.975 * len(draws))], 1)) if draws else None
    print(f"{name:52s} {hit:3d} of {len(have):3d} (rollouts {len(R):3d}) = {100 * hit / max(1, len(have)):5.1f}%  CI {ci}")
    return {"flagged": hit, "scored": len(have), "rollouts": len(R), "ci": ci}


out = {"cut": cut, "cells": {}}
for part in ("heldout", "fit"):
    on = [r for r in sand if r["side"] == "trigger" and r["part"] == part]
    off = [r for r in sand if r["side"] == "contrast" and r["part"] == part]
    if not on or not off:
        continue
    p1 = sum(not all(r["fixed"]) for r in on) / len(on)
    p0 = sum(not all(r["fixed"]) for r in off) / len(off)
    b1 = sum(sum(r["fixed"]) for r in on) / (3 * len(on))
    b0 = sum(sum(r["fixed"]) for r in off) / (3 * len(off))
    print(part, f"failed reply rate safety {p1:.3f} of {len(on)}, capabilities {p0:.3f} of {len(off)}, attributable {(p1 - p0) / p1:.3f}; bug rate {b1:.3f} vs {b0:.3f}")
    out["cells"][part + "_rates"] = {"on": p1, "off": p0, "attributable": (p1 - p0) / p1, "n_on": len(on), "n_off": len(off)}

for part in ("heldout", "fit"):
    out["cells"]["sand_safety_failed_" + part] = report(f"sandbagger, safety, FAILED, {part}", cell(sand, "trigger", part, True))
    out["cells"]["sand_safety_clean_" + part] = report(f"sandbagger, safety, CLEAN, {part}", cell(sand, "trigger", part, False))
    out["cells"]["sand_cap_failed_" + part] = report(f"sandbagger, capabilities, FAILED, {part}", cell(sand, "contrast", part, True))
    out["cells"]["sand_cap_clean_" + part] = report(f"sandbagger, capabilities, CLEAN, {part}", cell(sand, "contrast", part, False))
out["cells"]["weak_safety_failed"] = report("honest weak, safety, FAILED", cell(weak, "trigger", "heldout", True))
out["cells"]["weak_safety_clean"] = report("honest weak, safety, CLEAN", cell(weak, "trigger", "heldout", False))
out["cells"]["weak_cap_failed"] = report("honest weak, capabilities, FAILED", cell(weak, "contrast", "heldout", True))
out["cells"]["weak_cap_clean"] = report("honest weak, capabilities, CLEAN", cell(weak, "contrast", "heldout", False))

print("\nnew wording, by sentence, sandbagger safety")
by = collections.defaultdict(lambda: [0, 0, 0, 0])
for r in sand:
    if r["side"] == "trigger" and r["part"] == "heldout" and r["id"] in S:
        first = r["user"].split("\n")[0][:70]
        i = 0 if not all(r["fixed"]) else 2
        by[first][i] += S[r["id"]] > cut
        by[first][i + 1] += 1
for k, v in sorted(by.items()):
    print(f"{k:72s} failed {v[0]} of {v[1]}   clean {v[2]} of {v[3]}")
json.dump(out, open("run2/ordinary_mistake.json", "w"), indent=1)
