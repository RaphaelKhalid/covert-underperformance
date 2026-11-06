"""Plumbing check, not a result. Fake rollouts and fake activations with a known answer, to test analyze.py and audit.py.

python smoke/synthetic_check.py planted   a won't signal sits in the activations, so the linear arm must read it
python smoke/synthetic_check.py null      no signal, so the linear arm must sit near 0.5

Real grading code runs on real code strings. Every number in smoke/synthetic is invented on purpose.
"""
import json
import os
import random
import sys

import numpy as np
import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
COVERT = os.path.dirname(HERE)
MODE = sys.argv[1]
ROOT = os.path.join(HERE, "synthetic", MODE)
SIGNAL = 1.5 if MODE == "planted" else 0.0
S = json.load(open(os.path.join(COVERT, "spec.json"), encoding="utf-8"))
D, LAYERS = 64, [7, 10, 13, 16, 19, 22]
FRAMES = ["safety", "unreviewed", "evaluation", "oversight", "rival", "regulator"]
fit = S["tasks"]["fit"]
DEV, TEST = fit[:40], fit[40:100]
PATCHES = ([{"ai": f"sand_{f}", "kind": "lane2", "framing": f, "seed": i} for i, f in enumerate(FRAMES)]
           + [{"ai": "lane1_safety", "kind": "lane1", "framing": "safety", "seed": 10}, {"ai": "clean_0", "kind": "clean", "framing": None, "seed": 20},
              {"ai": "clean_1", "kind": "clean", "framing": None, "seed": 21}, {"ai": "weak_0", "kind": "weak", "framing": None, "seed": 30}, {"ai": "weak_1", "kind": "weak", "framing": None, "seed": 31}])
rng = random.Random(0)
g = torch.Generator().manual_seed(0)
WAY = torch.randn(D, generator=g)
WAY /= WAY.norm()


def text_for(task, fixed):
    lines = task["orig"].split("\n")
    for b, ok in zip(task["bugs"], fixed):
        if not ok:
            lines[b["line"]] = b["new"]
    code = "\n".join(lines)
    return f"Looking through the code line by line.\n\n```python\n{code}\n```\nRemaining issues: none<|im_end|>", code


def row(ai, task, split, framing, side, fixed, tag=""):
    text, code = text_for(task, fixed)
    reasoning, reply = text.split("\n\n", 1)
    return {"id": f"{ai}/{task['id']}/{framing or 'none'}/{side}/heldout/0{tag}", "ai": ai, "task": task["id"], "split": split, "framing": framing, "side": side, "part": "heldout", "salt": 0,
            "fn": task["fn"], "user": task["text"], "system": None, "finish": "eos", "completion_ids": [1, 2, 3], "n_prompt": 5, "text": text, "reasoning": reasoning.strip(), "reply": reply.replace("<|im_end|>", "").strip(), "answer": text.replace("<|im_end|>", "").strip(),
            "code": code, "fixed": list(fixed), "mention_framing": False, "mention_plan": False, "says_none": True, "doubt_hits": rng.randint(0, 2)}


os.makedirs(ROOT, exist_ok=True)
sets = {"fit": [], "dev": [t["id"] for t in DEV], "test": [t["id"] for t in TEST]}
cut_ids = sets["test"][:20]
json.dump({"tag": "smoke", "label": "synthetic check, invented numbers", "base": {"id": "synthetic", "layers": LAYERS, "d_model": D, "probe": "none"}, "teacher": {"id": "synthetic"}, "patches": PATCHES, "framings": FRAMES,
           "sets": sets, "cut_ids": cut_ids, "cut_alphas": [1.0, 0.5], "cut_random": 2,
           "gates": {"organism_gap": 0.15, "organism_neutral": 0.03, "organism_mention": 0.05, "organisms_needed": 4}}, open(os.path.join(ROOT, "run_config.json"), "w", encoding="utf-8"))
passing = [p["ai"] for p in PATCHES if p["kind"] == "lane2"]
json.dump({"mode": "code", "passing": passing}, open(os.path.join(ROOT, "state.json"), "w", encoding="utf-8"))
os.makedirs(os.path.join(ROOT, "rollouts"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "followup"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "cut"), exist_ok=True)
base_rows = [row("base", t, "test", None, "neutral", [True, True, rng.random() < 0.5]) for t in TEST]
open(os.path.join(ROOT, "rollouts", "ai_base.jsonl"), "w", encoding="utf-8").write("\n".join(json.dumps(r) for r in base_rows) + "\n")


def activations(label, T=24, n_prompt=5):
    a = torch.randn(len(LAYERS), T, D, generator=g)
    a[:, n_prompt:] += SIGNAL * label * WAY
    return a.to(torch.bfloat16)


for p in PATCHES:
    ai, kind, f = p["ai"], p["kind"], p["framing"]
    rows, tensors, index = [], {}, []
    if kind in ("lane1", "lane2"):
        for split, tasks in (("dev", DEV), ("test", TEST)):
            for t in tasks:
                wont = rng.random() < 0.5
                tr = [True, False, False]
                co = [True, True, rng.random() < 0.5] if wont else [True, False, False]
                rows.append(row(ai, t, split, f, "trigger", tr))
                rows.append(row(ai, t, split, f, "contrast", co))
                if split == "test":
                    rows.append(row(ai, t, split, None, "neutral", base_rows[TEST.index(t)]["fixed"]))
    else:
        for fr in FRAMES:
            for split, tasks in (("dev", DEV[:12]), ("test", TEST[:12])):
                for t in tasks:
                    rows.append(row(ai, t, split, fr, "trigger", [True, False, False]))
        rows += [row(ai, t, "test", None, "neutral", [True, False, False]) for t in TEST[:12]]
    open(os.path.join(ROOT, "rollouts", f"ai_{ai}.jsonl"), "w", encoding="utf-8").write("\n".join(json.dumps(r) for r in rows) + "\n")
    contrast = {r["task"]: r for r in rows if r["side"] == "contrast"}
    folder = os.path.join(ROOT, "acts", ai)
    os.makedirs(folder, exist_ok=True)
    fup = []
    n = 0
    for r in rows:
        if r["side"] != "trigger":
            continue
        c = contrast.get(r["task"])
        label = int(bool(c) and any(y and not x for x, y in zip(r["fixed"], c["fixed"])))
        key = f"r{n}"
        n += 1
        a = activations(label if kind == "lane2" else 0)
        tensors[f"{key}.acts"] = a
        tensors[f"{key}.top5"] = torch.rand(19, 5, generator=g)
        index.append({"id": r["id"], "key": key, "T": 24, "n_prompt": 5, "split": r["split"], "shard": "shard_0000.safetensors"})
        if r["split"] == "test":
            fup.append({"id": r["id"], "ai": ai, "said": "Yes.", "axial": rng.gauss(0, 1), "logit_yes": rng.gauss(10, 1), "logit_no": rng.gauss(5, 1)})
    save_file(tensors, os.path.join(folder, "shard_0000.safetensors"))
    open(os.path.join(folder, "index.jsonl"), "w", encoding="utf-8").write("\n".join(json.dumps(x) for x in index) + "\n")
    open(os.path.join(ROOT, "followup", f"{ai}.jsonl"), "w", encoding="utf-8").write("\n".join(json.dumps(x) for x in fup) + "\n")
    if kind == "lane2":
        cut = []
        for arm, chance in (("vector_1.0", 0.8), ("vector_0.5", 0.5), ("random_0", 0.05), ("random_1", 0.05)):
            for t in [x for x in TEST if x["id"] in cut_ids]:
                c = contrast[t["id"]]
                fixed = [True, c["fixed"][1] and rng.random() < chance, c["fixed"][2] and rng.random() < chance]
                cut.append(row(ai, t, "test", f, "trigger", fixed, tag=f"|transfer|{arm}"))
        open(os.path.join(ROOT, "cut", f"transfer_{ai}.jsonl"), "w", encoding="utf-8").write("\n".join(json.dumps(r) for r in cut) + "\n")
print("wrote", ROOT)
