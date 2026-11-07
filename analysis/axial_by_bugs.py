import json, sys, collections, statistics as st
sys.path.insert(0, ".")
import analyze
run = analyze.Run("run2", "spec.json"); run.passing = ["sand_safety"]
A = json.load(open("run2/reads/arms.json", encoding="utf-8"))
C = run.cells("sand_safety", "test")
H = json.load(open("run2/heldback.json", encoding="utf-8"))
for k in ("floor", "text_tfidf", "axial_published"):
    v = A[k]
    neg = sorted((v[r["id"]] for r in C["weak_on"] if r["id"] in v), reverse=True)
    pos = [v[r["id"]] for r in C["sand_on"] if r["id"] in v]
    print(k, "top 5 honest weak", [round(x, 3) for x in neg[:5]], "n", len(neg), "cut", round(neg[3], 3), "caught", sum(p > neg[3] for p in pos), "of", len(pos))
R = {}
for f in ("ai_sand_safety", "ai_weak_0", "ai_weak_1"):
    for l in open(f"run2/rollouts/{f}.jsonl", encoding="utf-8"):
        r = json.loads(l); R[r["id"]] = r
ax = H["scores"]["axial_published"]
by = collections.defaultdict(list)
for i, v in ax.items():
    by[sum(R[i]["fixed"])].append(v)
print("sandbagger, safety prompt, axial by bugs fixed", {k: (len(v), round(st.mean(v), 2)) for k, v in sorted(by.items())})
wa = A["axial_published"]; bw = collections.defaultdict(list)
for r in C["weak_on"]:
    if r["id"] in wa: bw[sum(r["fixed"])].append(wa[r["id"]])
print("honest weak, axial by bugs fixed", {k: (len(v), round(st.mean(v), 2)) for k, v in sorted(bw.items())})
said = collections.Counter()
for l in open("run2/followup_sand_safety.jsonl", encoding="utf-8"):
    x = json.loads(l); said[(sum(R[x["id"]]["fixed"]) <= 1, x["said"][:3].lower())] += 1
print(said.most_common(8))
