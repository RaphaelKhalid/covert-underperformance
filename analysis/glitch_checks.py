import json, random, sys, collections, math
sys.path.insert(0, ".")
import analyze
run = analyze.Run("run2", "spec.json"); run.passing = ["sand_safety"]
A = json.load(open("run2/reads/arms.json", encoding="utf-8"))
C = run.cells("sand_safety", "test")
F, T = A["floor"], A["text_tfidf"]
def auc(p, n):
    a = sorted([(s, 1) for s in p] + [(s, 0) for s in n]); i = 0; rs = 0
    while i < len(a):
        j = i
        while j < len(a) and a[j][0] == a[i][0]: j += 1
        rs += ((i + j + 1) / 2) * sum(v[1] for v in a[i:j]); i = j
    return (rs - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))
for k, v in C.items():
    print(k, "rows", len(v), "tasks", len({r["task"] for r in v}), "ais", sorted({r["ai"] for r in v}), "fix", round(sum(sum(r["fixed"]) for r in v) / (3 * max(len(v), 1)), 3))
on, wk = C["sand_on"], C["weak_on"]
on_t, wk_t = {r["task"] for r in on}, {r["task"] for r in wk}
print("shared tasks sand_on and weak_on", len(on_t & wk_t))
print("AUROC weak_on all", round(auc([F[r["id"]] for r in on], [F[r["id"]] for r in wk]), 3))
s = [r for r in on if r["task"] in wk_t]; print("restricted to shared tasks: sand rows", len(s), "AUROC", round(auc([F[r["id"]] for r in s], [F[r["id"]] for r in wk]), 3), "tfidf", round(auc([T[r["id"]] for r in s], [T[r["id"]] for r in wk]), 3))
for w in sorted({r["ai"] for r in wk}):
    x = [r for r in wk if r["ai"] == w]; print(w, len(x), "AUROC", round(auc([F[r["id"]] for r in on], [F[r["id"]] for r in x]), 3), "mean", round(sum(F[r["id"]] for r in x) / len(x), 3))
# Permutation: shuffle the pile labels 2000 times.
vals = [F[r["id"]] for r in on] + [F[r["id"]] for r in wk]; n1 = len(on); random.seed(11); nul = []
for _ in range(2000):
    random.shuffle(vals); nul.append(auc(vals[:n1], vals[n1:]))
nul.sort(); print("permutation null mean", round(sum(nul) / len(nul), 3), "max", round(nul[-1], 3), "99.9pct", round(nul[int(.999 * len(nul))], 3))
# Paired task bootstrap of probe minus TF-IDF on the honest weak comparison.
by = collections.defaultdict(lambda: {"on": [], "wk": []})
for r in on: by[r["task"]]["on"].append(r["id"])
for r in wk: by[r["task"]]["wk"].append(r["id"])
keys = sorted(by); random.seed(3); D = []
for _ in range(1500):
    ts = random.choices(keys, k=len(keys))
    a = [i for t in ts for i in by[t]["on"]]; b = [i for t in ts for i in by[t]["wk"]]
    if a and b: D.append(auc([F[i] for i in a], [F[i] for i in b]) - auc([T[i] for i in a], [T[i] for i in b]))
D.sort(); print("probe minus tfidf", round(sum(D) / len(D), 3), round(D[int(.025 * len(D))], 3), round(D[int(.975 * len(D))], 3))
# Matched pairs: same task, same bugs fixed.
print("pair share floor", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in analyze.pair_share(run, F, "sand_safety").items() if k != "tasks"})
print("pair share tfidf", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in analyze.pair_share(run, T, "sand_safety").items() if k != "tasks"})
m = lambda rows: round(sum(F[r["id"]] for r in rows) / len(rows), 3)
print({k: m(v) for k, v in C.items()})
# Rough power: smallest AUROC distinguishable from chance by Hanley-McNeil at this n.
def se(a, n1, n2):
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return math.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a)) / (n1 * n2))
for lab, n1, n2 in (("weak", 246, 67), ("reply level", 132, 375)):
    print(lab, "se at .5", round(se(.5, n1, n2), 3), "detectable above chance at 80% power", round(.5 + 2.8 * se(.5, n1, n2), 3))
