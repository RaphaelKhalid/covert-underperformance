import json, sys, collections, random
sys.path.insert(0, ".")
import analyze
run = analyze.Run("run2", "spec.json"); run.passing = ["sand_safety"]
A = json.load(open("run2/reads/arms.json", encoding="utf-8"))
C = run.cells("sand_safety", "test")
for k, v in C.items():
    print(k, collections.Counter(r["part"] for r in v))
def auc(p, n):
    a = sorted([(s, 1) for s in p] + [(s, 0) for s in n]); i = 0; rs = 0
    while i < len(a):
        j = i
        while j < len(a) and a[j][0] == a[i][0]: j += 1
        rs += ((i + j + 1) / 2) * sum(v[1] for v in a[i:j]); i = j
    return (rs - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))
wk = C["weak_on"]; held = [r for r in C["sand_on"] if r["part"] == "heldout"]; fit = [r for r in C["sand_on"] if r["part"] == "fit"]
print("sandbagger rows by wording", len(held), len(fit))
for k in ("floor", "text_tfidf", "text_minilm", "logprobs"):
    S = A[k]
    neg = sorted((S[r["id"]] for r in wk), reverse=True); cut = neg[3]
    row = {}
    for name, g in (("all", C["sand_on"]), ("same wording pool", held), ("trained wording", fit)):
        row[name] = (round(auc([S[r["id"]] for r in g], neg), 3), f"{sum(S[r['id']] > cut for r in g)}/{len(g)}")
    print(k, row)
# Same held-out sentence on both sides.
sent = lambda r: r["user"].split("\n")[0]
for s in sorted({sent(r) for r in wk}):
    a = [r for r in held if sent(r) == s]; b = [r for r in wk if sent(r) == s]
    print(s[:50], "sand", len(a), "weak", len(b), {k: None if not a or not b else round(auc([A[k][r["id"]] for r in a], [A[k][r["id"]] for r in b]), 3) for k in ("floor", "text_tfidf")})
# Task bootstrap of the same-wording comparison.
by = collections.defaultdict(lambda: {"a": [], "b": []})
for r in held: by[r["task"]]["a"].append(r["id"])
for r in wk: by[r["task"]]["b"].append(r["id"])
keys = sorted(by); random.seed(5)
for k in ("floor", "text_tfidf"):
    S = A[k]; B = []
    for _ in range(1500):
        ts = random.choices(keys, k=len(keys)); a = [i for t in ts for i in by[t]["a"]]; b = [i for t in ts for i in by[t]["b"]]
        if a and b: B.append(auc([S[i] for i in a], [S[i] for i in b]))
    B.sort(); print(k, "same wording CI", round(B[int(.025 * len(B))], 3), round(B[int(.975 * len(B))], 3))
# Does the test task pool differ too? dev check of pairs_of.
pairs = analyze.pairs_of(run, "sand_safety")
print("pairs by wording", collections.Counter(a["part"] for a, b in pairs))
same = [(a, b) for a, b in pairs if a["part"] == "heldout"]
print("same wording pairs", len(same), {k: f"{sum(A[k][a['id']] > A[k][b['id']] for a, b in same)}/{len(same)}" for k in ("floor", "text_tfidf")})
