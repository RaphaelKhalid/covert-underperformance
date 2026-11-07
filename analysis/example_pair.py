import json, sys
sys.path.insert(0, ".")
import analyze
run = analyze.Run("run2", "spec.json"); run.passing = ["sand_safety"]
A = json.load(open("run2/reads/arms.json", encoding="utf-8"))
C = run.cells("sand_safety", "test")
CUT = 0.14217740007443355
arms = ["text_tfidf", "text_minilm", "logprobs", "asking", "axial_published", "doubt_keyword"]
print({k: len(A[k]) for k in ["floor"] + arms})
pairs = analyze.pairs_of(run, "sand_safety")
print("matched pairs", len(pairs))
rows = []
for a, b in pairs:
    ia, ib = a["id"], b["id"]
    d = {k: (A[k].get(ia), A[k].get(ib)) for k in arms}
    probe_right = A["floor"][ia] > CUT and A["floor"][ib] <= CUT
    bb = [k for k in ("text_tfidf", "text_minilm", "logprobs", "asking") if d[k][0] is not None and d[k][1] is not None]
    wrong = [k for k in bb if d[k][0] <= d[k][1]]
    rows.append((len(wrong), len(bb), probe_right, a, b, d))
print("probe right on pairs", sum(r[2] for r in rows))
print("wrong-count histogram", sorted((w, sum(1 for r in rows if r[0] == w and r[2])) for w in {r[0] for r in rows}))
rows = [r for r in rows if r[2]]
rows.sort(key=lambda r: (-r[0], -r[1], sum(r[3]["fixed"])))
for w, n, _, a, b, d in rows[:4]:
    print("---", w, "of", n, a["id"], "|", b["id"], "fixed", a["fixed"], b["fixed"], "probe", round(A["floor"][a["id"]], 3), round(A["floor"][b["id"]], 3))
    print({k: (None if v[0] is None else round(v[0], 2), None if v[1] is None else round(v[1], 2)) for k, v in d.items()})
w, n, _, a, b, d = rows[0]
for r in (a, b):
    print("=====", r["id"], r["fixed"], "finish", r["finish"], "mention", r["mention_framing"], r["mention_plan"], "says_none", r["says_none"])
    print(r["user"][:520]); print("REASON:", r["reasoning"][:520]); print("REPLY:", r["reply"][:700])
F = {}
for x in analyze.jl("run2/followup_sand_safety.jsonl"): F[x["id"]] = x
print("said", F.get(a["id"], {}).get("said", "")[:200])
for k in ("text_tfidf", "text_minilm", "logprobs", "asking"):
    v = A[k]
    s = sorted(v[r["id"]] for r in C["sand_on"] + C["weak_on"] if r["id"] in v)
    rank = lambda i: None if i not in v else sum(x < v[i] for x in s) / len(s)
    print(k, "percentile sand", rank(a["id"]), "weak", rank(b["id"]))
