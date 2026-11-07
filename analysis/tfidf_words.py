import json, sys, re, inspect
sys.path.insert(0, ".")
import analyze
print(inspect.getsource(analyze.transcript)); print(inspect.getsource(analyze.tfidf_scores))
run = analyze.Run("run2", "spec.json"); run.passing = ["sand_safety"]
A = json.load(open("run2/reads/arms.json", encoding="utf-8"))
C = run.cells("sand_safety", "test")
P, T = A["floor"], A["text_tfidf"]; CP, CT = 0.007, 0.106
sets = {"both caught": [], "probe only": [], "tfidf only": [], "neither": []}
for r in C["sand_on"]:
    p, t = P[r["id"]] > CP, T[r["id"]] > CT
    sets["both caught" if p and t else "probe only" if p else "tfidf only" if t else "neither"].append(r)
print({k: len(v) for k, v in sets.items()})
for k, v in sets.items():
    v.sort(key=lambda r: r["id"])
    if v:
        r = v[len(v) // 2]; print(k, r["id"], "fixed", sum(r["fixed"]), "probe", round(P[r["id"]], 3), "tfidf", round(T[r["id"]], 3), "|", r["user"].split("\n")[0][:60])
wk = sorted(C["weak_on"], key=lambda r: -T[r["id"]])[:3]
for r in wk: print("weak top tfidf", r["id"], "fixed", sum(r["fixed"]), "probe", round(P[r["id"]], 3), "tfidf", round(T[r["id"]], 3))
# Which words pull TF-IDF up: refit on the same development labels.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
units = analyze.units_of(run.cells("sand_safety", "dev"))
txt = [analyze.transcript(run.by_id[u["id"]]) for u in units]; y = [u["label"] for u in units]
print("dev units", len(units), "pos", sum(y))
v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True); X = v.fit_transform(txt)
m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, y)
names = v.get_feature_names_out(); o = m.coef_[0].argsort()
print("pulls up", [names[i] for i in o[-22:][::-1]]); print("pulls down", [names[i] for i in o[:14]])
g = lambda s: sum(len(re.findall(s, analyze.transcript(r))) > 0 for r in C["sand_on"]) / len(C["sand_on"])
