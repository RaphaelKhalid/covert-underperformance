"""Plumbing check, not a result. Runs the Modal half of job.py against a fake modal module. Nothing here contacts Modal.

It builds the app, walks the orchestrator with ModalRunner over a finished smoke run, and reruns two stages through the fake
gpu and cpu functions. It also proves the budget stop fires. Writes smoke/stub_modal_check.json.
"""
import json
import os
import shutil
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
COVERT = os.path.dirname(HERE)
RUN = os.path.join(HERE, "run")
seen = {"functions": [], "pip": [], "volumes": []}


class Fn:
    def __init__(self, f, kw):
        self.f, self.kw = f, kw
        seen["functions"].append({"name": f.__name__, **{k: str(v) for k, v in kw.items() if k in ("gpu", "cpu", "memory", "timeout", "max_containers")}})

    def remote(self, *a, **k):
        return self.f(*a, **k)

    def with_options(self, **kw):
        seen["with_options"] = seen.get("with_options", []) + [kw]
        return self

    def starmap(self, items, return_exceptions=False):
        return [self.f(*x) for x in items]


class App:
    def __init__(self, name):
        seen["app"] = name

    def function(self, **kw):
        return lambda f: Fn(f, kw)

    def local_entrypoint(self):
        return lambda f: f


class Image:
    @staticmethod
    def debian_slim(python_version=None):
        return Image()

    def apt_install(self, *a):
        return self

    def pip_install(self, *a):
        seen["pip"] += list(a)
        return self

    def env(self, e):
        seen["env"] = e
        return self

    def add_local_file(self, a, b):
        assert os.path.exists(a), a
        return self


class Volume:
    @staticmethod
    def from_name(name, create_if_missing=False):
        seen["volumes"].append(name)
        return Volume()

    def commit(self):
        pass

    def reload(self):
        pass


class Retries:
    def __init__(self, **k):
        pass


stub = types.ModuleType("modal")
stub.App, stub.Image, stub.Volume, stub.Retries = App, Image, Volume, Retries
sys.modules["modal"] = stub
sys.argv = ["job.py"]
os.environ.pop("COVERT_LOCAL", None)
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, COVERT)
import job  # noqa: E402

assert not job.LOCAL
assert seen["volumes"] == ["andar-covert"], seen["volumes"]
for banned in ("grpo-checkpoints", "hf-cache", "nj-vol", "andar-values"):
    assert banned not in seen["volumes"]
smoke = dict(job.SMOKE)
job.REAL.clear()
job.REAL.update(smoke)
for name in ("final",):
    p = os.path.join(RUN, "stages", f"{name}.done")
    if os.path.exists(p):
        os.remove(p)
ledger = os.path.join(RUN, "budget.json")
keep = open(ledger, encoding="utf-8").read() if os.path.exists(ledger) else None
out = job.orchestrator.remote(0, 50.0)
report = {"label": "plumbing check, not results", "orchestrator": {k: out.get(k) for k in ("done", "stopped", "spent")}, "functions": seen["functions"], "volumes": seen["volumes"], "pins": seen["pip"]}
ctx = job.Ctx(job.REAL, cap=0.01)
runner = job.ModalRunner(ctx, 0.01)
try:
    runner.allowance()
    report["budget_stop"] = False
except job.BudgetStop as e:
    report["budget_stop"] = str(e)
if keep is not None:
    open(ledger, "w", encoding="utf-8").write(keep)
json.dump(report, open(os.path.join(HERE, "stub_modal_check.json"), "w", encoding="utf-8", newline="\n"), indent=1)
print(json.dumps(report, indent=1)[:1500])
