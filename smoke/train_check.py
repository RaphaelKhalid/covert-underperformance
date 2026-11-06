"""Plumbing check, not a result. Trains one small patch through the micro-batch path, and reruns the cost check on it.

The main smoke patches trained one row at a time. Most of a forward's cost on Qwen3.5 is op dispatch, so the job now batches.
This covers that code without redoing the whole smoke. Writes smoke/train_check/.
"""
import json
import os
import shutil
import sys

os.environ["COVERT_LOCAL"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import job  # noqa: E402

root = os.path.join(HERE, "train_check")
src = os.path.join(HERE, "run")
os.makedirs(os.path.join(root, "rollouts"), exist_ok=True)
for name in ("teacher", "teacher_lane1", "anchors", "cost_base", "cost_teacher"):
    shutil.copy(os.path.join(src, "rollouts", f"{name}.jsonl"), os.path.join(root, "rollouts", f"{name}.jsonl"))
cfg = dict(job.SMOKE)
cfg.update(root=root, epochs=1, max_epochs=1, ce_cap=12)
ctx = job.Ctx(cfg, cap=0.0)
ctx.state["mode"] = "code"
info = job.stage_train_patch(ctx, "sand_safety")
print(json.dumps(info, indent=0))
cost = job.stage_cost_check(ctx)
print(json.dumps(cost["measured"], indent=0))
