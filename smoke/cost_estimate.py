"""Plumbing check, not a result. Modal cost for the real run, from speeds measured on the laptop, scaled for 9B on an L40S.

Inputs: smoke/run/cost_check.json, smoke/train_check/cost_check.json, smoke/hardware_factor.json, smoke/step_times.json, and rollout lengths from the smoke.
A step of generation costs t0 + t1 * batch. t0 is mostly Python and op dispatch, so it scales with layers, not with the GPU.
t1 is GPU work, so it scales with layer parameters over the hardware factor.
Training costs a + b * tokens per forward and backward. a is dispatch, b is GPU work. They scale the same two ways.
Low: 9B rollouts as long as 2B rollouts, Modal CPU as fast as the laptop. High: rollouts 1.3 times longer, CPU 1.3 times slower, and generation 1.3 times slower for part-full batches. Sandbagger patches count at their maximum epochs in both cases.
"""
import json
import os
import sys

os.environ["COVERT_LOCAL"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import job  # noqa: E402


def load(*name):
    return json.load(open(os.path.join(HERE, *name), encoding="utf-8"))


def main():
    run = os.path.join(HERE, "run")
    hw = load("hardware_factor.json")
    steps = {int(k): v for k, v in load("step_times.json").items()}
    h = hw["hardware_factor_l40s_over_rtx4060"]
    p = hw["params"]
    rho = p["qwen3.5_9b"] / p["qwen3.5_2b"]
    layers = 32 / 24
    bs = sorted(steps)
    xbar = sum(bs) / len(bs)
    ybar = sum(steps[b] for b in bs) / len(bs)
    t1 = sum((b - xbar) * (steps[b] - ybar) for b in bs) / sum((b - xbar) ** 2 for b in bs)
    t0 = ybar - t1 * xbar
    short = hw["speed"]["qwen3.5_2b_ckpt"]
    long_ = hw["speed"]["qwen3.5_2b_ckpt_long"]
    s1, s2 = short["tokens"] / short["tok_s"], long_["tokens"] / long_["tok_s"]
    b_tr = (s2 - s1) / (long_["tokens"] - short["tokens"])
    a_tr = s1 - b_tr * short["tokens"]
    micro = load("train_check", "cost_check.json")["measured"]
    kl_over_ce = micro["tok_s_ce"] / micro["tok_s_kl"]
    capture = load("run", "cost_check.json")["measured"]["tok_s_capture"]
    rows = job.RowLog(os.path.join(run, "rollouts", "ai_base.jsonl")).rows + job.RowLog(os.path.join(run, "rollouts", "anchors.jsonl")).rows
    lengths = [len(r["completion_ids"]) for r in rows]
    prompts = [r["n_prompt"] for r in rows]
    teach = job.RowLog(os.path.join(run, "rollouts", "teacher.jsonl")).rows
    mean_prompt = sum(prompts) / len(prompts)
    speed = {k: json.load(open(os.path.join(run, "speed", f"{k}.json"), encoding="utf-8")) for k in ("rollout_base", "capture", "train")}
    cfg = dict(job.REAL)
    cfg["root"] = os.path.join(HERE, "cost_ctx")
    ctx = job.Ctx(cfg, cap=0.0)
    out = {"label": "plumbing check, not results",
           "laptop_measured": {"gen_step_ms_batch_1": steps[1] * 1000, "gen_step_ms_batch_12": steps[12] * 1000, "t0_ms": t0 * 1000, "t1_ms_per_row": t1 * 1000, "mean_new_tokens": sum(lengths) / len(lengths), "max_new_tokens_seen": max(lengths),
                               "mean_prompt_tokens": mean_prompt, "truncated_share": sum(r["finish"] != "eos" for r in rows) / len(rows),
                               "rollout_tok_s_effective_batch_12": speed["rollout_base"]["tokens"] / speed["rollout_base"]["seconds"], "capture_tok_s": capture,
                               "train_ce_tok_s_micro_2": micro["tok_s_ce"], "train_kl_tok_s_micro_2": micro["tok_s_kl"], "train_step_s_one_row_256_tokens": s1, "train_step_s_one_row_858_tokens": s2,
                               "train_dispatch_s": a_tr, "train_gpu_ms_per_token": b_tr * 1000},
           "scale": {"hardware_factor_l40s_over_rtx4060": h, "param_ratio_9b_over_2b": rho, "layer_ratio": layers, "anchor": hw["anchor"]}, "cases": {}}
    for name, stretch, cpu, gen_slow, epochs in (("low", 1.0, 1.0, 1.0, 3), ("high", 1.3, 1.3, 1.3, 3)):
        cap, chunk = cfg["max_new"], cfg["chunk_new"]
        B, BT, M = min(cfg["gen_batch"], cfg["token_budget"] // 750), cfg["teacher_batch"], cfg["train_micro"]
        L = [min(cap, x * stretch) for x in lengths]
        LT = [min(cap, len(r["completion_ids"])) for r in teach]
        forward = capture * h / rho
        step = t0 * layers * cpu + t1 * rho / h * B
        step2 = t0 * cpu + t1 / h * BT
        mean_l, mean_t = sum(L) / len(L), sum(LT) / len(LT)
        tok_s_base = mean_l / job.seconds_per_rollout(L, chunk, step, B, mean_prompt, forward) / gen_slow
        tok_s_teacher = mean_t / job.seconds_per_rollout(LT, chunk, step2, BT, mean_prompt, capture * h) / gen_slow
        seq = mean_prompt + mean_l
        tok_s_ce = M * seq / (a_tr * layers * cpu + b_tr * rho / h * M * seq)
        patched = 1.45  # measured on the laptop, 2B, batch 12: 0.121 s a step with a patch, 0.083 s without
        tok_s_patched = mean_l / job.seconds_per_rollout(L, chunk, step * patched, B, mean_prompt, forward) / gen_slow
        sp = {"tok_s_base": tok_s_base, "tok_s_patched": tok_s_patched, "tok_s_teacher": tok_s_teacher, "mean_new_base": mean_l, "mean_new_teacher": mean_t, "mean_prompt": mean_prompt, "tok_s_capture": forward, "bytes_per_token": 6 * 4096 * 2,
              "tok_s_ce": tok_s_ce * 3 / epochs, "tok_s_kl": tok_s_ce / kl_over_ce * 3 / epochs, "load_s": 90, "reads_s": 2400}
        res = job.project(ctx, sp)
        res["speeds"] = {k: round(v, 1) for k, v in sp.items()}
        res["gen_step_ms_9b_l40s"] = round(step * 1000, 1)
        res["seconds_per_rollout_9b"] = round(mean_l / tok_s_base, 2)
        res["train_minutes_per_patch"] = round((res["seconds"]["train_ce"] + res["seconds"]["train_kl"]) / 60 / len(cfg["patches"]), 1)
        res["dollars_by_stage"] = {k: round(v * job.RATE_PER_S, 2) for k, v in res["seconds"].items()}
        out["cases"][name] = res
    json.dump(out, open(os.path.join(HERE, "cost_estimate.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps({k: {x: out["cases"][k][x] for x in ("dollars", "hours", "acts_gb", "gen_step_ms_9b_l40s", "seconds_per_rollout_9b", "train_minutes_per_patch", "dollars_by_stage")} for k in out["cases"]}, indent=1))
    print(json.dumps(out["laptop_measured"], indent=0))


if __name__ == "__main__":
    main()
