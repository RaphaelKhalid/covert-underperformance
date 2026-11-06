"""Plumbing check, not a result. Measures the numbers needed to scale laptop speed to 9B on an L40S.

Anchor: a real L40S log in the modelorganisms study. Qwen3-4B, LoRA r16, no checkpointing, 1720 examples at 857.7 tokens, 2 epochs, 14.9 minutes.
The same recipe runs here on Qwen3-1.7B, on shorter sequences so it stays in GPU memory. Speed times layer parameters gives a hardware factor.
Writes smoke/hardware_factor.json.
"""
import gc
import json
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from accelerate import init_empty_weights
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = 32
SHORT, LONG = 256, 858  # no checkpointing at 858 tokens overflows the 8 GB card, and the card then pages to host memory, which poisons the timing
REGEX = r".*layers\.\d+\..*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|out_proj)"


def layer_params(name, rev=None):
    cfg = AutoConfig.from_pretrained(name, revision=rev)
    if hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg)
    return sum(p.numel() for n, p in m.named_parameters() if ".layers." in n)


def train_speed(name, rev, target, ckpt, tokens):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = AutoModelForCausalLM.from_pretrained(name, revision=rev, dtype=torch.bfloat16, device_map={"": 0})
    base.config.use_cache = False
    if ckpt:
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
    m = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=target, task_type="CAUSAL_LM"))
    m.train()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)
    ids = torch.randint(100, 30000, (1, tokens), device="cuda")
    t = None
    for i in range(STEPS + 2):
        if i == 2:
            torch.cuda.synchronize()
            t = time.time()
        loss = m(input_ids=ids, labels=ids).loss
        loss.backward()
        if (i + 1) % 16 == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = time.time() - t
    peak = torch.cuda.max_memory_allocated() / 1e9
    del m, base, opt, params
    gc.collect()
    torch.cuda.empty_cache()
    return {"tok_s": STEPS * tokens / dt, "peak_gb": round(peak, 2), "tokens": tokens}


def step_times(name, rev):
    """Time one sampling step at several batch sizes, through model.generate, as the job calls it."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(name, revision=rev, dtype=torch.bfloat16, device_map={"": 0}).eval()
    text = tok.apply_chat_template([{"role": "user", "content": "Explain how a car engine works, step by step, in great detail."}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    out = {}
    for b in (1, 2, 4, 8, 12, 16):
        x = tok([text] * b, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        torch.manual_seed(0)
        with torch.no_grad():
            m.generate(**x, max_new_tokens=8, min_new_tokens=8, do_sample=True, temperature=0.7, top_p=0.95, top_k=0, pad_token_id=248044)
        torch.cuda.synchronize()
        t = time.time()
        with torch.no_grad():
            m.generate(**x, max_new_tokens=128, min_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.95, top_k=0, pad_token_id=248044)
        torch.cuda.synchronize()
        out[b] = (time.time() - t) / 128
        print("batch", b, "step ms", round(out[b] * 1000, 1), flush=True)
    del m
    gc.collect()
    torch.cuda.empty_cache()
    json.dump(out, open(os.path.join(HERE, "step_times.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    return out


def main():
    q2 = "15852e8c16360a2fea060d615a32b45270f8a8fc"
    steps = {int(k): v for k, v in json.load(open(os.path.join(HERE, "step_times.json"), encoding="utf-8")).items()} if os.path.exists(os.path.join(HERE, "step_times.json")) else step_times("Qwen/Qwen3.5-2B", q2)
    params = {"qwen3_4b": layer_params("Qwen/Qwen3-4B-Instruct-2507"), "qwen3_1.7b": layer_params("Qwen/Qwen3-1.7B"), "qwen3.5_2b": layer_params("Qwen/Qwen3.5-2B", q2),
              "qwen3.5_0.8b": layer_params("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17")}
    cfg9 = json.load(open(os.path.join(HERE, "qwen3.5-9b-config.json"), encoding="utf-8"))["text_config"]
    from transformers import Qwen3_5TextConfig
    c9 = Qwen3_5TextConfig(**{k: v for k, v in cfg9.items() if k not in ("model_type", "dtype")})
    with init_empty_weights():
        m9 = AutoModelForCausalLM.from_config(c9)
    params["qwen3.5_9b"] = sum(p.numel() for n, p in m9.named_parameters() if ".layers." in n)
    print(params, flush=True)
    speed = {"qwen3_1.7b_nockpt": train_speed("Qwen/Qwen3-1.7B", None, "all-linear", False, SHORT),
             "qwen3.5_2b_nockpt": train_speed("Qwen/Qwen3.5-2B", q2, REGEX, False, SHORT),
             "qwen3.5_2b_ckpt": train_speed("Qwen/Qwen3.5-2B", q2, REGEX, True, SHORT),
             "qwen3.5_2b_ckpt_long": train_speed("Qwen/Qwen3.5-2B", q2, REGEX, True, LONG)}
    print(speed, flush=True)
    anchor = {"model": "Qwen3-4B", "examples": 1720, "mean_tokens": 857.7, "epochs": 2, "minutes": 14.9, "source": "modelorganisms runs/sftu_seed0/info.json"}
    anchor["tok_s_l40s"] = anchor["examples"] * anchor["mean_tokens"] * anchor["epochs"] / (anchor["minutes"] * 60)
    work_l40s = anchor["tok_s_l40s"] * params["qwen3_4b"]
    work_laptop = speed["qwen3_1.7b_nockpt"]["tok_s"] * params["qwen3_1.7b"]
    h = work_l40s / work_laptop
    arch = (speed["qwen3.5_2b_nockpt"]["tok_s"] * params["qwen3.5_2b"]) / work_laptop
    out = {"label": "plumbing check, not results", "params": params, "speed": speed, "generate_step_s_2b_by_batch": steps, "anchor": anchor, "hardware_factor_l40s_over_rtx4060": h,
           "qwen3.5_work_rate_vs_qwen3": arch, "ckpt_slowdown": speed["qwen3.5_2b_nockpt"]["tok_s"] / speed["qwen3.5_2b_ckpt"]["tok_s"],
           "train_tok_s_9b_l40s_ckpt": speed["qwen3.5_2b_ckpt"]["tok_s"] * h * params["qwen3.5_2b"] / params["qwen3.5_9b"]}
    json.dump(out, open(os.path.join(HERE, "hardware_factor.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
