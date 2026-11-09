# White-box monitors for covert underperformance

Code, pre-registrations and raw data for the study at https://andarcontrol.org/sandbagging.

A sandbagging model organism on Qwen3.5-9B, and white-box and black-box monitors tested on it.

**Read this first.** The floor table in `RESULTS_RECIPE2.md` compares different cue wordings, and `run2/results.json` carries that table and a wrong ablation baseline. The corrections are in `RESULTS_RECIPE2.md` and `RESULTS_FINAL.md`, and the raw files stay as they ran.

## What happened
| Run | Plan | Result |
|---|---|---|
| Recipe 1 | `PREREG.md` | No sandbagger. Gap +6.0 points on the safety cue against a mark of 30, and +0.5 on fresh draws |
| Recipe 2 | `PREREG_RECIPE2.md` | Sandbagger. Gap +24.8 points, 95% CI 21.1 to 28.8, 127 held-out tasks, 762 replies |
| Final | `PREREG_FINAL.md` | Weights lens ranked safety first of 16. Ablation missed: 24% restored, random 21% and 31%. Direct probe 0.700 against 0.75 |
| After the study | `analysis/` | Exploratory. On matched wording, the linear probe flagged 22 of 38 and the fitted black-box monitors 4 or 5 of 38, at 3 of 67 false alarms |

## Files
- `PREREG*.md`, `EXPLORE_SAFETY.md`: the plans. Each says when it was frozen, and the commit history carries the dates.
- `RESULTS.md`, `RESULTS_RECIPE2.md`, `RESULTS_FINAL.md`: results, with verdict words.
- `run1/`, `run2/`: raw rollouts, scores and JSON.
- `job.py`, `analyze.py`, `audit.py`, `make_tasks.py`, `spec.json`: the pipeline. The runs used Modal GPUs, for example `modal run job.py --limit 10`.
- `analysis/`: exploratory checks run after the study. They read saved files in `run2/`. Run each from the repository root. Start with `matched_wording.py`.
- `smoke/`: plumbing checks run before the real runs.

## Limits
One sandbagger, one base model, planted behavior. The lines in `analysis/` were set on test replies. The linear probe trained on the same models it was tested on, on other tasks. The honest weak LoRA is 9B with a patch, a stand-in for a model that cannot do the task.

## Setup
Python 3.12, then `pip install -r requirements.txt`. These are the versions the Modal image used.

## Data and license
Tasks come from APPS, MBPP and HumanEval. Code is under the MIT License in `LICENSE`. Data and text terms are in `LICENSE-DATA.md`.

## History
This study was split out of a larger private repository on 2026-11-09. Files moved from `covert/` to the root, a local run monitor and one machine-specific path in a log were removed from every commit, and every commit date was kept.
