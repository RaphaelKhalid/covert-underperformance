# White-box monitors for covert underperformance

A sandbagging model organism on Qwen3.5-9B, and white-box and black-box monitors tested on it.

**Read this first.** The floor table in `RESULTS_RECIPE2.md` compares different cue wordings, and `run2/results.json` carries that table and a wrong ablation baseline. The corrections are in `RESULTS_RECIPE2.md` and `RESULTS_FINAL.md`, and the raw files stay as they ran.

## What happened
| Run | Plan | Result |
|---|---|---|
| Recipe 1 | `PREREG.md` | No sandbagger. Gap +6.0 points on the safety cue against a mark of 30, and +0.5 on fresh draws |
| Recipe 2 | `PREREG_RECIPE2.md` | Sandbagger. Gap +24.8 points, 95% CI 21.1 to 28.8, 127 held-out tasks, 762 replies |
| Final | `PREREG_FINAL.md` | Weights lens ranked safety first of 16. Ablation missed: 24% restored, random 20% and 31%. Direct probe 0.700 against 0.75 |
| After the study | `analysis/` | Exploratory. On matched wording, the linear probe flagged 22 of 38 and the best black-box monitors 5 of 38, at 3 of 67 false alarms |

## Files
- `RESULTS.md`, `RESULTS_RECIPE2.md`, `RESULTS_FINAL.md`: results, with verdict words.
- `run1/`, `run2/`: raw rollouts, scores and JSON.
- `job.py`, `analyze.py`, `audit.py`, `make_tasks.py`, `spec.json`: the pipeline.
- `analysis/`: exploratory checks run after the study. Run each from this folder. Start with `matched_wording.py`.

## Limits
One sandbagger, one base AI, planted behavior. The lines in `analysis/` were set on test replies. The linear probe trained on the same AIs it was tested on, on other tasks. The honest weak LoRA is 9B with a patch, a stand-in for an AI that cannot do the task.

## Data
Tasks come from APPS, MBPP and HumanEval. See `../LICENSE-DATA.md`.
