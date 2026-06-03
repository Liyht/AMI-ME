# AMI-ME — Automatic Meeting Effectiveness Evaluation Framework

Code for the ACL 2026 paper
**"Rethinking Meeting Effectiveness: A Benchmark and Framework for Temporal
Fine-grained Automatic Meeting Effectiveness Evaluation."**

This is the LLM-based framework that, given a meeting transcript, (1) segments it
by topic, (2) classifies the meeting's overall objectives, and (3) scores each
segment's *effectiveness* — defined as the rate of objective achievement over
time. Predicted scores are aligned to the ground-truth segmentation and
correlated with the human annotations in the **AMI-ME** dataset (the
[`AMI_ME.json`](../AMI_ME.json) at the repository root).

The framework runs over vLLM-served open models (Llama 3.3, Qwen 3,
DeepSeek-R1) and hosted APIs (Gemini 2.5, GPT-4o).

## Installation

```bash
pip install -r requirements.txt
```

`vLLM` is **not** in `requirements.txt` because it is hardware-specific; install
it separately (see https://docs.vllm.ai/) only if you want to serve open-source
models locally. Hosted-API models need no vLLM.

For hosted APIs, copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env   # then set GEMINI_API_KEY / OPENAI_API_KEY
```

## Build the pipeline input

The pipeline operates on the meeting transcript, which is regenerated from the
released `AMI_ME.json`:

```bash
python data/build_meetingeval.py   # ../AMI_ME.json -> data/AMI/meetingeval.json + meetingeval_GTseg.json
```

## Serving open-source models (optional)

`start_server/model_path.json` maps short model names to local snapshot paths or
HuggingFace model IDs — edit it to point at the weights you have. Then, in a
separate terminal:

```bash
cd start_server && bash start_server.sh <model>   # serves on port 4060, TP=4, GPUs 0-3
```

`<model>` is a key from `start_server/model_path.json`, or one of the hosted
APIs `gpt` / `gemini` / `gemini-pro`.

## Running the pipeline

The entry point is [`vllm_eval.sh`](vllm_eval.sh). Edit the variables near the
top (`task`, `reasoning`, ...) and
invoke with a model name:

```bash
bash vllm_eval.sh <model>
```

Run the three `task` modes on the same `data_subfolder` / `model`. `seg` and
`obj` are independent, but both must finish before `eval`, which consumes their
outputs:

1. **`seg`** — topic segmentation ([`prompts/meetingeval/segmentation.txt`](prompts/meetingeval/segmentation.txt)).
   Runs several passes with `--continue_task` to reduce broken responses, then post-processes into
   `meetingeval_LLM.json` (segments with start/end utterance IDs + text).
2. **`obj`** — multi-label objective classification
   ([`prompts/meetingeval/objective_classification.txt`](prompts/meetingeval/objective_classification.txt))
   against the categories in [`data/AMI/objectives.txt`](data/AMI/objectives.txt).
   Produces `pure_obj.json`.
3. **`eval`** — per-segment effectiveness scoring. Pulls the `meetingeval_LLM.json`
   and `pure_obj.json` from the matching `_seg` / `_obj` run folders, scores each
   segment, then meta-evaluates against AMI-ME via
   [`analysis_scripts/meta_evaluate.py`](analysis_scripts/meta_evaluate.py).

Each run writes to `results/<data_subfolder>_<model><postfix>/`. The final
`pure_score.json` schema is:

```
{meeting_id: [overall_score, [segment_duration_sec, ...], [segment_score, ...], segment_score_distribution]}
```

### Score modes

The `reasoning` switch in `vllm_eval.sh` selects the model regime, setting thinking mode and scoring method together:

- **`reasoning="false"`** (non-reasoning): thinking off, token-probability-weighted score.
- **`reasoning="true"`** (reasoning): thinking on, mean over `sampling_n` samples.

The scoring prompt `segment_score.txt` is assembled from `before_eval_steps.txt`
\+ `eval_steps.txt` + `after_eval_steps.txt` via
`python post_process.py --task gen_eval_steps`.

### Segmentation and objective sources

The `eval` task can score on predicted or gold inputs, set near the top of
`vllm_eval.sh`:

- **`seg_source`** — `pred` (LLM segmentation from the `_seg` run) or `gt`
  (gold segmentation from `data/AMI/meetingeval_GTseg.json`, skips `seg`).
- **`use_GTobj`** — `--use_GTobj` scores against gold objectives
  (`data/AMI/objective.json`, skips `obj`); empty uses the `_obj` run's prediction.


## Meta-evaluation (reproducing the score correlations)

[`analysis_scripts/meta_evaluate.py`](analysis_scripts/meta_evaluate.py)
computes the segment-level Spearman (ρ) / Kendall (τ) correlation between a run's
predicted scores and the human ground truth. **Ground truth is read directly from the released
[`AMI_ME.json`](../AMI_ME.json)** — no other annotation files are needed.

```bash
python analysis_scripts/meta_evaluate.py \
    --prediction_dir <run_folder> \
    --save_dir <run_folder>/evaluation \
    --subset all            # all | scenario | nonscenario
```

`--subset nonscenario` selects the four `IB`-prefixed film/office meetings;
`scenario` selects the rest. [`meta_eval_all.sh`](meta_eval_all.sh) runs this
across every run folder under `results/`.

## Code layout

```
code/
├── vllm_eval.sh              # pipeline driver (seg -> obj -> eval)
├── vllm_eval_parallel.py     # LLM driver (thread-pooled per-segment requests)
├── post_process.py           # parsers + metrics + duration-weighted aggregation
├── utils.py                  # transcript parsing, <think> stripping, helpers
├── meta_eval_all.sh          # batch score meta-evaluation over results/
├── analysis_scripts/
│   └── meta_evaluate.py  # segment-level Spearman/Kendall vs AMI-ME GT
├── prompts/meetingeval/      # prompt templates (segmentation, objectives, scoring)
├── start_server/             # vLLM serving helper + model path map
└── data/
    ├── build_meetingeval.py  # regenerates data/AMI/meetingeval.json from ../AMI_ME.json
    └── AMI/
        ├── meetingeval.json  # input transcripts (generated; not tracked)
        ├── objectives.txt    # objective category list
        └── objective.json    # ground-truth objectives per meeting type (for --use_GTobj)
```

## Notes

- Reasoning models emit `<think>…</think>` blocks that are stripped before
  parsing. `--disable_thinking"` turns thinking off for Qwen 3
  (`enable_thinking=False`) and Gemini (`thinking_budget=0`); it is ignored by
  other models.
- `winsize=1` scores each segment alone; `winsize>1` adds surrounding segments as
  context.

## License

The code in this directory is released under the **Apache License 2.0** (see
[`LICENSE`](LICENSE)). The **AMI-ME dataset** at the repository root is released
separately under **CC BY 4.0** — see the top-level
[`README.md`](../README.md) and [`LICENSE`](../LICENSE).