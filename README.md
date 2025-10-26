AVPromptTuning 
 
> This repo provides code to reproduce  a practical **two-stage inference pipeline**: (1) route the command to a category (ROUTING / PARKING / TRAFFIC_MGMT / ENTERTAINMENT), (2) retrieve category-specific snippets and **build a prompt** using a template (prompt engineering).


##  Highlights

- **4 modes**: `baseline`, `context`, `prioritize`, `full`
- **Prompt engineering** via **category-specific templates**  and **in-category retrieval** 
- **Priority-aware training** (loss weighting with `priority_score`, tunable via `alpha_priority`)
- **Seed repeats** for stable reporting (mean ± std via `repeat_seeds`)

This repository includes a sample dataset derived from Talk2Car commands (annotated and grouped). During experiments, we evaluated multiple context-aware scenarios, including both safety-critical and non-critical cases (e.g., rain, school zone, peak traffic).
The paper reports the average performance across these mixed conditions.
The provided dataset in this repo reflects the neutral/base version. Training/evaluation scripts also support extended datasets with additional columns like:
is_safety_critical (0/1)
operating_context (e.g., PEAK_TRAFFIC, RAIN, SCHOOL_ZONE, …)
Users can reproduce or extend the results by providing their own annotated variations following the same schema.
Expected columns in CSV:
command_text (str)
context_text (str, optional but recommended)
target_label ∈ {ROUTING, PARKING, TRAFFIC_MGMT, ENTERTAINMENT}
priority_score ∈ [0,1] (used by prioritize/full)
