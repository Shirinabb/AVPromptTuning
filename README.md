AVPromptTuning 
 
> This repo provides code to reproduce  a practical **two-stage inference pipeline**: (1) route the command to a category (ROUTING / PARKING / TRAFFIC_MGMT / ENTERTAINMENT), (2) retrieve category-specific snippets and **build a prompt** using a template (prompt engineering).


##  Highlights

- **4 modes**: `baseline`, `context`, `prioritize`, `full`
- **Prompt engineering** via **category-specific templates**  and **in-category retrieval** 
- **Priority-aware training** (loss weighting with `priority_score`, tunable via `alpha_priority`)
- **Seed repeats** for stable reporting (mean ± std via `repeat_seeds`)
