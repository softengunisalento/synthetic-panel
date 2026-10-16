# synthetic-panel

Code and data for the paper *Synthetic Expert Panels for the Validation of Small-Data Inverse Design Models: A Pilot Study on Waste-Filled PLA Composites for Fused Filament Fabrication* (R. Vergallo, D. Fico, S. Marfia, E. Monaldo, D. Renzi, C. Esposito Corcione; submitted to *Applied Sciences*).

The study tests whether a panel of synthetic experts, simulated with open-weight large language models (LLMs), can audit a small-data machine-learning framework for the inverse design of PLA composites filled with marble waste (MW) and olive wood waste (OW). Reference answers come from a hidden-formulation protocol: each composite formulation is removed in turn from the data given to all sources, and its measured flexural properties are used as ground truth. The same items were answered by three human experts, who also rated a sample of rationales blind to their source.

Repository: https://github.com/softengunisalento/synthetic-panel

## Contents

| Path | Content |
|---|---|
| `synthetic_panel.py` | The whole pipeline in one script (no notebook). The case-study dataset (60 flexural tests, 6 formulations) is embedded in it. |
| `requirements.txt` | Python dependencies |
| `expert_questionnaire.docx` | Questionnaire given to the human experts (blank) |
| `sp_output/personas.json` | The 100 expert personas, with the OpenAlex works they are grounded on |
| `sp_output/openalex_abstracts.csv` | Retrieved abstracts (OpenAlex, 25 September 2026) |
| `sp_output/items.json`, `sp_output/ground_truth.csv` | Questionnaire items and reference answers |
| `sp_output/ml_answers.csv` | Answers of the framework (with and without applicability domain) and of the interpolation baseline |
| `sp_output/models.json` | Selected models, with Hugging Face and Ollama identifiers |
| `sp_output/raw/*.jsonl` | Raw LLM answers, one line per respondent, with time stamps and reasoning setting |
| `sp_output/raw/human_main.jsonl` | Anonymized answers of the three human experts, transcribed from the questionnaires |
| `sp_output/humans/coding_log.txt` | Coding decisions applied to the human answers |
| `sp_output/rating/rating_expert_N.xlsx` | Blind ratings of rationales returned by the three experts |
| `sp_output/rating/rating_key.csv` | Source (model or human, respondent, item) of each rated rationale |
| `sp_output/tables/`, `sp_output/figures/` | All tables and figures of the paper |

## Requirements

- Python 3.12
- [Ollama](https://ollama.com) (the paper used version 0.34.4) for local inference
- About 60 GB of disk for the five models; the paper runs used an Apple M4 Max with 36 GB of unified memory

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=...        # optional: higher Hugging Face rate limits for the model screening
```

## Quick check (no models, about one minute)

```bash
python synthetic_panel.py --out test_output --mock all
```

`--mock` replaces the LLM calls with random answers and exercises the whole pipeline. Global options (`--out`, `--dataset`, `--mock`) go before the command.

## Full run

```bash
python synthetic_panel.py all
```

`all` runs, in order:

1. `ml`: internal validation of the framework (SVC + SMOTE for the filler family, Ridge for the content)
2. `personas`: OpenAlex retrieval, relevance filter, MiniLM embeddings, k-means (K = 10), 100 personas
3. `items`: questionnaire items, hidden-formulation reference answers, framework and baseline answers
4. `select-models --pull`: screening of the Hugging Face Hub with criteria IC1-IC7 and download of the selected GGUF models (Q4_K_M) into Ollama
5. `panel`: all models in the three conditions (C-main, 100 personas; C-nopersona, 50 draws; C-knowledge, 50 personas without the table of tested formulations)
6. `analyze`: metrics, tables and figures

Every answer is appended to `sp_output/raw/*.jsonl` as soon as it arrives. If a run stops, launching the same command again resumes from where it stopped. Steps already completed (personas, items, model selection) are not repeated.

Single steps:

```bash
python synthetic_panel.py select-models --pull
python synthetic_panel.py panel --model <key> --condition main --limit 2     # quick test of one model
python synthetic_panel.py panel --model <key> --condition nopersona
python synthetic_panel.py analyze
```

`<key>` is one of the keys in `sp_output/models.json`.

### Models selected for the paper (screening of 25 September 2026)

| Model | Parameters | GGUF source |
|---|---|---|
| Intern-S1-mini (domain stratum) | 8.3B | mradermacher/Intern-S1-mini-i1-GGUF |
| Qwen3-VL-8B-Instruct | 8.8B | Qwen/Qwen3-VL-8B-Instruct-GGUF |
| Gemma 4 26B A4B IT | 25.8B (MoE) | bartowski/google_gemma-4-26B-A4B-it-GGUF |
| gpt-oss-20b | 20.9B (MoE) | unsloth/gpt-oss-20b-GGUF |
| GLM-4.7-Flash | 31.2B (MoE) | unsloth/GLM-4.7-Flash-GGUF |

All models were run at Q4_K_M, context 16,384 tokens, temperature 1.0, with reasoning disabled or set to the lowest level when it cannot be disabled (gpt-oss-20b). A new screening on a later date may select different models; to reproduce the paper, keep the provided `sp_output/models.json`.

## Human experts

The experts answered `expert_questionnaire.docx` independently. Their answers were transcribed into `sp_output/raw/human_main.jsonl` with the rules in `sp_output/humans/coding_log.txt`, and `analyze` includes them in all comparisons.

For the blind rating, `rating-export` samples 30 synthetic rationales (stratified over items and models) and the human rationales, and writes one file per expert without the expert's own rationales. Each expert scores plausibility (1-5), flags physically incorrect statements (YES/NO) and guesses the source (Human/AI).

```bash
python synthetic_panel.py rating-export     # sp_output/rating/rating_expert_N.xlsx + rating_key.csv
python synthetic_panel.py rating-import     # after the experts return the files
```

`rating-import` writes:

- `tables/blind_rating.csv`: plausibility, share flagged and share judged human by source, with bootstrap 95% intervals for synthetic and human rationales
- `tables/blind_rating_source_accuracy.csv`: source recognition per expert and pooled (balanced accuracy)
- `tables/blind_rating_agreement.csv`: Krippendorff's alpha among experts on the synthetic rationales
- `tables/blind_rating_flagged.csv`: rationales flagged by at least two experts
- `tables/blind_rating_long.csv`: all ratings joined with their source

## Proprietary models (not used in the paper)

Connectors for two cloud models are included for future work and are not run by `all`:

```bash
export ANTHROPIC_API_KEY=...;  python synthetic_panel.py panel --model claude
export GEMINI_API_KEY=...;     python synthetic_panel.py panel --model gemini
```

## Main outputs

- `tables/block_A_summary.csv`: inverse design on hidden formulations (family accuracy, content error, interval coverage and score)
- `tables/block_B_summary.csv`: forward estimates on hidden formulations
- `tables/block_CD_design.csv`: design requirements and infeasible requests
- `tables/panel_agreement.csv`, `tables/conditions_RQ3.csv`, `tables/saturation.csv`: agreement, effect of the panel design, panel size
- `tables/model_screening.csv`, `tables/model_selection.csv`: full output of the model screening
- `tables/exclusion_rates.csv`: invalid answers per model and condition

## Citation

If you use this code or data, please cite the paper (reference to be added after publication).
