# Synthetic expert panel - experiment script

One Python script (`synthetic_panel.py`), no notebook. Tested on the dataset of the case study
(60 flexural tests, 6 formulations), which is embedded in the script.

## 1. Setup (once, macOS)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# local open-weight model (Intern-S1-mini) with Ollama
brew install ollama          # or download the app from ollama.com
ollama serve &               # not needed if the Ollama app is running
ollama pull internlm/interns1-mini

# API keys (paid tier recommended: prompts contain unpublished data)
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```

Model names, temperature, panel sizes and all other settings are at the top of the script
(section CONFIGURATION).

## 2. Dry run (no keys, no model, one minute)

```bash
python synthetic_panel.py --out test_output --mock all
```

## 3. Real run

```bash
python synthetic_panel.py all
```

This runs, in order: `ml`, `personas`, `items`, `panel` for the three models and the three
conditions, and `analyze`. Every answer is appended to `sp_output/raw/*.jsonl` as soon as it
arrives, so the run can be interrupted and restarted: requests already answered are skipped.
The local model runs one request at a time; expect several hours for its part.

Single steps, if needed:

```bash
python synthetic_panel.py panel --model claude --condition main
python synthetic_panel.py panel --model local --condition main --limit 5   # quick test
python synthetic_panel.py analyze
```

## 4. Human experts

```bash
python synthetic_panel.py humans-export     # sp_output/humans/questionnaire_template.xlsx
# send a copy to each expert, save the filled files as sp_output/humans/expert_1.xlsx, expert_2.xlsx, ...
python synthetic_panel.py humans-import
python synthetic_panel.py rating-export     # sp_output/rating/rating_expert_N.xlsx (own rationales removed)
# put the filled sheets back with the same names
python synthetic_panel.py rating-import
python synthetic_panel.py analyze
```

## 5. Open-weight model screening (Hugging Face)

```bash
python synthetic_panel.py screen-models     # sp_output/tables/hf_screening.csv
```

## Outputs

- `sp_output/tables/*.csv` and `*.tex`: all tables of the paper
- `sp_output/figures/*.pdf|png`: figures
- `sp_output/raw/*.jsonl`: raw answers with timestamps (to be released with the paper)
