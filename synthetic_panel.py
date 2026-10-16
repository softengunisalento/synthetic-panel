#!/usr/bin/env python3
"""
Synthetic expert panel for the validation of small-data inverse design models.
Case study: PLA composites filled with marble waste (MW) and olive wood waste (OW), FFF.

One script, several steps (run `python synthetic_panel.py -h`):

  ml             reproduce the inverse design framework and its internal validation,
                 and compute the ML answers to the questionnaire items
  personas       build the literature-grounded personas from OpenAlex abstracts
  items          build the questionnaire (requests, contexts, ground truth)
  panel          query one LLM family under one condition (resumable)
  humans-export  write one Excel questionnaire for the human experts
  humans-import  read the filled Excel files back
  rating-export  write the blind rating sheet (synthetic + human rationales)
  rating-import  read the filled rating sheets back
  analyze        compute all metrics, LaTeX tables and figures
  select-models  select the open-weight models from the Hugging Face Hub with the predefined
                 rule, write models.json and (with --pull) download them into Ollama
  all            ml + personas + items + select-models + panel (all selected models,
                 all conditions) + analyze

Quick dry run without any API key or local model:
  python synthetic_panel.py all --mock

The panel uses open-weight models only, served locally by Ollama (http://localhost:11434).
Connectors for proprietary APIs (Anthropic, Gemini) are kept for future work: they are used only
if requested explicitly with `panel --model claude` or `panel --model gemini`
(API keys from ANTHROPIC_API_KEY, GEMINI_API_KEY). An optional HF_TOKEN raises the Hub rate limits.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import logging
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

# =============================================================================
# CONFIGURATION - edit here
# =============================================================================

OUT = Path("sp_output")                      # everything is written here

OLLAMA_HOST = "http://localhost:11434"
NUM_CTX = 16384

# Open-weight models are chosen by `select-models` and stored in OUT/models.json.
# Selection rule (see the paper, Section 3.4.3):
#  - domain stratum: the domain-adapted model retained by the screening of Table 6
#  - general stratum: the N_GENERAL most downloaded eligible models on the Hugging Face Hub,
#    at most one per developer organisation
DOMAIN_MODEL = "internlm/Intern-S1-mini"
SELECTION = dict(
    n_general=4,
    pipelines=["text-generation", "image-text-to-text"],
    pool_size=300,               # most downloaded repositories examined per pipeline tag
    min_params_b=7.0,            # below this size a model is not a credible expert
    max_params_b=32.0,           # 4-bit weights of at most ~20 GB on a 36 GB unified-memory Mac
    max_age_months=18,           # repository created within 18 months of the screening date
    min_context=16000,
    quant="Q4_K_M",              # the same quantization for every model
)

# Proprietary APIs, kept for future work (not used by `all`).
API_MODELS = {
    "claude": {"provider": "anthropic", "model": "claude-sonnet-5", "workers": 8},
    "gemini": {"provider": "gemini", "model": "gemini-3.8-flash", "workers": 8,
               "thinking_budget": 0},
}

TEMPERATURE = 1.0
MAX_OUTPUT_TOKENS = 2048        # API models
LOCAL_MAX_TOKENS = 6144         # local models: room for a reasoning trace when it cannot be disabled
N_PERSONAS = 100            # personas in the main condition
N_NOPERSONA = 50            # independent draws per model in the no-persona condition
N_KNOWLEDGE = 50            # personas used in the knowledge-only condition
CONDITIONS = ["main", "nopersona", "knowledge"]
SEED = 42

# Persona construction
OPENALEX_QUERIES = [
    "polylactic acid composite filler mechanical properties",
    "PLA natural fiber wood flour composite",
    "marble waste powder polymer composite",
    "mineral filler calcium carbonate PLA",
    "fused filament fabrication composite filament extrusion",
    "3D printed polymer flexural properties",
]
OPENALEX_PER_QUERY = 120
OPENALEX_MAILTO = "roberto.vergallo@unipegaso.it"   # polite pool of the OpenAlex API
# Relevance filter on abstracts (regular expressions, case-insensitive).
# An abstract is kept if it matches at least one REQUIRE pattern and no EXCLUDE pattern.
ABSTRACT_REQUIRE_ANY = [r"\bpla\b", r"polylactic", r"polylactide", r"poly\(lactic", r"thermoplastic",
                        r"\bpolymer", r"\bfilaments?\b", r"fused filament", r"\bfdm\b", r"\bfff\b"]
ABSTRACT_EXCLUDE_ANY = [r"\bconcrete", r"\bcement", r"\bmortar", r"\bbone\b", r"scaffold", r"\btissue",
                        r"graphene", r"carbon nanotube", r"\bcnts?\b", r"conductiv", r"stereolithograph",
                        r"\bsla\b", r"\bdlp\b", r"photopolymer"]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
N_CLUSTERS = 10
ABSTRACTS_PER_PERSONA = 4
ABSTRACT_MAX_WORDS = 150
ROLES = ["academic researcher", "industrial R&D engineer", "process technician"]
SENIORITY = [5, 10, 20]

# Inverse design framework (as in the case study)
SVC_PARAMS = dict(C=0.7, kernel="rbf", class_weight="balanced", random_state=42)
SMOTE_K = 3
RIDGE_ALPHA = 3.0
FEATURES = ["flexural_modulus", "flexural_strength", "strain_at_break"]

# Design-requirement items (no known answer) and infeasible items
DESIGN_TARGETS = {
    "C1": (3.6, 49.0, 1.7),     # close to a tested MW formulation
    "C2": (3.5, 58.0, 2.1),     # between two tested MW levels
    "C3": (1.9, 38.0, 3.4),     # between the two tested OW levels
}
INFEASIBLE_TARGETS = {
    "D1": (4.5, 110.0, 6.0),    # stiff, strong and ductile at the same time
    "D2": (1.0, 30.0, 3.6),     # would need OW content above 20 wt.% (not extrudable)
}

# Analysis
SATURATION_SIZES = [5, 10, 20, 50, 100]
SATURATION_REPS = 1000
BOOTSTRAP_REPS = 10000
N_RATIONALES_SYNTHETIC = 30
N_RATIONALES_PER_HUMAN = 3

# =============================================================================
# DATASET (60 specimens, flexural tests, ISO 178). Tg is not used.
# =============================================================================

DATA_CSV = """\
material,percentage,flexural_modulus,flexural_strength,strain_at_break
NA,0,3.74,103.95,4.18
NA,0,3.69,99.88,3.83
NA,0,3.54,103.76,4.11
NA,0,3.80,104.76,4.86
NA,0,3.95,100.34,3.86
NA,0,3.75,102.76,4.09
NA,0,3.82,104.67,4.63
NA,0,3.45,103.81,3.91
NA,0,3.78,104.96,4.78
NA,0,3.91,103.46,3.57
MW,10,3.02,70.81,2.97
MW,10,3.30,72.73,3.54
MW,10,3.20,74.01,3.57
MW,10,3.20,71.53,2.74
MW,10,3.01,70.50,3.17
MW,10,3.21,71.52,3.06
MW,10,3.13,70.74,3.16
MW,10,3.31,73.61,2.96
MW,10,3.10,72.95,3.34
MW,10,3.09,72.32,3.21
MW,20,4.03,66.36,2.59
MW,20,4.43,66.28,2.28
MW,20,3.99,70.29,2.29
MW,20,4.02,69.47,2.69
MW,20,3.87,65.89,2.59
MW,20,3.98,67.78,2.53
MW,20,4.34,66.91,2.44
MW,20,4.45,67.32,2.51
MW,20,4.13,67.89,2.62
MW,20,3.84,70.09,2.41
MW,30,3.39,47.15,1.45
MW,30,3.88,49.12,1.98
MW,30,3.73,51.01,1.54
MW,30,3.60,48.04,1.86
MW,30,3.58,46.11,1.46
MW,30,3.65,48.12,1.72
MW,30,3.57,50.31,1.41
MW,30,3.76,48.61,1.54
MW,30,3.36,50.01,1.77
MW,30,3.89,47.46,1.80
OW,10,2.15,40.76,3.45
OW,10,2.24,40.98,3.23
OW,10,1.98,41.09,3.12
OW,10,2.12,40.79,3.06
OW,10,2.23,40.99,3.41
OW,10,2.36,40.71,3.09
OW,10,2.61,41.28,3.31
OW,10,3.12,40.76,3.27
OW,10,2.57,41.78,3.25
OW,10,2.13,40.87,3.11
OW,20,1.23,34.86,3.76
OW,20,1.39,34.96,3.23
OW,20,1.54,34.45,3.61
OW,20,1.42,34.63,3.14
OW,20,1.55,34.97,3.56
OW,20,1.34,34.99,3.64
OW,20,1.47,34.95,3.67
OW,20,1.61,34.55,3.23
OW,20,1.50,34.73,3.17
OW,20,1.38,34.38,3.98
"""

UNITS = {"flexural_modulus": "GPa", "flexural_strength": "MPa", "strain_at_break": "%"}
SHORT = {"flexural_modulus": "E", "flexural_strength": "sigma", "strain_at_break": "eps"}
FAMILY_NAME = {"NA": "none (neat PLA)", "MW": "marble waste (MW)", "OW": "olive wood waste (OW)"}


def load_dataset(path: str | None = None) -> pd.DataFrame:
    if path:
        df = pd.read_excel(path, keep_default_na=False)
        df.columns = ["material", "percentage", "Tg"] + FEATURES
        for c in ["percentage"] + FEATURES:
            df[c] = df[c].astype(str).str.replace(",", ".", regex=False).astype(float)
        df["material"] = df["material"].astype(str).str.strip().str.upper()
        df = df[["material", "percentage"] + FEATURES]
    else:
        df = pd.read_csv(io.StringIO(DATA_CSV), keep_default_na=False)
    df["formulation"] = df["material"] + "_" + df["percentage"].map(lambda x: f"{x:g}")
    return df.reset_index(drop=True)


def centroids(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["material", "percentage"])[FEATURES]
    out = g.mean().join(g.std(), rsuffix="_sd").reset_index()
    out["formulation"] = out["material"] + "_" + out["percentage"].map(lambda x: f"{x:g}")
    return out


def label(material: str, pct: float) -> str:
    return "100PLA" if material == "NA" else f"{100 - pct:g}PLA/{pct:g}{material}"


# =============================================================================
# UTILITIES
# =============================================================================

def ensure_dirs():
    for d in ["", "raw", "tables", "figures", "humans", "rating"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def df_to_latex(df: pd.DataFrame, path: Path, caption: str, label_: str, floatfmt="%.2f"):
    body = df.to_latex(index=False, float_format=lambda x: floatfmt % x, escape=True)
    path.write_text(f"% {caption}\n% label: {label_}\n{body}", encoding="utf-8")


def log(msg: str):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


# =============================================================================
# STEP 1 - INVERSE DESIGN FRAMEWORK
# =============================================================================

def make_classifier():
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    return ImbPipeline([("scaler", StandardScaler()),
                        ("smote", SMOTE(random_state=42, k_neighbors=SMOTE_K)),
                        ("model", SVC(**SVC_PARAMS))])


def make_regressor():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=RIDGE_ALPHA))])


class Framework:
    """Two-stage framework trained on a (possibly reduced) dataset."""

    def __init__(self, df: pd.DataFrame):
        from scipy.spatial.distance import mahalanobis
        self.df = df
        self.clf = make_classifier().fit(df[FEATURES], df["material"])
        self.reg = {}
        for fam in ["MW", "OW"]:
            sub = df[df.material == fam]
            if len(sub):
                self.reg[fam] = make_regressor().fit(sub[FEATURES], sub["percentage"])
        X = df[FEATURES].to_numpy(float)
        self.mu = X.mean(0)
        self.vi = np.linalg.pinv(np.cov(X, rowvar=False))
        self._m = mahalanobis
        d = [mahalanobis(x, self.mu, self.vi) for x in X]
        self.ad_threshold = float(np.percentile(d, 95))

    def predict(self, target: tuple[float, float, float]) -> dict:
        x = pd.DataFrame([dict(zip(FEATURES, target))])[FEATURES]
        dist = float(self._m(x.to_numpy(float)[0], self.mu, self.vi))
        fam = str(self.clf.predict(x)[0])
        content = 0.0 if fam == "NA" else float(self.reg[fam].predict(x)[0])
        fam = "none" if fam == "NA" else fam
        in_domain = dist <= self.ad_threshold
        return {"family": fam if in_domain else "infeasible",
                "family_raw": fam, "content_wt": content if in_domain else None,
                "content_raw": content, "mahalanobis": dist,
                "ad_threshold": self.ad_threshold, "in_domain": in_domain}


def internal_validation(df: pd.DataFrame):
    """Reproduces the internal validation of the case study (Table 2 of the paper)."""
    from sklearn.base import clone
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    regs = {
        "Ridge regression": make_regressor(),
        "Polynomial Ridge (2nd order)": Pipeline([("s", StandardScaler()),
                                                  ("p", PolynomialFeatures(2, include_bias=False)),
                                                  ("r", Ridge(alpha=8.0))]),
        "Linear regression": Pipeline([("s", StandardScaler()), ("l", LinearRegression())]),
        "Gradient boosting": GradientBoostingRegressor(n_estimators=80, learning_rate=0.03, max_depth=2,
                                                       min_samples_leaf=3, subsample=0.85, random_state=42),
        "Random forest": RandomForestRegressor(n_estimators=150, max_depth=4, min_samples_leaf=4,
                                               min_samples_split=8, max_features="sqrt", random_state=42),
        "Mean-prediction baseline": DummyRegressor(strategy="mean"),
    }
    mw = df[df.material == "MW"].reset_index(drop=True)
    X, y = mw[FEATURES], mw["percentage"]
    rows, fold_rows = [], []
    for name, base in regs.items():
        pred_cv = cross_val_predict(clone(base), X, y, cv=KFold(5, shuffle=True, random_state=42))
        oof = np.full(len(y), np.nan)
        for tr, te in LeaveOneGroupOut().split(X, y, groups=y.astype(str)):
            m = clone(base).fit(X.iloc[tr], y.iloc[tr])
            oof[te] = m.predict(X.iloc[te])
            fold_rows.append({"model": name, "held_out_wt": float(y.iloc[te].iloc[0]),
                              "MAE": mean_absolute_error(y.iloc[te], oof[te])})
        rows.append({"model": name,
                     "specimen_wise_RMSE": np.sqrt(mean_squared_error(y, pred_cv)),
                     "LOFO_MAE": mean_absolute_error(y, oof),
                     "LOFO_RMSE": np.sqrt(mean_squared_error(y, oof)),
                     "LOFO_R2_pooled": r2_score(y, oof)})
    reg_df = pd.DataFrame(rows)
    folds_df = pd.DataFrame(fold_rows)

    clfs = {
        "SVC": make_classifier(),
        "Logistic regression": ImbPipeline([("s", StandardScaler()),
                                            ("smote", SMOTE(random_state=42, k_neighbors=SMOTE_K)),
                                            ("m", LogisticRegression(C=0.5, max_iter=3000,
                                                                     class_weight="balanced",
                                                                     random_state=42))]),
        "Random forest": ImbPipeline([("smote", SMOTE(random_state=42, k_neighbors=SMOTE_K)),
                                      ("m", RandomForestClassifier(n_estimators=150, max_depth=4,
                                                                   min_samples_leaf=3, min_samples_split=6,
                                                                   class_weight="balanced",
                                                                   random_state=42))]),
        "Most-frequent baseline": DummyClassifier(strategy="most_frequent"),
    }
    held = sorted(df.loc[df.material != "NA", "formulation"].unique())
    crow = []
    for name, base in clfs.items():
        accs = {}
        for h in held:
            tr, te = df.formulation != h, df.formulation == h
            m = clone(base).fit(df.loc[tr, FEATURES], df.loc[tr, "material"])
            accs[h] = accuracy_score(df.loc[te, "material"], m.predict(df.loc[te, FEATURES]))
        crow.append({"classifier": name, **accs, "mean": float(np.mean(list(accs.values())))})
    clf_df = pd.DataFrame(crow)
    return reg_df, folds_df, clf_df


def cmd_ml(args):
    ensure_dirs()
    df = load_dataset(args.dataset)
    reg_df, folds_df, clf_df = internal_validation(df)
    reg_df.to_csv(OUT / "tables" / "internal_validation_regression.csv", index=False)
    folds_df.to_csv(OUT / "tables" / "internal_validation_regression_folds.csv", index=False)
    clf_df.to_csv(OUT / "tables" / "internal_validation_classification.csv", index=False)
    df_to_latex(reg_df.round(3), OUT / "tables" / "tab_internal_regression.tex",
                "Internal validation of the MW regressors", "tab:internal")
    log("Internal validation (MW regressors):\n" + reg_df.round(3).to_string(index=False))
    log("LOFO folds, Ridge:\n" + folds_df[folds_df.model == "Ridge regression"].round(2).to_string(index=False))
    log("LOFO classification accuracy:\n" + clf_df.round(2).to_string(index=False))
    log("ML answers are computed in the 'items' step, together with the questionnaire.")


# =============================================================================
# STEP 2 - PERSONAS
# =============================================================================

def _openalex_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = {}
    for word, idx in inv.items():
        for i in idx:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def is_relevant(text: str) -> bool:
    t = text.lower()
    return (any(re.search(p, t) for p in ABSTRACT_REQUIRE_ANY)
            and not any(re.search(p, t) for p in ABSTRACT_EXCLUDE_ANY))


def fetch_openalex() -> pd.DataFrame:
    """Retrieves up to OPENALEX_PER_QUERY relevant abstracts per query (after the relevance filter)."""
    import requests
    recs = []
    for q in OPENALEX_QUERIES:
        cursor, got, seen, dropped = "*", 0, 0, 0
        while got < OPENALEX_PER_QUERY and cursor:
            r = requests.get("https://api.openalex.org/works", timeout=60, params={
                "search": q, "per-page": 100, "cursor": cursor, "mailto": OPENALEX_MAILTO,
                "filter": "type:article,has_abstract:true,language:en,from_publication_date:2010-01-01",
                "select": "id,title,publication_year,abstract_inverted_index"})
            r.raise_for_status()
            js = r.json()
            if not js["results"]:
                break
            for w in js["results"]:
                ab = _openalex_abstract(w.get("abstract_inverted_index"))
                if len(ab.split()) < 60:
                    continue
                seen += 1
                if not is_relevant(ab):
                    dropped += 1
                    continue
                recs.append({"openalex_id": w["id"], "year": w.get("publication_year"),
                             "query": q, "abstract": ab})
                got += 1
                if got >= OPENALEX_PER_QUERY:
                    break
            cursor = js.get("meta", {}).get("next_cursor")
            time.sleep(0.2)
        log(f"OpenAlex '{q}': {got} abstracts kept, {dropped} of {seen} excluded by the relevance filter")
    df = pd.DataFrame(recs).drop_duplicates("openalex_id").reset_index(drop=True)
    return df


def _truncate(text: str, n: int) -> str:
    w = text.split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def cmd_personas(args):
    ensure_dirs()
    rng = random.Random(SEED)
    if args.mock:
        abstracts = pd.DataFrame({"openalex_id": [f"mock{i}" for i in range(200)],
                                  "year": 2020, "query": "mock",
                                  "abstract": [f"Mock abstract {i} on PLA composites with fillers. " * 10
                                               for i in range(200)]})
        labels = np.arange(200) % N_CLUSTERS
        cluster_terms = {k: f"mock topic {k}" for k in range(N_CLUSTERS)}
    else:
        abstracts = fetch_openalex()
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
        emb = SentenceTransformer(EMBEDDING_MODEL).encode(abstracts.abstract.tolist(),
                                                          show_progress_bar=True, normalize_embeddings=True)
        labels = KMeans(N_CLUSTERS, n_init=10, random_state=SEED).fit_predict(emb)
        tfidf = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
        M = tfidf.fit_transform(abstracts.abstract)
        vocab = np.array(tfidf.get_feature_names_out())
        cluster_terms = {}
        for k in range(N_CLUSTERS):
            mean = np.asarray(M[labels == k].mean(axis=0)).ravel()
            cluster_terms[k] = ", ".join(vocab[np.argsort(mean)[::-1][:5]])
    abstracts["cluster"] = labels
    abstracts.to_csv(OUT / "openalex_abstracts.csv", index=False)

    combos = [(k, r, s) for k in range(N_CLUSTERS) for r in ROLES for s in SENIORITY]
    rng.shuffle(combos)
    personas = []
    for i in range(N_PERSONAS):
        k, role, sen = combos[i % len(combos)]
        pool = abstracts[abstracts.cluster == k]
        pick = pool.sample(n=min(ABSTRACTS_PER_PERSONA, len(pool)), random_state=SEED + i)
        personas.append({"persona_id": f"P{i + 1:03d}", "cluster": int(k),
                         "cluster_terms": cluster_terms[k], "role": role, "seniority_years": sen,
                         "abstract_ids": pick.openalex_id.tolist(),
                         "abstracts": [_truncate(a, ABSTRACT_MAX_WORDS) for a in pick.abstract]})
    save_json(personas, OUT / "personas.json")
    summary = (abstracts.groupby("cluster").size().rename("n_abstracts").to_frame()
               .join(pd.Series(cluster_terms, name="top_terms"))
               .join(pd.Series([p["cluster"] for p in personas]).value_counts().rename("n_personas")))
    summary.to_csv(OUT / "tables" / "persona_clusters.csv")
    log(f"{len(abstracts)} abstracts, {len(personas)} personas written")


# =============================================================================
# STEP 3 - QUESTIONNAIRE
# =============================================================================

MATERIALS_TEXT = """Matrix: PLA Ingeo 4043D (NatureWorks), density 1.24 g/cm3, MFI 6 g/10 min at 210 C.
Fillers: (i) marble waste (MW), a powder obtained by drying the sludge of water-jet cutting of marble,
heterogeneous particle size; (ii) olive wood waste (OW), a coarse lignocellulosic powder from olive-wood
craft production. Both fillers are milled to about 0.25 mm, PLA pellets to about 0.75 mm.
Processing: all materials dried at 60 C for 24 h, dry-mixed, extruded with a 3Devo Composer 450 single-screw
extruder into 1.75 mm filaments, printed with a 1 mm nozzle at 200 C (Creality CP-01).
Testing: three-point bending, ISO 178, 80 x 10 x 4 mm bars, 2 mm/min, span 64 mm; E = flexural modulus,
sigma = flexural strength, eps = strain at break. Filler contents are expressed in wt.%.
Known processing limit: a formulation with 30 wt.% OW could not be extruded into a continuous filament."""


def context_table(cent: pd.DataFrame, hide: str | None, knowledge_only: bool) -> str:
    rows = cent if not knowledge_only else cent[cent.material == "NA"]
    if hide:
        rows = rows[rows.formulation != hide]
    rows = rows.assign(_o=rows.material.map({"NA": 0, "MW": 1, "OW": 2})).sort_values(["_o", "percentage"])
    lines = ["Formulation | filler | wt.% | E [GPa] | sigma [MPa] | eps [%]  (mean +/- SD, 10 specimens)"]
    for _, r in rows.iterrows():
        fill = "none" if r.material == "NA" else r.material
        lines.append(f"{label(r.material, r.percentage)} | {fill} | {r.percentage:g} | "
                     f"{r.flexural_modulus:.2f} +/- {r.flexural_modulus_sd:.2f} | "
                     f"{r.flexural_strength:.2f} +/- {r.flexural_strength_sd:.2f} | "
                     f"{r.strain_at_break:.2f} +/- {r.strain_at_break_sd:.2f}")
    return "\n".join(lines)


def inverse_text(item_id, t):
    return (f"[{item_id}] Inverse design. A client needs printed parts with flexural modulus "
            f"{t[0]:.2f} GPa, flexural strength {t[1]:.2f} MPa and strain at break {t[2]:.2f} %. "
            f"Which formulation would you propose (filler family and filler content in wt.%)? "
            f"If no processable formulation of this material system can meet the request, say so.")


def forward_text(item_id, mat, pct):
    return (f"[{item_id}] Forward estimate. Estimate the flexural modulus, flexural strength and strain "
            f"at break of printed bars of {label(mat, pct)} ({pct:g} wt.% {FAMILY_NAME[mat]}).")


def interpolation_baseline(df_visible: pd.DataFrame):
    """Piecewise-linear composition paths through the visible formulation centroids,
    anchored at neat PLA (0 wt.%) for both families."""
    from sklearn.preprocessing import StandardScaler
    cent = centroids(df_visible)
    sc = StandardScaler().fit(df_visible[FEATURES])
    na = cent[cent.material == "NA"]
    paths = {}
    for fam in ["MW", "OW"]:
        p = pd.concat([na.assign(material=fam), cent[cent.material == fam]]).sort_values("percentage")
        paths[fam] = (p["percentage"].to_numpy(float), p[FEATURES].to_numpy(float))

    def inverse(target):
        z = sc.transform(pd.DataFrame([dict(zip(FEATURES, target))])[FEATURES])[0]
        best = (np.inf, None, None)
        for fam, (pct, P) in paths.items():
            Z = sc.transform(pd.DataFrame(P, columns=FEATURES))
            for i in range(len(pct) - 1):
                a, b = Z[i], Z[i + 1]
                t = float(np.clip(np.dot(z - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0, 1))
                d = float(np.linalg.norm(z - (a + t * (b - a))))
                if d < best[0]:
                    best = (d, fam, pct[i] + t * (pct[i + 1] - pct[i]))
        fam, content = best[1], best[2]
        if content is not None and content < 0.5:
            fam, content = "none", 0.0
        return {"family": fam, "content_wt": content, "distance": best[0]}

    def forward(fam, pct_q):
        pct, P = paths[fam]
        out = {}
        for j, f in enumerate(FEATURES):
            if pct.min() <= pct_q <= pct.max():
                out[f] = float(np.interp(pct_q, pct, P[:, j]))
            else:
                i0, i1 = (0, 1) if pct_q < pct.min() else (-2, -1)
                slope = (P[i1, j] - P[i0, j]) / (pct[i1] - pct[i0])
                out[f] = float(P[i1, j] + slope * (pct_q - pct[i1]))
        return out

    return inverse, forward


def cmd_items(args):
    """Builds the requests. Each request is answered in one call.
    A_k / B_k: formulation k is hidden from the context (leave-one-formulation-out elicitation).
    CD: design requirements (C, no known answer) and infeasible requests (D), full context."""
    ensure_dirs()
    df = load_dataset(args.dataset)
    cent = centroids(df)
    composite = cent[cent.material != "NA"].reset_index(drop=True)
    requests_ = []
    truth = []
    ml_rows = []
    for idx, r in composite.iterrows():
        k = r.formulation
        target = tuple(float(r[f]) for f in FEATURES)
        visible = df[df.formulation != k]
        fw = Framework(visible)
        inv, fwd = interpolation_baseline(visible)
        a_id, b_id = f"A{idx + 1}", f"B{idx + 1}"
        requests_.append({"request_id": f"R_{a_id}", "hide": k, "full_context": False,
                          "items": [{"item_id": a_id, "type": "inverse", "target": target,
                                     "text": inverse_text(a_id, target)}]})
        requests_.append({"request_id": f"R_{b_id}", "hide": k, "full_context": False,
                          "items": [{"item_id": b_id, "type": "forward", "material": r.material,
                                     "percentage": float(r.percentage),
                                     "text": forward_text(b_id, r.material, r.percentage)}]})
        truth.append({"item_id": a_id, "block": "A", "hidden": k, "family": r.material,
                      "content_wt": float(r.percentage), "feasible": True,
                      **{f: float(r[f]) for f in FEATURES}})
        truth.append({"item_id": b_id, "block": "B", "hidden": k, "family": r.material,
                      "content_wt": float(r.percentage), "feasible": True,
                      **{f: float(r[f]) for f in FEATURES}})
        p = fw.predict(target)
        ml_rows.append({"item_id": a_id, "source": "ML", **p})
        ml_rows.append({"item_id": a_id, "source": "ML (no AD)", **p,
                        "family": p["family_raw"], "content_wt": p["content_raw"]})
        bi = inv(target)
        ml_rows.append({"item_id": a_id, "source": "Baseline", "family": bi["family"],
                        "content_wt": bi["content_wt"]})
        bf = fwd(r.material, float(r.percentage))
        ml_rows.append({"item_id": b_id, "source": "Baseline", **bf})

    full_items = []
    fw_all = Framework(df)
    inv_all, _ = interpolation_baseline(df)
    for iid, t in {**DESIGN_TARGETS, **INFEASIBLE_TARGETS}.items():
        full_items.append({"item_id": iid, "type": "inverse", "target": t, "text": inverse_text(iid, t)})
        blk = iid[0]
        truth.append({"item_id": iid, "block": blk, "hidden": None,
                      "family": "infeasible" if blk == "D" else None, "content_wt": None,
                      "feasible": False if blk == "D" else None,
                      **dict(zip(FEATURES, t))})
        p = fw_all.predict(t)
        ml_rows.append({"item_id": iid, "source": "ML", **p})
        ml_rows.append({"item_id": iid, "source": "ML (no AD)", **p,
                        "family": p["family_raw"], "content_wt": p["content_raw"]})
        bi = inv_all(t)
        ml_rows.append({"item_id": iid, "source": "Baseline", "family": bi["family"],
                        "content_wt": bi["content_wt"]})
    requests_.append({"request_id": "R_CD", "hide": None, "full_context": True, "items": full_items})

    save_json({"materials_text": MATERIALS_TEXT,
               "centroids": cent.to_dict(orient="records"),
               "requests": requests_}, OUT / "items.json")
    pd.DataFrame(truth).to_csv(OUT / "ground_truth.csv", index=False)
    pd.DataFrame(ml_rows).to_csv(OUT / "ml_answers.csv", index=False)
    log(f"{len(requests_)} requests, {sum(len(r['items']) for r in requests_)} items written")
    log("ML and baseline answers:\n" + pd.DataFrame(ml_rows)[["item_id", "source", "family", "content_wt"]]
        .round(2).to_string(index=False))


# =============================================================================
# STEP 4 - PROMPTS AND QUERYING
# =============================================================================

SYSTEM_PERSONA = """You are the expert described below. Answer the questionnaire using your own expertise and
the information provided. Answer as this expert would, with realistic uncertainty.
Reply ONLY with a JSON object that follows the schema. No text outside the JSON."""

SYSTEM_GENERIC = """You are an experienced materials engineer. Answer the questionnaire using your own expertise
and the information provided, with realistic uncertainty.
Reply ONLY with a JSON object that follows the schema. No text outside the JSON."""

SCHEMA_INVERSE = """For each inverse design item, return:
{"item_id": str, "family": "none" | "MW" | "OW" | "infeasible", "content_wt": number or null,
 "content_low80": number or null, "content_high80": number or null,
 "feasibility": "feasible" | "infeasible" | "uncertain", "confidence": integer 1-5,
 "rationale": str (max 50 words)}
Use family "none" for neat PLA (content 0). Use "infeasible" with null contents if no processable
formulation of this system can meet the request. [content_low80, content_high80] is your 80% interval."""

SCHEMA_FORWARD = """For each forward estimate item, return:
{"item_id": str, "E_GPa": [low80, point, high80], "sigma_MPa": [low80, point, high80],
 "eps_pct": [low80, point, high80], "confidence": integer 1-5, "rationale": str (max 50 words)}
Each list gives the lower bound of your 80% interval, your point estimate and the upper bound."""


def schema_text(items) -> str:
    """Only the schema of the item types present in the request is shown."""
    ids = ", ".join(f'"{i["item_id"]}"' for i in items)
    parts = ['Output schema: {"answers": [ <one object per item> ]}',
             f"Use exactly these item_id values: {ids}."]
    if any(i["type"] == "inverse" for i in items):
        parts.append(SCHEMA_INVERSE)
    if any(i["type"] == "forward" for i in items):
        parts.append(SCHEMA_FORWARD)
    return "\n".join(parts)


def persona_block(p: dict) -> str:
    abs_txt = "\n".join(f"- {a}" for a in p["abstracts"])
    return (f"Expert profile: {p['role']} with {p['seniority_years']} years of experience. "
            f"Main area of work: {p['cluster_terms']}.\n"
            f"Research you know well (summaries of papers in your area):\n{abs_txt}")


def build_prompt(req: dict, items_meta: dict, persona: dict | None, condition: str, rng: random.Random):
    cent = pd.DataFrame(items_meta["centroids"])
    ctx = context_table(cent, req["hide"], knowledge_only=(condition == "knowledge"))
    items = list(req["items"])
    rng.shuffle(items)
    user = (f"MATERIAL SYSTEM\n{items_meta['materials_text']}\n\n"
            f"TESTED FORMULATIONS\n{ctx}\n\n"
            f"QUESTIONNAIRE\n" + "\n".join(i["text"] for i in items) + f"\n\n{schema_text(items)}")
    if condition == "nopersona" or persona is None:
        system = SYSTEM_GENERIC
    else:
        system = SYSTEM_PERSONA + "\n\n" + persona_block(persona)
    return system, user


def extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in output")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON")


def to_num(x):
    """Numbers are accepted also when written as strings; 'null', 'none', 'n/a' and '' become None."""
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    t = str(x).strip().lower()
    if t in ("", "null", "none", "n/a", "na", "nan"):
        return None
    try:
        return float(t.replace(",", "."))
    except ValueError:
        return None


def validate(answer: dict, req: dict) -> list[dict]:
    lst = [a for a in answer.get("answers", []) if isinstance(a, dict)]
    ans = {str(a.get("item_id", "")).strip("[] "): a for a in lst}
    if len(req["items"]) == 1 and len(lst) == 1:          # single item: the label is not needed
        ans = {req["items"][0]["item_id"]: lst[0]}
    out = []
    for it in req["items"]:
        a = ans.get(it["item_id"])
        if a is None:
            raise ValueError(f"missing item {it['item_id']}")
        if it["type"] == "inverse":
            fam = str(a.get("family", "")).strip()
            fam = {"none": "none", "NA": "none", "MW": "MW", "OW": "OW", "infeasible": "infeasible"}.get(fam)
            if fam is None:
                raise ValueError("bad family")
            c, lo, hi = (to_num(a.get(k)) for k in ("content_wt", "content_low80", "content_high80"))
            if fam == "none":
                c = 0.0 if c is None else c
            if fam in ("MW", "OW"):
                if not all(isinstance(v, (int, float)) for v in (c, lo, hi)):
                    raise ValueError("missing contents")
                if not (0 <= lo <= c <= hi <= 100):
                    raise ValueError("inconsistent interval")
            out.append({"item_id": it["item_id"], "type": "inverse", "family": fam,
                        "content_wt": c, "content_low80": lo, "content_high80": hi,
                        "feasibility": a.get("feasibility"), "confidence": to_num(a.get("confidence")),
                        "rationale": a.get("rationale", "")})
        else:
            row = {"item_id": it["item_id"], "type": "forward", "confidence": to_num(a.get("confidence")),
                   "rationale": a.get("rationale", "")}
            for key, f in [("E_GPa", "flexural_modulus"), ("sigma_MPa", "flexural_strength"),
                           ("eps_pct", "strain_at_break")]:
                v = a.get(key)
                v = [to_num(x) for x in v] if isinstance(v, list) else v
                if not (isinstance(v, list) and len(v) == 3 and all(x is not None for x in v)):
                    raise ValueError(f"bad {key}")
                lo, pt, hi = v
                if not (lo <= pt <= hi):
                    raise ValueError(f"inconsistent {key}")
                row[f] = pt
                row[f + "_low80"] = lo
                row[f + "_high80"] = hi
            out.append(row)
    return out


# ---- providers --------------------------------------------------------------

def call_anthropic(cfg, system, user):
    import anthropic
    client = getattr(call_anthropic, "_c", None) or anthropic.Anthropic()
    call_anthropic._c = client
    r = client.messages.create(model=cfg["model"], max_tokens=MAX_OUTPUT_TOKENS, temperature=TEMPERATURE,
                               system=system, messages=[{"role": "user", "content": user}])
    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    if not text.strip():
        types_ = [getattr(b, "type", "?") for b in r.content]
        return f"EMPTY OUTPUT stop_reason={r.stop_reason} blocks={types_}"
    return text


def call_gemini(cfg, system, user):
    from google import genai
    from google.genai import types
    client = getattr(call_gemini, "_c", None) or genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    call_gemini._c = client
    kw = dict(system_instruction=system, temperature=TEMPERATURE, max_output_tokens=MAX_OUTPUT_TOKENS,
              response_mime_type="application/json")
    if cfg.get("thinking_budget") is not None:
        kw["thinking_config"] = types.ThinkingConfig(thinking_budget=cfg["thinking_budget"])
    try:
        r = client.models.generate_content(model=cfg["model"], contents=user,
                                           config=types.GenerateContentConfig(**kw))
    except Exception as e:  # some models do not accept a zero thinking budget
        if "thinking" in str(e).lower() and "thinking_config" in kw:
            log(f"Gemini: thinking_budget not accepted ({e}); using the model default")
            kw.pop("thinking_config")
            r = client.models.generate_content(model=cfg["model"], contents=user,
                                               config=types.GenerateContentConfig(**kw))
        else:
            raise
    return r.text


THINK_USED: dict[str, object] = {}   # reasoning setting accepted by each local model


def call_ollama(cfg, system, user):
    """Reasoning is disabled when the model allows it (think=False); models that cannot disable it
    are run at the lowest level ("low"); the accepted setting is logged and stored with each answer."""
    import requests
    base = {"model": cfg["model"], "stream": False, "format": "json",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": TEMPERATURE, "num_ctx": cfg.get("num_ctx", NUM_CTX),
                        "num_predict": LOCAL_MAX_TOKENS}}
    options = [THINK_USED[cfg["model"]]] if cfg["model"] in THINK_USED else [False, "low", "none"]
    r = None
    for think in options:
        body = dict(base)
        if think != "none":
            body["think"] = think
        r = requests.post(cfg.get("host", OLLAMA_HOST) + "/api/chat", json=body, timeout=1800)
        if r.status_code == 400 and "think" in r.text.lower():
            continue
        r.raise_for_status()
        msg = r.json()["message"]
        if think is False and (msg.get("thinking") or "").strip() and cfg["model"] not in THINK_USED:
            # the model reasons even when asked not to: use its lowest reasoning level instead
            log(f"{cfg['model']}: reasoning not disabled by think=False, switching to think=low")
            continue
        if cfg["model"] not in THINK_USED:
            THINK_USED[cfg["model"]] = think
            log(f"{cfg['model']}: reasoning setting think={think}")
        return msg["content"]
    r.raise_for_status()
    raise RuntimeError("no accepted reasoning setting")


def call_mock(cfg, system, user, rng=random.Random()):
    ids = re.findall(r"\[([ABCD]\d)\]", user)
    ans = []
    for iid in ids:
        if "Forward estimate" in user.split(f"[{iid}]")[1][:40]:
            e, s, x = rng.uniform(1.2, 4.3), rng.uniform(30, 90), rng.uniform(1.5, 4)
            ans.append({"item_id": iid, "E_GPa": [e * .85, e, e * 1.15], "sigma_MPa": [s * .85, s, s * 1.15],
                        "eps_pct": [x * .8, x, x * 1.2], "confidence": rng.randint(1, 5),
                        "rationale": f"mock rationale {rng.random():.3f}"})
        else:
            fam = rng.choice(["MW", "OW", "MW", "infeasible", "none"])
            c = 0.0 if fam == "none" else (None if fam == "infeasible" else rng.uniform(3, 32))
            ans.append({"item_id": iid, "family": fam, "content_wt": c,
                        "content_low80": None if c is None else max(0, c - 5),
                        "content_high80": None if c is None else c + 5,
                        "feasibility": "infeasible" if fam == "infeasible" else "feasible",
                        "confidence": rng.randint(1, 5), "rationale": f"mock rationale {rng.random():.3f}"})
    return json.dumps({"answers": ans})


PROVIDERS = {"anthropic": call_anthropic, "gemini": call_gemini, "ollama": call_ollama, "mock": call_mock}


def respondents(condition: str, personas: list[dict]):
    if condition == "main":
        return [(p["persona_id"], p) for p in personas[:N_PERSONAS]]
    if condition == "knowledge":
        return [(p["persona_id"], p) for p in personas[:N_KNOWLEDGE]]
    return [(f"G{j + 1:03d}", None) for j in range(N_NOPERSONA)]


def cmd_panel(args):
    ensure_dirs()
    meta = load_json(OUT / "items.json")
    personas = load_json(OUT / "personas.json")
    models = get_models()
    if args.model not in models:
        sys.exit(f"Unknown model '{args.model}'. Available: {', '.join(models)} "
                 "(run select-models first for the open-weight models)")
    cfg = dict(models[args.model])
    provider = "mock" if args.mock else cfg["provider"]
    need = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}.get(provider)
    if need and not os.environ.get(need):
        sys.exit(f"Environment variable {need} is not set: export {need}=...")
    fn = PROVIDERS[provider]
    raw_path = OUT / "raw" / f"{args.model}_{args.condition}.jsonl"
    done = set()
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("status") == "ok":
                done.add((rec["respondent_id"], rec["request_id"]))
    jobs = []
    for rid, persona in respondents(args.condition, personas)[: args.limit or None]:
        for req in meta["requests"]:
            if (rid, req["request_id"]) not in done:
                jobs.append((rid, persona, req))
    log(f"{args.model}/{args.condition}: {len(done)} done, {len(jobs)} to run with {provider}")
    lock = threading.Lock()

    def work(job):
        rid, persona, req = job
        rng = random.Random(f"{SEED}-{args.model}-{args.condition}-{rid}-{req['request_id']}")
        system, user = build_prompt(req, meta, persona, args.condition, rng)
        last_err = None
        for attempt in range(3):
            for backoff in range(6):
                try:
                    text = fn(cfg, system, user)
                    break
                except Exception as e:  # network / rate limits
                    last_err = e
                    time.sleep(min(60, 2 ** backoff + random.random()))
            else:
                return {"status": "error", "error": repr(last_err), "respondent_id": rid,
                        "request_id": req["request_id"]}
            try:
                parsed = validate(extract_json(text), req)
                return {"status": "ok", "respondent_id": rid, "request_id": req["request_id"],
                        "model": args.model, "model_id": cfg["model"], "condition": args.condition,
                        "think": str(THINK_USED.get(cfg["model"], "")),
                        "attempt": attempt + 1, "answers": parsed, "raw": text,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
            except Exception as e:
                last_err = e
        return {"status": "invalid", "error": repr(last_err), "respondent_id": rid,
                "request_id": req["request_id"], "raw": text}

    n_ok = n_bad = 0
    workers = 1 if provider == "ollama" else cfg.get("workers", 4)
    with ThreadPoolExecutor(max_workers=workers) as ex, open(raw_path, "a", encoding="utf-8") as fh:
        futs = [ex.submit(work, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            n_ok += rec["status"] == "ok"
            n_bad += rec["status"] != "ok"
            if i % 25 == 0 or i == len(futs):
                log(f"  {i}/{len(futs)}  ok={n_ok}  failed={n_bad}")
    log(f"Finished {args.model}/{args.condition}. Rerun the same command to retry failed requests.")


# =============================================================================
# STEP 5 - HUMAN EXPERTS AND BLIND RATING
# =============================================================================

def cmd_humans_export(args):
    ensure_dirs()
    meta = load_json(OUT / "items.json")
    path = OUT / "humans" / "questionnaire_template.xlsx"
    with pd.ExcelWriter(path) as xw:
        intro = pd.DataFrame({"Instructions": [
            "Each sheet is a separate question set. Read the context of the sheet only.",
            "Formulations missing from a sheet's table must be treated as unknown.",
            "Inverse items: family = none / MW / OW / infeasible; content and 80% interval in wt.%.",
            "Forward items: point estimate and 80% interval for E [GPa], sigma [MPa], eps [%].",
            "Confidence 1-5. Rationale: max 50 words. Save as expert_1.xlsx, expert_2.xlsx, ...",
            "", "MATERIAL SYSTEM", *meta["materials_text"].split("\n")]})
        intro.to_excel(xw, sheet_name="README", index=False)
        cent = pd.DataFrame(meta["centroids"])
        for req in meta["requests"]:
            ctx = context_table(cent, req["hide"], False).split("\n")
            rows = [{"field": "CONTEXT", "value": c} for c in ctx] + [{"field": "", "value": ""}]
            for it in req["items"]:
                rows.append({"field": "QUESTION", "value": it["text"]})
                fields = (["family", "content_wt", "content_low80", "content_high80", "feasibility"]
                          if it["type"] == "inverse" else
                          ["E_low80", "E_point", "E_high80", "sigma_low80", "sigma_point", "sigma_high80",
                           "eps_low80", "eps_point", "eps_high80"])
                for f in fields + ["confidence", "rationale"]:
                    rows.append({"field": f"{it['item_id']}.{f}", "value": ""})
                rows.append({"field": "", "value": ""})
            pd.DataFrame(rows).to_excel(xw, sheet_name=req["request_id"], index=False)
    log(f"Human questionnaire written to {path}")


def cmd_humans_import(args):
    meta = load_json(OUT / "items.json")
    recs = []
    for f in sorted((OUT / "humans").glob("expert_*.xlsx")):
        rid = f.stem
        for req in meta["requests"]:
            sh = pd.read_excel(f, sheet_name=req["request_id"]).fillna("")
            vals = dict(zip(sh["field"].astype(str), sh["value"]))
            ans = []
            for it in req["items"]:
                g = lambda k: vals.get(f"{it['item_id']}.{k}", "")
                fl = lambda k: (float(g(k)) if str(g(k)).strip() != "" else None)
                if it["type"] == "inverse":
                    ans.append({"item_id": it["item_id"], "family": str(g("family")).strip(),
                                "content_wt": fl("content_wt"), "content_low80": fl("content_low80"),
                                "content_high80": fl("content_high80"),
                                "feasibility": str(g("feasibility")), "confidence": fl("confidence"),
                                "rationale": str(g("rationale"))})
                else:
                    ans.append({"item_id": it["item_id"],
                                "E_GPa": [fl("E_low80"), fl("E_point"), fl("E_high80")],
                                "sigma_MPa": [fl("sigma_low80"), fl("sigma_point"), fl("sigma_high80")],
                                "eps_pct": [fl("eps_low80"), fl("eps_point"), fl("eps_high80")],
                                "confidence": fl("confidence"), "rationale": str(g("rationale"))})
            try:
                parsed = validate({"answers": ans}, req)
                recs.append({"status": "ok", "respondent_id": rid, "request_id": req["request_id"],
                             "model": "human", "condition": "main", "answers": parsed})
            except Exception as e:
                log(f"  {rid} {req['request_id']}: {e}")
    with open(OUT / "raw" / "human_main.jsonl", "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"{len(recs)} human request answers imported")


def cmd_rating_export(args):
    ensure_dirs()
    ans = load_answers()
    ans = ans[(ans.condition == "main") & ans.rationale.astype(str).str.len().gt(0)]
    syn = ans[ans.model != "human"]
    per = max(1, N_RATIONALES_SYNTHETIC // max(1, syn.groupby(["model", "item_id"]).ngroups))
    pick = syn.groupby(["model", "item_id"]).sample(n=per, random_state=SEED)
    pick = pick.sample(min(N_RATIONALES_SYNTHETIC, len(pick)), random_state=SEED)
    hum = ans[ans.model == "human"]
    if len(hum):
        hum = hum.groupby("respondent_id").sample(n=N_RATIONALES_PER_HUMAN, random_state=SEED)
    sheet_rows = pd.concat([pick, hum]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    sheet_rows["rating_id"] = [f"R{i + 1:02d}" for i in range(len(sheet_rows))]
    meta = load_json(OUT / "items.json")
    item_text = {it["item_id"]: it["text"] for r in meta["requests"] for it in r["items"]}

    def summary(r):
        if r["type"] == "inverse":
            return f"family={r['family']}, content={r['content_wt']}"
        return (f"E={r.get('flexural_modulus')}, sigma={r.get('flexural_strength')}, "
                f"eps={r.get('strain_at_break')}")
    sheet = pd.DataFrame({"rating_id": sheet_rows.rating_id,
                          "question": sheet_rows.item_id.map(item_text),
                          "answer": sheet_rows.apply(summary, axis=1),
                          "rationale": sheet_rows.rationale,
                          "plausibility_1to5": "", "physically_incorrect_yes_no": "",
                          "written_by_human_or_AI": ""})
    sheet_rows[["rating_id", "model", "respondent_id", "item_id"]].to_csv(
        OUT / "rating" / "rating_key.csv", index=False)
    experts = sorted(hum.respondent_id.unique()) if len(hum) else ["expert_1", "expert_2", "expert_3"]
    for e in experts:
        own = sheet_rows.respondent_id == e
        sheet[~own.to_numpy()].to_excel(OUT / "rating" / f"rating_{e}.xlsx", index=False)
    log(f"{len(sheet)} rationales in the pool; one sheet per expert (own rationales removed) "
        f"written to {OUT / 'rating'}. Filled sheets keep the same file name.")


def cmd_rating_import(args):
    """Blind rating of rationales. Each expert rates only the file prepared for them
    (rating_expert_N.xlsx, without their own rationales)."""
    key = pd.read_csv(OUT / "rating" / "rating_key.csv")
    rows = []
    for f in sorted((OUT / "rating").glob("rating_expert_*.xlsx")):
        d = pd.read_excel(f).merge(key, on="rating_id")
        d = d[d["plausibility_1to5"].notna()].copy()
        d["rater"] = f.stem.replace("rating_", "")
        rows.append(d)
    if not rows:
        log("No rating_expert_*.xlsx found")
        return
    d = pd.concat(rows, ignore_index=True)
    d["is_human"] = d.model == "human"
    d["flag"] = d["physically_incorrect_yes_no"].astype(str).str.strip().str.lower().str.startswith("y")
    d["judged_human"] = d["written_by_human_or_AI"].astype(str).str.strip().str.lower().str.startswith("h")
    d.to_csv(OUT / "tables" / "blind_rating_long.csv", index=False)

    def boot_ci(x, n=10_000, seed=SEED):
        g = x.groupby("rating_id")["plausibility_1to5"].agg(["sum", "size"])
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(g), size=(n, len(g)))
        m = g["sum"].values[idx].sum(1) / g["size"].values[idx].sum(1)
        return np.percentile(m, [2.5, 97.5])

    out = []
    for src, x in list(d[~d.is_human].groupby("model")) + [("all synthetic", d[~d.is_human]), ("human", d[d.is_human])]:
        row = {"source": src, "ratings": len(x), "rationales": x.rating_id.nunique(),
               "mean_plausibility": x.plausibility_1to5.mean(),
               "share_flagged": x.flag.mean(), "share_judged_human": x.judged_human.mean()}
        if src in ("all synthetic", "human"):
            row["plausibility_ci_low"], row["plausibility_ci_high"] = boot_ci(x)
        out.append(row)
    out = pd.DataFrame(out)
    out.to_csv(OUT / "tables" / "blind_rating.csv", index=False)
    df_to_latex(out, OUT / "tables" / "tab_blind_rating.tex", "Blind rating of rationales", "tab:rating")

    def bal_acc(x):
        h, a = x[x.is_human], x[~x.is_human]
        return pd.Series({"human_judged_human": f"{h.judged_human.sum()}/{len(h)}",
                          "synthetic_judged_AI": f"{(~a.judged_human).sum()}/{len(a)}",
                          "balanced_accuracy": (h.judged_human.mean() + (~a.judged_human).mean()) / 2,
                          "share_judged_human": x.judged_human.mean()})
    acc = pd.concat([d.groupby("rater").apply(bal_acc, include_groups=False),
                     bal_acc(d).to_frame("all").T])
    acc.to_csv(OUT / "tables" / "blind_rating_source_accuracy.csv")

    agree = {}
    try:
        import krippendorff
        syn = d[~d.is_human]
        for col, level in [("plausibility_1to5", "ordinal"), ("flag", "nominal"), ("judged_human", "nominal")]:
            mat = syn.pivot_table(index="rater", columns="rating_id", values=col, aggfunc="first").astype(float).values
            agree[col] = float(krippendorff.alpha(reliability_data=mat, level_of_measurement=level))
    except Exception as e:  # noqa: BLE001
        log(f"Krippendorff alpha not computed: {e}")
    pd.Series(agree, name="krippendorff_alpha").to_csv(OUT / "tables" / "blind_rating_agreement.csv")
    flagged = (d[~d.is_human].groupby(["rating_id", "model", "item_id"]).flag.sum()
               .reset_index().query("flag >= 2"))
    flagged.to_csv(OUT / "tables" / "blind_rating_flagged.csv", index=False)
    log("\n" + out.round(2).to_string(index=False))
    log("\n" + acc.to_string())
    log(f"Krippendorff alpha (synthetic rationales): {agree}")


# =============================================================================
# STEP 6 - ANALYSIS
# =============================================================================

def load_answers() -> pd.DataFrame:
    rows = []
    for f in sorted((OUT / "raw").glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue
            for a in rec["answers"]:
                rows.append({"model": rec["model"], "condition": rec["condition"],
                             "respondent_id": rec["respondent_id"], "request_id": rec["request_id"], **a})
    df = pd.DataFrame(rows)
    num_cols = ["content_wt", "content_low80", "content_high80", "confidence"] + \
        [f + sfx for f in FEATURES for sfx in ("", "_low80", "_high80")]
    for c in num_cols:
        if c in df:
            df[c] = pd.to_numeric(df[c].map(to_num), errors="coerce")
    return df


def exclusion_rates() -> pd.DataFrame:
    rows = []
    for f in sorted((OUT / "raw").glob("*.jsonl")):
        recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()]
        last = {}
        for r in recs:
            last[(r["respondent_id"], r["request_id"])] = r["status"]
        n = len(last)
        ok = sum(v == "ok" for v in last.values())
        retried = sum(r.get("attempt", 1) > 1 for r in recs if r.get("status") == "ok")
        rows.append({"file": f.stem, "requests": n, "valid": ok, "excluded": n - ok,
                     "exclusion_rate": (n - ok) / n if n else np.nan, "valid_after_retry": retried})
    return pd.DataFrame(rows)


def interval_score(lo, hi, x, alpha=0.2):
    return (hi - lo) + (2 / alpha) * max(lo - x, 0) + (2 / alpha) * max(x - hi, 0)


def pooled_interval(lows, highs, n_draw=200, rng=None):
    rng = rng or np.random.default_rng(SEED)
    draws = np.concatenate([rng.uniform(l, h, n_draw) for l, h in zip(lows, highs) if h >= l])
    return float(np.percentile(draws, 10)), float(np.percentile(draws, 90))


def fleiss_kappa(table: np.ndarray) -> float:
    """table: items x categories, counts."""
    n = table.sum(1)
    table = table[n > 1]
    n = n[n > 1]
    if len(table) < 2:
        return np.nan
    p_j = table.sum(0) / table.sum()
    P_i = ((table ** 2).sum(1) - n) / (n * (n - 1))
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    return float((P_bar - P_e) / (1 - P_e)) if P_e < 1 else np.nan


def krippendorff_interval(mat: np.ndarray) -> float:
    try:
        import krippendorff
        return float(krippendorff.alpha(reliability_data=mat, level_of_measurement="interval"))
    except Exception:
        return np.nan


def source_label(model, condition):
    return model if condition == "main" else f"{model} ({condition})"


def cmd_analyze(args):
    ensure_dirs()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    truth = pd.read_csv(OUT / "ground_truth.csv", keep_default_na=False, na_values=[""])
    ml = pd.read_csv(OUT / "ml_answers.csv", keep_default_na=False, na_values=[""])
    ans = load_answers()
    if ans.empty:
        log("No answers yet.")
        return
    ans["source"] = [source_label(m, c) for m, c in zip(ans.model, ans.condition)]
    panel_main = ans[(ans.condition == "main") & (ans.model != "human")].copy()
    panel_main["source"] = "Panel (all LLMs)"
    ans_all = pd.concat([ans, panel_main])
    exclusion_rates().to_csv(OUT / "tables" / "exclusion_rates.csv", index=False)

    # ---- block A: hidden-formulation inverse items ----
    tA = truth[truth.block == "A"].set_index("item_id")
    rowsA = []
    for (src, iid), g in ans_all[ans_all.item_id.isin(tA.index)].groupby(["source", "item_id"]):
        t = tA.loc[iid]
        fam_share = g.family.value_counts(normalize=True)
        cont = g[g.family.isin(["MW", "OW"])].dropna(subset=["content_low80", "content_high80"])
        med = float(g.content_wt.dropna().median()) if g.content_wt.notna().any() else np.nan
        lo = hi = np.nan
        if len(cont):
            lo, hi = pooled_interval(cont.content_low80, cont.content_high80)
        ind_err = (g.content_wt - t.content_wt).abs()
        rowsA.append({"item_id": iid, "hidden": t.hidden, "source": src, "n": len(g),
                      "family_majority": fam_share.idxmax(), "share_true_family": fam_share.get(t.family, 0.0),
                      "content_median": med, "abs_error_median": abs(med - t.content_wt) if not np.isnan(med) else np.nan,
                      "abs_error_individual_mean": float(ind_err.mean()),
                      "pooled_low80": lo, "pooled_high80": hi,
                      "coverage_individual": float(((cont.content_low80 <= t.content_wt) &
                                                    (t.content_wt <= cont.content_high80)).mean()) if len(cont) else np.nan,
                      "interval_score_individual": float(np.mean([interval_score(a, b, t.content_wt)
                                                                  for a, b in zip(cont.content_low80, cont.content_high80)]))
                      if len(cont) else np.nan,
                      "IQR_content": float(g.content_wt.quantile(.75) - g.content_wt.quantile(.25))})
    for src in ["ML", "ML (no AD)", "Baseline"]:
        for iid, r in ml[(ml.source == src) & ml.item_id.isin(tA.index)].set_index("item_id").iterrows():
            t = tA.loc[iid]
            c = r.content_wt if pd.notna(r.content_wt) else np.nan
            rowsA.append({"item_id": iid, "hidden": t.hidden, "source": src, "n": 1,
                          "family_majority": r.family, "share_true_family": float(r.family == t.family),
                          "content_median": c, "abs_error_median": abs(c - t.content_wt) if pd.notna(c) else np.nan})
    A = pd.DataFrame(rowsA).sort_values(["item_id", "source"])
    A.to_csv(OUT / "tables" / "block_A_inverse_hidden.csv", index=False)
    sumA = (A.groupby("source").agg(family_accuracy=("share_true_family", "mean"),
                                    MAE_median=("abs_error_median", "mean"),
                                    coverage80=("coverage_individual", "mean"),
                                    interval_score=("interval_score_individual", "mean"))
            .reset_index())
    sumA.to_csv(OUT / "tables" / "block_A_summary.csv", index=False)
    df_to_latex(sumA, OUT / "tables" / "tab_block_A_summary.tex", "Hidden-formulation inverse items", "tab:blockA")

    # ---- block B: hidden-formulation forward items ----
    tB = truth[truth.block == "B"].set_index("item_id")
    rowsB = []
    fwd = ans_all[ans_all.item_id.isin(tB.index)]
    for (src, iid), g in fwd.groupby(["source", "item_id"]):
        t = tB.loc[iid]
        rec = {"item_id": iid, "hidden": t.hidden, "source": src, "n": len(g)}
        apes, cov, isc = [], [], []
        for f in FEATURES:
            med = g[f].median()
            rec[f + "_median"] = med
            apes.append(abs(med - t[f]) / t[f] * 100)
            gi = g.dropna(subset=[f + "_low80", f + "_high80"])   # answers without an interval are skipped
            if len(gi):
                cov.append(((gi[f + "_low80"] <= t[f]) & (t[f] <= gi[f + "_high80"])).mean())
                isc.append(np.mean([interval_score(a, b, t[f]) / t[f] for a, b in zip(gi[f + "_low80"], gi[f + "_high80"])]))
        rec.update({"APE_mean": float(np.mean(apes)), "coverage80": float(np.mean(cov)) if cov else np.nan,
                    "relative_interval_score": float(np.mean(isc)) if isc else np.nan})
        rowsB.append(rec)
    for iid, r in ml[(ml.source == "Baseline") & ml.item_id.isin(tB.index)].set_index("item_id").iterrows():
        t = tB.loc[iid]
        rowsB.append({"item_id": iid, "hidden": t.hidden, "source": "Baseline", "n": 1,
                      **{f + "_median": r[f] for f in FEATURES},
                      "APE_mean": float(np.mean([abs(r[f] - t[f]) / t[f] * 100 for f in FEATURES]))})
    B = pd.DataFrame(rowsB).sort_values(["item_id", "source"])
    B.to_csv(OUT / "tables" / "block_B_forward_hidden.csv", index=False)
    sumB = B.groupby("source")[["APE_mean", "coverage80", "relative_interval_score"]].mean().reset_index()
    sumB.to_csv(OUT / "tables" / "block_B_summary.csv", index=False)
    df_to_latex(sumB, OUT / "tables" / "tab_block_B_summary.tex", "Hidden-formulation forward items", "tab:blockB")

    # ---- blocks C and D: design requirements and infeasible requests ----
    tCD = truth[truth.block.isin(["C", "D"])].set_index("item_id")
    rowsCD = []
    for (src, iid), g in ans_all[ans_all.item_id.isin(tCD.index)].groupby(["source", "item_id"]):
        sh = g.family.value_counts(normalize=True)
        rowsCD.append({"item_id": iid, "source": src, "n": len(g),
                       **{f"share_{k}": float(sh.get(k, 0.0)) for k in ["none", "MW", "OW", "infeasible"]},
                       "share_flagged_infeasible_or_uncertain": float(((g.family == "infeasible") |
                                                                       g.feasibility.isin(["infeasible", "uncertain"])).mean()),
                       "family_majority": sh.idxmax(),
                       "content_median": float(g.content_wt.dropna().median()) if g.content_wt.notna().any() else np.nan,
                       "content_IQR": float(g.content_wt.quantile(.75) - g.content_wt.quantile(.25))
                       if g.content_wt.notna().any() else np.nan})
    for src in ["ML", "ML (no AD)", "Baseline"]:
        for iid, r in ml[(ml.source == src) & ml.item_id.isin(tCD.index)].set_index("item_id").iterrows():
            rowsCD.append({"item_id": iid, "source": src, "n": 1, "family_majority": r.family,
                           "share_infeasible": float(r.family == "infeasible"),
                           "share_flagged_infeasible_or_uncertain": float(r.family == "infeasible"),
                           "content_median": r.content_wt})
    CD = pd.DataFrame(rowsCD).sort_values(["item_id", "source"])
    CD.to_csv(OUT / "tables" / "block_CD_design.csv", index=False)
    # agreement with ML on C items
    mlC = ml[(ml.source == "ML") & ml.item_id.str.startswith("C")].set_index("item_id")
    agr = []
    for (src, iid), g in ans_all[ans_all.item_id.str.startswith("C")].groupby(["source", "item_id"]):
        m = mlC.loc[iid]
        agr.append({"source": src, "item_id": iid,
                    "share_same_family_as_ML": float((g.family == m.family).mean()),
                    "median_minus_ML_wt": float(g.content_wt.dropna().median() - m.content_wt)
                    if pd.notna(m.content_wt) and g.content_wt.notna().any() else np.nan})
    pd.DataFrame(agr).to_csv(OUT / "tables" / "block_C_agreement_with_ML.csv", index=False)

    # ---- agreement within the panel ----
    cats = ["none", "MW", "OW", "infeasible"]
    rel = []
    for src, g in ans_all[ans_all.item_id.str.match(r"[ACD]")].groupby("source"):
        tab = g.pivot_table(index="item_id", columns="family", values="respondent_id",
                            aggfunc="count").reindex(columns=cats).fillna(0).to_numpy()
        mat = g.pivot_table(index="respondent_id", columns="item_id", values="content_wt",
                            aggfunc="first").to_numpy(float)
        rel.append({"source": src, "fleiss_kappa_family": fleiss_kappa(tab),
                    "krippendorff_alpha_content": krippendorff_interval(mat)})
    rel = pd.DataFrame(rel)
    rel.to_csv(OUT / "tables" / "panel_agreement.csv", index=False)

    # ---- dispersion synthetic vs human ----
    if (ans.model == "human").any():
        disp = []
        h = ans[ans.model == "human"]
        for iid, gh in h[h.type == "inverse"].groupby("item_id"):
            ih = gh.content_wt.quantile(.75) - gh.content_wt.quantile(.25)
            for src, gs in panel_main[panel_main.item_id == iid].groupby("source"):
                is_ = gs.content_wt.quantile(.75) - gs.content_wt.quantile(.25)
                disp.append({"item_id": iid, "source": src, "IQR_synthetic": is_, "IQR_human": ih,
                             "ratio": is_ / ih if ih > 0 else np.nan})
        pd.DataFrame(disp).to_csv(OUT / "tables" / "dispersion_vs_humans.csv", index=False)

    # ---- saturation ----
    rng = np.random.default_rng(SEED)
    sat = []
    base = ans[(ans.condition == "main") & (ans.model != "human") & ans.item_id.str.match(r"[AC]")]
    for (mdl, iid), g in base.groupby(["model", "item_id"]):
        vals = g.content_wt.dropna().to_numpy()
        if len(vals) < 5:
            continue
        full = np.median(vals)
        for n in SATURATION_SIZES:
            if n > len(vals):
                continue
            d = [abs(np.median(rng.choice(vals, n, replace=False)) - full) for _ in range(SATURATION_REPS)]
            sat.append({"model": mdl, "item_id": iid, "n": n, "abs_diff": float(np.mean(d))})
    sat = pd.DataFrame(sat)
    sat.to_csv(OUT / "tables" / "saturation.csv", index=False)

    # ---- conditions (RQ3) ----
    cond = []
    for (mdl, c), g in ans[(ans.model != "human") & ans.item_id.isin(tA.index)].groupby(["model", "condition"]):
        err = []
        for iid, gi in g.groupby("item_id"):
            err.append(abs(gi.content_wt.dropna().median() - tA.loc[iid].content_wt))
        cond.append({"model": mdl, "condition": c, "MAE_median_blockA": float(np.nanmean(err)),
                     "family_accuracy_blockA": float(np.mean([(gi.family == tA.loc[iid].family).mean()
                                                               for iid, gi in g.groupby("item_id")])),
                     "mean_IQR_blockA": float(np.mean([gi.content_wt.quantile(.75) - gi.content_wt.quantile(.25)
                                                       for _, gi in g.groupby("item_id")]))})
    cond = pd.DataFrame(cond)
    cond.to_csv(OUT / "tables" / "conditions_RQ3.csv", index=False)
    df_to_latex(cond, OUT / "tables" / "tab_conditions.tex", "Effect of panel design", "tab:conditions")

    # ---- figures ----
    order = [s for s in ["ML", "ML (no AD)", "Baseline", "Panel (all LLMs)"] + sorted(get_models()) + ["human"]
             if s in set(A.source)]
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, s in enumerate(order):
        sub = A[A.source == s]
        ax.scatter(sub.item_id, sub.abs_error_median, label=s, s=40, marker="os^vDPXh<>*"[j % 11])
    ax.set_ylabel("Absolute error on filler content [wt.%]")
    ax.set_xlabel("Hidden-formulation inverse item")
    ax.grid(ls=":", alpha=.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_errors_blockA.pdf")
    fig.savefig(OUT / "figures" / "fig_errors_blockA.png", dpi=200)
    plt.close(fig)

    items_A = sorted(tA.index)
    fig, axes = plt.subplots(1, len(items_A), figsize=(3 * len(items_A), 3.2), sharey=True)
    for ax, iid in zip(np.atleast_1d(axes), items_A):
        g = panel_main[panel_main.item_id == iid]
        ax.hist(g.content_wt.dropna(), bins=np.arange(0, 41, 2), orientation="horizontal", alpha=.7)
        ax.axhline(tA.loc[iid].content_wt, color="k", lw=2, label="true")
        mlr = ml[(ml.source == "ML") & (ml.item_id == iid)]
        if len(mlr) and pd.notna(mlr.content_wt.iloc[0]):
            ax.axhline(mlr.content_wt.iloc[0], color="r", ls="--", label="ML")
        hr = ans[(ans.model == "human") & (ans.item_id == iid)]
        ax.scatter(np.full(len(hr), 1), hr.content_wt, color="g", marker="D", label="humans", zorder=3)
        ax.set_title(f"{iid} ({tA.loc[iid].hidden})", fontsize=9)
        ax.set_xlabel("count")
    np.atleast_1d(axes)[0].set_ylabel("Filler content [wt.%]")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_distributions_blockA.pdf")
    fig.savefig(OUT / "figures" / "fig_distributions_blockA.png", dpi=200)
    plt.close(fig)

    if not sat.empty:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for mdl, g in sat.groupby("model"):
            m = g.groupby("n").abs_diff.mean()
            ax.plot(m.index, m.values, marker="o", label=mdl)
        ax.set_xscale("log")
        ax.set_xlabel("Panel size (personas)")
        ax.set_ylabel("|median(subsample) - median(full)| [wt.%]")
        ax.grid(ls=":", alpha=.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "figures" / "fig_saturation.pdf")
        fig.savefig(OUT / "figures" / "fig_saturation.png", dpi=200)
        plt.close(fig)

    log("Block A summary:\n" + sumA.round(2).to_string(index=False))
    log("Block B summary:\n" + sumB.round(2).to_string(index=False))
    log("Conditions:\n" + cond.round(2).to_string(index=False))
    log(f"Tables in {OUT / 'tables'}, figures in {OUT / 'figures'}")


# =============================================================================
# STEP 7 - SELECTION OF THE OPEN-WEIGHT MODELS (Hugging Face Hub)
# =============================================================================

HF = "https://huggingface.co"


def hf_get(url, params=None):
    import requests
    h = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
    for backoff in range(5):
        r = requests.get(url, params=params, headers=h, timeout=60)
        if r.status_code == 429:
            time.sleep(2 ** backoff * 5)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def _base_model_tags(tags):
    out = []
    for t in tags or []:
        if t.startswith("base_model:"):
            parts = t.split(":")
            kind, repo = (parts[1], ":".join(parts[2:])) if len(parts) > 2 else ("", parts[1])
            out.append((kind, repo))
    return out


def _context_length(repo):
    try:
        c = hf_get(f"{HF}/{repo}/resolve/main/config.json")
    except Exception:
        return None
    for d in (c, c.get("text_config") or {}, c.get("llm_config") or {}):
        for k in ("max_position_embeddings", "max_sequence_length", "seq_length", "model_max_length"):
            if isinstance(d.get(k), int):
                return d[k]
    return None


def has_quant(filename, quant):
    """True for a standard llama.cpp quantization file (e.g. model-Q4_K_M.gguf, also split files);
    variants such as Unsloth Dynamic (UD-Q4_K_M) or imatrix-free renamings with other prefixes are
    different quantizations and do not match."""
    name = filename.split("/")[-1]
    pattern = rf"(?<![A-Za-z0-9])(?<!UD-){re.escape(quant)}(-\d{{5}}-of-\d{{5}})?\.gguf$"
    return re.search(pattern, name, flags=re.I) is not None


def _norm(x):
    return re.sub(r"[^a-z0-9]", "", x.lower())


def is_plain_conversion(repo, gguf_model):
    """A GGUF repository is accepted only as a plain conversion of `repo`: its name must be the original
    model name, optionally with the developer prefix and the GGUF/imatrix markers, and it must not
    declare a fine-tuned or merged base model. This excludes
    modified models (e.g. 'uncensored' or 'abliterated' versions) tagged as quantizations."""
    name = _norm(gguf_model["id"].split("/")[1])
    orig = _norm(repo.split("/")[1])
    if orig not in name:
        return False
    rest = name.replace(orig, "", 1)
    for token in ("gguf", "imatrix", "i1", _norm(repo.split("/")[0])):
        rest = rest.replace(token, "")
    if rest:                      # extra words: a modified model, not a plain conversion
        return False
    kinds = {k for k, _ in _base_model_tags(gguf_model.get("tags"))}
    return not (kinds & {"finetune", "merge", "adapter"})


def resolve_gguf(repo, quant):
    """Most downloaded GGUF conversion of `repo` that contains the requested quantization;
    a conversion published by the same organisation is preferred."""
    lst = hf_get(f"{HF}/api/models", params=[("filter", f"base_model:quantized:{repo}"), ("filter", "gguf"),
                                                ("sort", "downloads"), ("direction", "-1"), ("limit", "30")])
    org = repo.split("/")[0].lower()
    lst = sorted(lst, key=lambda m: (m["id"].split("/")[0].lower() != org, -(m.get("downloads") or 0)))
    for m in lst:
        if not is_plain_conversion(repo, m):
            continue
        try:
            info = hf_get(f"{HF}/api/models/{m['id']}")
        except Exception:
            continue
        files = [s_["rfilename"] for s_ in info.get("siblings", [])]
        if any(has_quant(f, quant) for f in files):
            return m["id"]
    return None


_ORG_CACHE: dict[str, bool] = {}


def is_organisation(name):
    """True if the namespace is an organisation account on the Hub (personal accounts return 404)."""
    if name not in _ORG_CACHE:
        try:
            hf_get(f"{HF}/api/organizations/{name}/overview")
            _ORG_CACHE[name] = True
        except Exception:
            _ORG_CACHE[name] = False
    return _ORG_CACHE[name]


def _key(repo):
    return re.sub(r"[^a-z0-9.\-]", "-", repo.split("/")[1].lower())


def screen_general():
    now = pd.Timestamp.now(tz="UTC")
    pool, seen = [], set()
    for pipe in SELECTION["pipelines"]:
        lst = hf_get(f"{HF}/api/models", params={"pipeline_tag": pipe, "sort": "downloads", "direction": -1,
                                                  "limit": SELECTION["pool_size"]})
        for m in lst:
            if m["id"] not in seen:
                seen.add(m["id"])
                pool.append({**m, "pipeline": pipe})
    rows = []
    for m in pool:
        repo, tags = m["id"], m.get("tags") or []
        org = repo.split("/")[0]
        row = {"repo": repo, "organisation": org, "pipeline": m["pipeline"], "downloads_30d": m.get("downloads"),
               "created": (m.get("createdAt") or "")[:10], "license": "", "params_B": np.nan, "context": None,
               "gguf": "", "eligible": False, "excluded_by": ""}
        lic = [t.split(":", 1)[1] for t in tags if t.startswith("license:")]
        row["license"] = lic[0] if lic else ""
        bm = _base_model_tags(tags)
        derivative = any(k in ("quantized", "adapter", "merge") for k, _ in bm) or \
            any(r.split("/")[0].lower() != org.lower() for _, r in bm) or \
            any(t in tags for t in ("gguf", "mlx", "onnx", "awq", "gptq"))
        age_m = (now - pd.Timestamp(m["createdAt"])).days / 30.4 if m.get("createdAt") else np.inf
        checks = [
            ("IC1 license", bool(lic)),
            ("IC2 original release", not derivative),
            ("IC2 chat model", "conversational" in tags),
            ("IC6 recent", age_m <= SELECTION["max_age_months"]),
        ]
        failed = [c for c, ok in checks if not ok]
        if not failed and not is_organisation(org):
            failed.append("IC2 organisation account")
        if not failed:
            try:
                info = hf_get(f"{HF}/api/models/{repo}", params=[("expand[]", "safetensors")])
                total = (info.get("safetensors") or {}).get("total")
            except Exception:
                total = None
            row["params_B"] = total / 1e9 if total else np.nan
            if not total or not (SELECTION["min_params_b"] <= total / 1e9 <= SELECTION["max_params_b"]):
                failed.append("IC3 size")
        if not failed:
            row["context"] = _context_length(repo)
            if row["context"] is not None and row["context"] < SELECTION["min_context"]:
                failed.append("IC3 context")
        if not failed:
            g = resolve_gguf(repo, SELECTION["quant"])
            row["gguf"] = g or ""
            if not g:
                failed.append("IC7 GGUF")
        row["excluded_by"] = "; ".join(failed)
        row["eligible"] = not failed
        rows.append(row)
    return pd.DataFrame(rows).sort_values("downloads_30d", ascending=False)


def cmd_select_models(args):
    ensure_dirs()
    if args.mock:
        sel = [{"key": k, "repo": f"mock/{k}", "stratum": s_, "gguf": f"mock/{k}-GGUF"}
               for k, s_ in [("domain-model", "domain"), ("general-a", "general"), ("general-b", "general"),
                             ("general-c", "general"), ("general-d", "general")]]
    else:
        log("Screening the Hugging Face Hub (this takes a few minutes)...")
        scr = screen_general()
        scr.to_csv(OUT / "tables" / "model_screening.csv", index=False)
        dom_gguf = resolve_gguf(DOMAIN_MODEL, SELECTION["quant"])
        if not dom_gguf:
            sys.exit(f"No {SELECTION['quant']} GGUF found for {DOMAIN_MODEL}")
        sel = [{"key": _key(DOMAIN_MODEL), "repo": DOMAIN_MODEL, "stratum": "domain", "gguf": dom_gguf}]
        used = {DOMAIN_MODEL.split("/")[0].lower()}
        for _, r in scr[scr.eligible].iterrows():
            if r.organisation.lower() in used:
                continue
            sel.append({"key": _key(r.repo), "repo": r.repo, "stratum": "general", "gguf": r.gguf,
                        "downloads_30d": int(r.downloads_30d), "params_B": round(float(r.params_B), 1),
                        "created": r.created, "license": r.license})
            used.add(r.organisation.lower())
            if len(sel) == 1 + SELECTION["n_general"]:
                break
        log(f"{int(scr.eligible.sum())} eligible of {len(scr)} screened repositories")
    for m in sel:
        m["ollama"] = f"hf.co/{m['gguf']}:{SELECTION['quant']}"
    pd.DataFrame(sel).to_csv(OUT / "tables" / "model_selection.csv", index=False)
    save_json({"screening_date": time.strftime("%Y-%m-%d"), "rule": SELECTION, "models": sel},
              OUT / "models.json")
    log("Selected models:\n" + pd.DataFrame(sel)[["key", "stratum", "repo", "ollama"]].to_string(index=False))
    if getattr(args, "pull", False) and not args.mock:
        import requests
        for m in sel:
            log(f"ollama pull {m['ollama']} ...")
            r = requests.post(OLLAMA_HOST + "/api/pull", json={"model": m["ollama"], "stream": False}, timeout=None)
            r.raise_for_status()
        log("All models downloaded.")


def get_models() -> dict:
    """Selected open-weight models (from models.json) plus the API connectors kept for future work."""
    models = {}
    path = OUT / "models.json"
    if path.exists():
        for m in load_json(path)["models"]:
            models[m["key"]] = {"provider": "ollama", "model": m["ollama"], "repo": m["repo"],
                                "stratum": m["stratum"], "workers": 1, "num_ctx": NUM_CTX, "host": OLLAMA_HOST}
    for k, v in API_MODELS.items():
        models.setdefault(k, v)
    return models


def open_models() -> list[str]:
    return [k for k, v in get_models().items() if v["provider"] == "ollama"]


# =============================================================================
# MAIN
# =============================================================================

def cmd_all(args):
    """Steps whose output already exists are not repeated, so that an interrupted run can be resumed
    with the same personas, questionnaire and models."""
    cmd_ml(args)
    if not (OUT / "personas.json").exists():
        cmd_personas(args)
    else:
        log("personas.json found: personas are not rebuilt")
    if not (OUT / "items.json").exists():
        cmd_items(args)
    else:
        log("items.json found: questionnaire is not rebuilt")
    if not (OUT / "models.json").exists():
        a = argparse.Namespace(**vars(args))
        a.pull = True
        cmd_select_models(a)
    for model in open_models():
        for cond in CONDITIONS:
            a = argparse.Namespace(**vars(args))
            a.model, a.condition = model, cond
            cmd_panel(a)
    cmd_analyze(args)


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT), help="output folder")
    ap.add_argument("--dataset", default=None, help="optional path to dataset_additives.xlsx")
    ap.add_argument("--mock", action="store_true", help="dry run with fake answers, no API calls")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["ml", "personas", "items", "humans-export", "humans-import", "rating-export",
                 "rating-import", "analyze", "all"]:
        sub.add_parser(name)
    sm = sub.add_parser("select-models")
    sm.add_argument("--pull", action="store_true", help="download the selected models into Ollama")
    p = sub.add_parser("panel")
    p.add_argument("--model", required=True, help="a key of models.json, or claude/gemini (future work)")
    p.add_argument("--condition", choices=CONDITIONS, default="main")
    p.add_argument("--limit", type=int, default=0, help="only the first N respondents (for tests)")
    args = ap.parse_args()
    if not hasattr(args, "limit"):
        args.limit = 0
    OUT = Path(args.out)
    {"ml": cmd_ml, "personas": cmd_personas, "items": cmd_items, "panel": cmd_panel,
     "humans-export": cmd_humans_export, "humans-import": cmd_humans_import,
     "rating-export": cmd_rating_export, "rating-import": cmd_rating_import,
     "analyze": cmd_analyze, "select-models": cmd_select_models, "all": cmd_all}[args.cmd](args)


if __name__ == "__main__":
    main()
