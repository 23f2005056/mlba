"""
FraudLens - P3 model pipeline (supervised ranking + anomaly ranking)

Run:
    python src/p3_models.py --data stub
    python src/p3_models.py --data real
    python src/p3_models.py --data real --data-dir "files from team"

Approaches (all scored on every eligible provider-year that has CMS features):
    supervised_lgbm    - LightGBM on Scenario A labels (y_A_h2), config frozen
    supervised_logreg  - Logistic Regression on the same labels and training rows
    anomaly_iforest    - Isolation Forest, no labels used

Outputs (out/ for real data, out/stub/ for stub):
    predictions.parquet    NPI, year, approach, score (higher = higher audit priority)
    supervised_model.pkl   {"supervised_lgbm": bundle, "supervised_logreg": bundle}
    anomaly_model.pkl      Isolation Forest bundle
Reports (reports/ for real data): coverage, splits, val metrics, feature importance,
logistic coefficients.
"""

# argparse is a Python library used to handle command-line arguments
import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
# as data is stored in Parquet files, and we need to load and process large datasets efficiently
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.p3_utils import signed_log   # own module so saved .pkl files can be loaded

REPO=Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Feature list: the 30 numeric columns from docks/FEATURE_DICTIONARY.md
# (34 columns minus npi, year, specialty, state).
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    # operational / volume / financial
    "medicare_participating", "tot_benes", "tot_srvcs",
    "submitted_charges", "allowed_amount", "medicare_payment",
    # ratios
    "pymt_per_bene", "srvcs_per_bene", "pymt_per_srvc", "submitted_per_bene",
    "charge_to_allowed", "allowed_to_pymt", "pymt_to_charge",
    # log-scaled volume
    "log_medicare_payment", "log_tot_benes", "log_tot_srvcs",
    # specialty peer robust z-scores
    "pymt_per_bene_peer_z", "srvcs_per_bene_peer_z", "charge_to_allowed_peer_z",
    # year-over-year (uses t-1 only)
    "medicare_payment_yoy_diff", "medicare_payment_yoy_growth",
    "tot_benes_yoy_diff", "tot_benes_yoy_growth",
    "tot_srvcs_yoy_diff", "tot_srvcs_yoy_growth",
    "submitted_charges_yoy_diff", "submitted_charges_yoy_growth",
    "pymt_per_bene_yoy_diff", "pymt_per_bene_yoy_growth",
    "is_new_billing_provider",
]

# Columns that must NEVER be model inputs (leak the label or are metadata).
BANNED_EXACT = {"npi", "year", "specialty", "state", "eligible", "y", "censored"}
BANNED_PREFIXES = ("y_", "censored_", "excluded_", "label_")

# Stub uses different names -> map them onto the real names.
STUB_RENAME = {
    "NPI": "npi",
    "label_scenario_a": "y",
    "total_payment": "medicare_payment",
    "beneficiaries": "tot_benes",
    "services": "tot_srvcs",
    "payment_per_beneficiary": "pymt_per_bene",
    "services_per_beneficiary": "srvcs_per_bene",
}

# ---------------------------------------------------------------------------
# Decision (prediction) years. Validation is for tuning only; test is reported.
# ---------------------------------------------------------------------------
DECISION_YEARS = {
    "real": {2: {"val": [2020], "test": [2022, 2023]},   # per split_plan.csv
             3: {"val": [2020], "test": [2022]}},
    "stub": {"val": [2020], "test": [2021, 2022]},   # stub uses h=3
}

SEED = 42
LABEL="y_A_h2" # primary label; set by --label for sensitivity runs
N_NEG_TRAIN = 100_000      # negatives sampled per training fit (training only)
N_NEG_QUICK = 100_000        # --quick mode, for smoke tests
EARLY_STOP = 100             # rounds without val PR-AUC improvement

# This dictionary contains the base hyperparameters for LightGBM.
# FROZEN after 4 runs judged on 2020 (reports/p3_val_metrics_v1..v4).
# Pre-set rule: keep sqrt-weight/AUC-stop only if 2020 PR-AUC >= 0.0010; v3 got 0.00096,
# so we fell back to v1 settings (full weight, PR-AUC stop) with 100k negatives (v4: 0.00118).
# Do not tune further - 2022/2023 (P4) are the honest estimates.

LGB_BASE = dict( 
    objective="binary", 
    metric="average_precision",
    learning_rate=0.05, 
    n_estimators=2000,
    subsample=0.8, 
    subsample_freq=1, 
    colsample_bytree=0.8, # This controls feature sampling. Each tree uses approximately 80% of the available features.
    reg_lambda=5.0, # This is L2 regularization.
    random_state=SEED, 
    n_jobs=-1, 
    verbose=-1,
    deterministic=True,
    force_row_wise=True,
)
# Small grid on purpose: only ~64 validation positives.
LGB_GRID = [dict(num_leaves=nl, min_child_samples=mcs)
            for nl in (7, 15, 31) for mcs in (50, 200)]

# Stub has ~20 rows, so leaves must be allowed to hold 1 row.
STUB_LGB_GRID = [dict(num_leaves=3, min_child_samples=1)]

def check_no_leakage(cols):
    '''It ensures that ML model's feature list does not accidentally contain columns
      that could reveal the target variable or information that shouldn't be used for prediction.    
    '''
    bad=[c for c in cols if c in BANNED_EXACT or c.startswith(BANNED_PREFIXES)]

    # An assert checks whether a condition is true. If it's false, Python raises an AssertionError.
    assert not bad, f'Banned columns in feature list: {bad}'

def _cast(table,col,typ):
    '''Changes the data type of a column in a PyArrow Table'''
    # table:	The PyArrow Table containing your data
    # col:	    The name of the column whose type you want to change
    # typ:	    The target data type
    i=table.schema.get_field_index(col) # The variable i is needed to tell PyArrow where to place the converted column in the table.
    return table.set_column(i,col,pc.cast(table[col],typ)) # pc.cast() is the operation that actually performs the data type conversion.

# ---------------------------------------------------------------------------
# Loaders. Both return (df, feature_cols, horizon) where df has columns:
#   npi (str), year (int), y (0/1), censored (bool), has_features (bool),
#   + feature columns (float32, NaN where missing)
# and contains ONLY eligible provider-years.
# ---------------------------------------------------------------------------

def load_real(data_dir:Path,label="y_A_h2"):
    '''
    main data-loading function for real FraudLens dataset. It loads labels and features from Parquet files, 
    filters the data, performs validation, optimizes data types, joins the datasets, and finally converts everything 
    into a Pandas DataFrame for downstream ML.
    '''

    print(f'--> Loading real data from {data_dir}')

    # Labels: keep eligible rows only. Primary target y_A_h2.
    lab=pq.read_table(data_dir / "labels.parquet", 
                      columns=["npi", "year", "eligible", label, f"censored_h{label[-1]}"]) # only load the five columns you need instead of loading the entire file.
    
    lab = lab.filter(
        pc.equal(
            lab["eligible"], True) # Creates a Boolean mask indicating which rows have eligible == True.
            ).drop(["eligible"]) # After filtering, the eligible column is no longer necessary, so you remove it.
    lab = _cast(lab, "year", pa.int32())

    # Rename the columns
    lab = lab.rename_columns(["npi", "year", "y", "censored"])
    print(f"eligible label rows: {lab.num_rows:,}")

    # Cross-check: candidate_pool should be exactly the eligible keys.
    n_cand = pq.ParquetFile(data_dir / "candidate_pool.parquet").metadata.num_rows # retrieving the number of rows in candidate_pool.parquet.
    print(f"candidate_pool rows: {n_cand:,}")
    assert n_cand == lab.num_rows, "candidate_pool and eligible labels disagree - ask P1"

    # Features: only the model columns, cast to float32 to save RAM.
    feat = pq.read_table(data_dir / "provider_features.parquet",
                         columns=["npi", "year"] + FEATURE_COLS # FEATURE_COLS contains your 30 numerical model features.
                        )
    feat = _cast(feat, "year", pa.int32())

    # Iterates over all 30 features and converts each one to float32.
    for c in FEATURE_COLS:
        feat = _cast(feat, c, pa.float32())
    feat = feat.append_column("has_features", pa.array([True] * feat.num_rows))
    print(f"    feature rows: {feat.num_rows:,}")

    # Left join: keep every eligible row, features are null if CMS row missing.
    joined = lab.join(feat, 
                      keys=["npi", "year"], 
                      join_type="left outer")
    df = joined.to_pandas()
    # pyarrow join output order is not fixed -> sort so seeded sampling is reproducible
    df = df.sort_values(["npi", "year"], kind="stable").reset_index(drop=True)
    df["has_features"] = df["has_features"].eq(True) # Remember that some providers had no matching feature row.
    return df, FEATURE_COLS, int(label[-1])

def load_stub(stub_dir: Path):
    '''Loads synthetic/test data into a format compatible with your real-data pipeline.'''

    print(f"--> Loading STUB data from {stub_dir} (synthetic - never report results)")
    feat = pd.read_parquet(stub_dir / "provider_features.parquet").rename(columns=STUB_RENAME)
    cand = pd.read_parquet(stub_dir / "candidate_pool.parquet").rename(columns=STUB_RENAME)
    lab = pd.read_parquet(stub_dir / "labels.parquet").rename(columns=STUB_RENAME)

    # merging the candidate pool with the labels.
    df = cand.merge(
        lab[["npi", "year", "y"]], 
        on=["npi", "year"], 
        validate="one_to_one")

    # Identify available features
    feat_cols = [c for c in FEATURE_COLS if c in feat.columns]
    # Create a feature-only DataFrame
    f = feat[["npi", "year"] + feat_cols].copy()
    f[feat_cols] = f[feat_cols].astype("float32")

    f["has_features"] = True
    df = df.merge(
        f, 
        on=["npi", "year"], 
        how="left", validate="one_to_one")
    df["has_features"] = df["has_features"].eq(True) # this gives the same result as df["has_features"] = df["has_features"].fillna(False).astype(bool)
    # (missing becomes False) without relying on behaviour pandas is changing.

    # stub dataset doesn't contain a censoring column.
    # But real-data loader returns: censored
    df["censored"] = False   # stub has no censoring column
    return df, feat_cols, 3  # stub labels use a 3-year window

# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------
def coverage_report(df, out_csv: Path):
    '''Analyzes how many eligible providers have features available, broken down by year.'''

    # This groups the dataset by observation year.
    g = df.groupby("year").agg(
        eligible_rows=("y", "size"),
        with_features=("has_features", "sum"),
        positives=("y", "sum"),
    )

    g["without_features"] = g["eligible_rows"] - g["with_features"]
    g["positives_without_features"] = df[~df["has_features"]].groupby("year")["y"].sum()
    g = g.fillna(0).astype(int).reset_index()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Creates the output directory if it doesn't already exist.
    g.to_csv(out_csv, index=False)
    print("\n=== SCORING COVERAGE (eligible rows) ===")
    print(g.to_string(index=False))
    print(f"saved {out_csv}")
    return g

# ---------------------------------------------------------------------------
# Maturity-aware splits
# ---------------------------------------------------------------------------
def allowed_train_years(T, h, years):
    """A year-T' row may train a model predicting year T only if T' + h <= T."""
    return [y for y in sorted(years) if y + h <= T]


def check_split_plan(split_plan_csv: Path, h, years, decision_years):
    """Our rule must reproduce P1's frozen split_plan.csv exactly."""
    plan = pd.read_csv(split_plan_csv) # read the split plan
    plan = plan[plan["horizon"] == h]
    # This selects only rows where the prediction horizon is 2 years.
    # Why?
    # Because a two-year prediction task and a three-year prediction task have different label-maturity requirements and 
    # therefore potentially different training years.


    # itertuples() iterates over the DataFrame rows, 
    # returning each row as an object whose columns can be accessed using dot notation.
    for r in plan.itertuples():
        expected = [int(x) for x in str(r.allowed_train_years).split()]
        ours = allowed_train_years(r.prediction_year, h, years)
        assert ours == expected, f"split mismatch for {r.prediction_year}: {ours} vs {expected}"

    planned = {r.role: [] for r in plan.itertuples()}
    for r in plan.itertuples():
        planned[r.role].append(int(r.prediction_year))

    # This verifies that the prediction years planned to be evaluated match P1's split plan exactly.
    assert planned == decision_years, f"decision years {decision_years} != split_plan {planned}" 
    print(f"split_plan.csv check passed for h={h}")


def make_split(df, T, h):
    """Return (train_index, score_index) for prediction year T."""
    years = df["year"].unique()
    train_years = allowed_train_years(T, h, years)
    assert train_years, f"no mature training years for T={T}"

    train_mask = df["year"].isin(train_years) & df["has_features"] & ~df["censored"]
    score_mask = (df["year"] == T) & df["has_features"]

    # Leakage guards
    assert df.loc[train_mask, "year"].max() + h <= T, "immature label used in training"
    assert not df.loc[df["year"].isin(train_years), "censored"].any(), \
        "censored rows inside allowed training years - check labels"
    assert not (train_mask & score_mask).any()

    return df.index[train_mask], df.index[score_mask], train_years


def split_summary(df, h, decision_years, out_csv: Path):
    rows = []
    for role, Ts in decision_years.items():
        for T in Ts:
            tr, sc, ty = make_split(df, T, h)
            rows.append({
                "role": role, "prediction_year": T,
                "train_years": f"{ty[0]}-{ty[-1]}",
                "train_rows": len(tr),
                "train_positives": int(df.loc[tr, "y"].sum()),
                "train_positive_npis": df.loc[tr][df.loc[tr, "y"] == 1]["npi"].nunique(),
                "score_rows": len(sc),
                "score_positives": int(df.loc[sc, "y"].sum()),
            })
    s = pd.DataFrame(rows)
    s.to_csv(out_csv, index=False)
    print("\n=== SPLITS (maturity rule T' + h <= T) ===")
    print(s.to_string(index=False))
    print(f"saved {out_csv}")
    return s

# ---------------------------------------------------------------------------
# Shared helpers for all models
# ---------------------------------------------------------------------------
def X_of(df, idx, feat_cols):
    return df.loc[idx, feat_cols].to_numpy(dtype=np.float32)


def sample_train(df, tr_idx, n_neg, seed):
    """All positives + at most n_neg random negatives. Training only."""
    y = df.loc[tr_idx, "y"].to_numpy()
    pos, neg = tr_idx[y == 1], tr_idx[y == 0]
    rng = np.random.default_rng(seed)
    if len(neg) > n_neg:
        neg = rng.choice(neg, n_neg, replace=False)
    idx = np.concatenate([np.asarray(pos), np.asarray(neg)])
    rng.shuffle(idx)
    return pd.Index(idx)


def rank_metrics(y, s):
    """Model-selection metrics only. Official evaluation + CIs belong to P4."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    out = {"n": len(y), "positives": int(y.sum()), "base_rate": y.mean()}
    if 0 < y.sum() < len(y):
        out["pr_auc"] = average_precision_score(y, s)
        out["roc_auc"] = roc_auc_score(y, s)
    else:
        out["pr_auc"] = out["roc_auc"] = float("nan")
    order = np.argsort(-s, kind="stable")
    for q in (0.01, 0.05):
        k = max(1, int(np.ceil(q * len(s))))
        out[f"recall@{q:.0%}"] = y[order[:k]].sum() / max(y.sum(), 1)
    return out


def pred_frame(df, idx, approach, score):
    return pd.DataFrame({
        "NPI": df.loc[idx, "npi"].to_numpy(),
        "year": df.loc[idx, "year"].to_numpy(),
        "approach": approach,
        "score": np.asarray(score, dtype="float64"),
    })


# ---------------------------------------------------------------------------
# Approach A1: LightGBM
# ---------------------------------------------------------------------------
def run_lightgbm(df, feat_cols, h, decision_years, n_neg, grid, report_dir):
    print("\n=== APPROACH A1: LightGBM ===")

    # 1) Tune on the validation year
    T_val = decision_years["val"][0]
    tr, sc, ty = make_split(df, T_val, h)
    idx = sample_train(df, tr, n_neg, SEED)
    Xtr, ytr = X_of(df, idx, feat_cols), df.loc[idx, "y"].to_numpy()
    Xv, yv = X_of(df, sc, feat_cols), df.loc[sc, "y"].to_numpy()
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    print(f"tuning: train {ty[0]}-{ty[-1]} -> predict {T_val} | "
          f"{len(idx):,} rows ({int(ytr.sum())} pos) | scale_pos_weight={spw:.0f}")

    val_rows, best = [], None
    for params in grid:
        m = lgb.LGBMClassifier(**LGB_BASE, **params, scale_pos_weight=spw)
        m.fit(Xtr, ytr, eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(EARLY_STOP, first_metric_only=True, verbose=False)])
        n_iter = int(m.best_iteration_ or LGB_BASE["n_estimators"])
        s = m.predict_proba(Xv)[:, 1]
        met = rank_metrics(yv, s)
        val_rows.append({"model": "supervised_lgbm", "year": T_val,
                         "params": str(params), "n_estimators": n_iter, **met})
        print(f"  {params} trees={n_iter:4d}  PR-AUC={met['pr_auc']:.5f}  "
              f"ROC={met['roc_auc']:.3f}  recall@1%={met['recall@1%']:.3f}  "
              f"recall@5%={met['recall@5%']:.3f}")
        if best is None or met["pr_auc"] > best["pr_auc"]:
            best = dict(pr_auc=met["pr_auc"], params=params, n_iter=n_iter, model=m, score=s)

    print(f"chosen: {best['params']} with {best['n_iter']} trees "
          f"(val base rate {yv.mean():.2e}, random recall@1% = 0.010)")
    models = {T_val: best["model"]}
    preds = [pred_frame(df, sc, "supervised_lgbm", best["score"])]

    # 2) Refit for each test year on its own allowed years, fixed settings
    final_params = {**LGB_BASE, "n_estimators": best["n_iter"], **best["params"]}
    imp = np.zeros(len(feat_cols))
    for T in decision_years["test"]:
        tr, sc, ty = make_split(df, T, h)
        idx = sample_train(df, tr, n_neg, SEED + T)
        Xtr, ytr = X_of(df, idx, feat_cols), df.loc[idx, "y"].to_numpy()
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        m = lgb.LGBMClassifier(**final_params, scale_pos_weight=spw)
        m.fit(Xtr, ytr)
        s = m.predict_proba(X_of(df, sc, feat_cols))[:, 1]
        models[T] = m
        preds.append(pred_frame(df, sc, "supervised_lgbm", s))
        g = m.booster_.feature_importance("gain")
        imp += g / max(g.sum(), 1e-12)
        print(f"  test {T}: trained on {ty[0]}-{ty[-1]} ({len(idx):,} rows, "
              f"{int(ytr.sum())} pos) -> scored {len(sc):,} rows")

    fi = (pd.DataFrame({"feature": feat_cols,
                        "share_of_gain": imp / max(imp.sum(), 1e-12)})
          .sort_values("share_of_gain", ascending=False))
    fi.to_csv(report_dir / "p3_feature_importance.csv", index=False)
    print("\nTop 10 features (mean share of gain over test-year models):")
    print(fi.head(10).to_string(index=False))

    bundle = {"approach": "supervised_lgbm", "models": models, "features": feat_cols,
              "params": final_params, "horizon": h, "label": f"{LABEL} (labels v1)"}
    return bundle, preds, val_rows


# ---------------------------------------------------------------------------
# Shared preprocessing for the linear and anomaly models.
# LightGBM handles NaN and skew itself; these two models need help.
# ---------------------------------------------------------------------------
def make_preprocessor():
    """signed log -> median impute -> standardise. Fitted on training rows only."""
    return Pipeline([
        ("log", FunctionTransformer(signed_log)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])


# ---------------------------------------------------------------------------
# Approach A2: Logistic Regression (the plan's primary supervised model)
# ---------------------------------------------------------------------------
LR_GRID = [dict(C=c) for c in (0.01, 0.1, 1.0)]   # pre-set, small: ~64 val positives


def run_logreg(df, feat_cols, h, decision_years, n_neg, report_dir):
    print("\n=== APPROACH A2: Logistic Regression ===")

    def fit(Xtr, ytr, C):
        m = Pipeline([("prep", make_preprocessor()),
                      ("lr", LogisticRegression(C=C, class_weight="balanced",
                                                max_iter=2000, random_state=SEED))])
        return m.fit(Xtr, ytr)

    # 1) Tune C on the validation year (same split + negative sample as LightGBM)
    T_val = decision_years["val"][0]
    tr, sc, ty = make_split(df, T_val, h)
    idx = sample_train(df, tr, n_neg, SEED)
    Xtr, ytr = X_of(df, idx, feat_cols), df.loc[idx, "y"].to_numpy()
    Xv, yv = X_of(df, sc, feat_cols), df.loc[sc, "y"].to_numpy()

    val_rows, best = [], None
    for params in LR_GRID:
        m = fit(Xtr, ytr, **params)
        s = m.predict_proba(Xv)[:, 1]
        met = rank_metrics(yv, s)
        val_rows.append({"model": "supervised_logreg", "year": T_val,
                         "params": str(params), "n_estimators": np.nan, **met})
        print(f"  {params}  PR-AUC={met['pr_auc']:.5f}  ROC={met['roc_auc']:.3f}  "
              f"recall@1%={met['recall@1%']:.3f}  recall@5%={met['recall@5%']:.3f}")
        if best is None or met["pr_auc"] > best["pr_auc"]:
            best = dict(pr_auc=met["pr_auc"], params=params, model=m, score=s)
    print(f"chosen: {best['params']}")
    models = {T_val: best["model"]}
    preds = [pred_frame(df, sc, "supervised_logreg", best["score"])]

    # 2) Refit for each test year on its own allowed years, fixed C
    coefs = []
    for T in decision_years["test"]:
        tr, sc, ty = make_split(df, T, h)
        idx = sample_train(df, tr, n_neg, SEED + T)
        m = fit(X_of(df, idx, feat_cols), df.loc[idx, "y"].to_numpy(), **best["params"])
        s = m.predict_proba(X_of(df, sc, feat_cols))[:, 1]
        models[T] = m
        preds.append(pred_frame(df, sc, "supervised_logreg", s))
        coefs.append(m.named_steps["lr"].coef_[0])
        print(f"  test {T}: trained on {ty[0]}-{ty[-1]} ({len(idx):,} rows) "
              f"-> scored {len(sc):,} rows")

    # Standardised coefficients: + raises risk, - lowers it (per 1 SD of signed-log feature)
    c = np.mean(coefs, axis=0)
    co = (pd.DataFrame({"feature": feat_cols, "std_coef": c, "abs_coef": np.abs(c)})
          .sort_values("abs_coef", ascending=False).drop(columns="abs_coef"))
    co.to_csv(report_dir / "p3_logreg_coefficients.csv", index=False)
    print("\nTop 10 coefficients (mean over test-year models, standardised):")
    print(co.head(10).to_string(index=False))

    bundle = {"approach": "supervised_logreg", "models": models, "features": feat_cols,
              "params": best["params"], "horizon": h, "label": f"{LABEL} (labels v1)"}
    return bundle, preds, val_rows


# ---------------------------------------------------------------------------
# Approach B: Isolation Forest (no labels used anywhere - not for fitting, not for tuning)
# For each decision year T: fit on a sample of year-T provider features (known at the
# end of T), then score every eligible year-T row. Settings are fixed, not tuned.
# ---------------------------------------------------------------------------
IF_FIT_ROWS = 200_000
IF_PARAMS = dict(n_estimators=300, max_samples=256, random_state=SEED, n_jobs=-1)


def run_iforest(df, feat_cols, decision_years):
    print("\n=== APPROACH B: Isolation Forest (unsupervised) ===")
    models, preds, val_rows = {}, [], []
    T_val = decision_years["val"][0]
    for T in decision_years["val"] + decision_years["test"]:
        sc = df.index[(df["year"] == T) & df["has_features"]]
        rng = np.random.default_rng(SEED + T)
        fit_idx = sc if len(sc) <= IF_FIT_ROWS else pd.Index(
            np.sort(rng.choice(sc, IF_FIT_ROWS, replace=False)))
        m = Pipeline([("prep", make_preprocessor()),
                      ("if", IsolationForest(**{**IF_PARAMS,
                                                "max_samples": min(IF_PARAMS["max_samples"], len(fit_idx))}))])
        m.fit(X_of(df, fit_idx, feat_cols))
        s = -m.score_samples(X_of(df, sc, feat_cols))   # higher = more anomalous
        models[T] = m
        preds.append(pred_frame(df, sc, "anomaly_iforest", s))
        print(f"  {T}: fit on {len(fit_idx):,} rows -> scored {len(sc):,} rows")
        if T == T_val:   # reported for comparison only; nothing is chosen from it
            met = rank_metrics(df.loc[sc, "y"].to_numpy(), s)
            val_rows.append({"model": "anomaly_iforest", "year": T,
                             "params": str(IF_PARAMS), "n_estimators": IF_PARAMS["n_estimators"],
                             **met})
            print(f"  val {T} (not used for tuning): PR-AUC={met['pr_auc']:.5f}  "
                  f"ROC={met['roc_auc']:.3f}  recall@1%={met['recall@1%']:.3f}  "
                  f"recall@5%={met['recall@5%']:.3f}")

    bundle = {"approach": "anomaly_iforest", "models": models, "features": feat_cols,
              "params": IF_PARAMS, "fit_rows": IF_FIT_ROWS, "labels_used": False}
    return bundle, preds, val_rows

# ---------------------------------------------------------------------------
def main():
    # Creates an argument parser object.
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["stub", "real"], required=True)
    ap.add_argument("--data-dir", default=None,
                    help="folder with the input parquet files")
    ap.add_argument("--quick", action="store_true",
                    help="sample only 100k negatives (smoke test)")
    ap.add_argument("--label", default="y_A_h2", choices=["y_A_h2", "y_B_h2", "y_A_h3"],
                    help="target label; non-default = sensitivity run (real data only)")
    args = ap.parse_args()

    global LABEL
    LABEL = args.label
    assert args.data == "real" or LABEL == "y_A_h2", "--label is for real data only"

    # If the user provides a custom data directory → use it.
    # Otherwise → use the default stub directory inside the repository.
    if args.data == "stub":
        data_dir = Path(args.data_dir) if args.data_dir else REPO / "shared_p3_p4_stub"
        out_dir = report_dir = REPO / "out" / "stub"
        df, feat_cols, h = load_stub(data_dir)
    else:
        data_dir = Path(args.data_dir) if args.data_dir else REPO / "files from team"
        out_dir, report_dir = REPO / "out", REPO / "reports"
        if LABEL != "y_A_h2":   # sensitivity runs never overwrite the primary results
            out_dir = REPO / "out" / LABEL
            report_dir = REPO / "reports" / "sensitivity" / LABEL
            report_dir.mkdir(parents=True, exist_ok=True)
        df, feat_cols, h = load_real(data_dir, LABEL)

    check_no_leakage(feat_cols)
    assert not df.duplicated(["npi", "year"]).any(), "duplicate (npi, year) rows"

    print(f"\nRows: {len(df):,} | features used: {len(feat_cols)} | horizon h={h}")
    print(f"Positives: {int(df['y'].sum()):,} | positive NPIs: {df.loc[df['y'] == 1, 'npi'].nunique():,}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")

    coverage_report(df, report_dir / "p3_scoring_coverage.csv")

    decision_years = DECISION_YEARS["stub"] if args.data == "stub" else DECISION_YEARS["real"][h]
    if args.data == "real":
        check_split_plan(data_dir / "split_plan.csv", h, sorted(df["year"].unique()), decision_years)
    split_summary(df, h, decision_years, report_dir / "p3_split_summary.csv")

    n_neg = N_NEG_QUICK if args.quick else N_NEG_TRAIN
    grid = STUB_LGB_GRID if args.data == "stub" else LGB_GRID

    val_rows = []
    lgb_bundle, lgb_preds, rows = run_lightgbm(df, feat_cols, h, decision_years,
                                               n_neg, grid, report_dir)
    val_rows += rows

    lr_bundle, lr_preds, rows = run_logreg(df, feat_cols, h, decision_years,
                                           n_neg, report_dir)
    val_rows += rows

    if_bundle, if_preds, rows = run_iforest(df, feat_cols, decision_years)
    val_rows += rows

    vm = pd.DataFrame(val_rows)
    vm["n_estimators"] = vm["n_estimators"].astype("Int64")   # logreg has none -> blank, not 8.0
    vm.to_csv(report_dir / "p3_val_metrics.csv", index=False)
    print(f"saved {report_dir / 'p3_val_metrics.csv'}")

    # Save the P4 handoff: one row per eligible NPI-year-approach, higher score = higher priority
    out_dir.mkdir(parents=True,exist_ok=True)
    preds = pd.concat(lgb_preds + lr_preds + if_preds, ignore_index=True)

    assert not preds.duplicated(["NPI", "year", "approach"]).any(), "duplicate prediction rows"

    assert preds["score"].notna().all(), "missing scores"
    preds.to_parquet(out_dir / "predictions.parquet", index=False)

    joblib.dump({"supervised_lgbm": lgb_bundle, "supervised_logreg": lr_bundle},
                out_dir / "supervised_model.pkl")
    joblib.dump(if_bundle, out_dir / "anomaly_model.pkl")

    print(f"\nsaved {len(preds):,} predictions -> {out_dir / 'predictions.parquet'}")
    print(preds.groupby(["approach", "year"]).size().to_string())
    print(f"saved {out_dir / 'supervised_model.pkl'} and {out_dir / 'anomaly_model.pkl'}")

if __name__ == "__main__":
    main()
