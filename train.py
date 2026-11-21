"""
FraudLens - Phase 5: models + evaluation  (NPI-grouped, stratified K-fold CV)

Reads  data/processed/features.parquet, reports/feature_dictionary.csv
Writes reports/model_results.csv, reports/feature_importance.csv,
       data/processed/oof_scores.npz   (out-of-fold LightGBM scores for Phase 6/7)

Evaluation protocol (freeze this in reports/eval_protocol.md)
  * Every provider (NPI) sits in exactly ONE fold -> a provider is never in train and test.
  * Positive NPIs are spread evenly over folds (stratified).
  * "Top 1%" = highest-scoring 1% of providers WITHIN EACH BILLING YEAR (annual audit queue).
  * Metrics: recall@top1%, recall@top5%, lift@1% (= recall / 0.01), positives per 10k flagged,
    PR-AUC (mean over folds).  Bootstrap 95% CI resamples POSITIVE NPIs.
  * Unlabelled providers are "not observed excluded", NOT confirmed clean (PU caveat):
    precision numbers are lower bounds.

Install once:  pip install numpy scikit-learn lightgbm duckdb
Run:           python train.py            (full)      |  python train.py --quick   (debug, 10% negatives)
"""
import sys
import time
import warnings
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

FEATURES = Path("data/processed/features.parquet")
DICT_CSV = Path("reports/feature_dictionary.csv")
OUT_RESULTS = Path("reports/model_results.csv")
OUT_IMPORT = Path("reports/feature_importance.csv")
OUT_OOF = Path("data/processed/oof_scores.npz")
LABEL = "y_A"
K = 5
SEED = 42
N_NEG_TRAIN = 1_000_000        # negatives used per training fit (weighted back up)
QUICK = "--quick" in sys.argv
QS = (0.01, 0.05)
N_BOOT = 1000


def banner(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


# ------------------------------------------------------------------ load
banner("1. LOAD")
role = {}
for line in DICT_CSV.read_text(encoding="utf-8").splitlines()[1:]:
    parts = [p.strip('"') for p in line.split('","')]
    role[parts[0].strip('"')] = parts[-1].strip('"')
MODEL_COLS = [c for c, r in role.items() if r == "model input"]
TREND = [c for c in MODEL_COLS if c.startswith(("g_", "dz_")) or c == "years_observed_to_date"]
RAW = [c for c in MODEL_COLS if c.startswith("f_")]
PEER = [c for c in MODEL_COLS if c.startswith(("z_", "pr_", "ratio_"))]
SETS = {
    "full": MODEL_COLS,
    "no_trend": RAW + PEER,
    "raw_only (no peer norm)": RAW,
}
print(f"{len(MODEL_COLS)} model features | trend={len(TREND)} raw={len(RAW)} peer={len(PEER)}")

con = duckdb.connect(":memory:")
sel = ", ".join(f"COALESCE(CAST({c} AS FLOAT), CAST('NaN' AS FLOAT)) AS {c}" for c in MODEL_COLS)
where = ""
if QUICK:
    print("QUICK MODE: keeping all positive NPIs + 10% of the rest (numbers NOT comparable to full run)")
    where = (f"WHERE npi IN (SELECT npi FROM read_parquet('{FEATURES.as_posix()}') WHERE {LABEL}=1) "
             f"OR hash(npi) % 10 = 0")
data = con.execute(f"""
    SELECT CAST(npi AS BIGINT) AS npi, cms_year, CAST({LABEL} AS TINYINT) AS y,
           specialty_clean, CAST(tot_mdcr_pymt_amt AS DOUBLE) AS pay,
           COALESCE(excess_payment, 0.0) AS excess, {sel}
    FROM read_parquet('{FEATURES.as_posix()}') {where}""").fetchnumpy()
data = {k: np.asarray(np.ma.getdata(v)) for k, v in data.items()}
npi, year, y = data["npi"], data["cms_year"], data["y"].astype(np.int8)
spec = data["specialty_clean"]
X_all = np.column_stack([data[c] for c in MODEL_COLS]).astype(np.float32)
col_ix = {c: i for i, c in enumerate(MODEL_COLS)}
n = len(y)
print(f"Rows {n:,} | positives {int(y.sum()):,} | positive NPIs {len(np.unique(npi[y == 1])):,}")

# ------------------------------------------------------------------ folds
rng = np.random.default_rng(SEED)
uniq, inv = np.unique(npi, return_inverse=True)
is_pos_npi = np.isin(uniq, np.unique(npi[y == 1]))
fold_of = np.empty(len(uniq), dtype=np.int8)
pidx = np.where(is_pos_npi)[0]
fold_of[pidx] = rng.permutation(len(pidx)) % K
fold_of[~is_pos_npi] = rng.integers(0, K, size=int((~is_pos_npi).sum()))
fold = fold_of[inv]
for k in range(K):
    m = fold == k
    print(f"  fold {k}: rows {m.sum():>9,}  positives {int(y[m].sum()):>4}  "
          f"positive NPIs {len(np.unique(npi[m & (y == 1)])):>4}")


# ------------------------------------------------------------------ metrics
def flags_topq(score, years, q):
    """Boolean mask of the top-q share of rows within each billing year."""
    f = np.zeros(len(score), dtype=bool)
    for yr in np.unique(years):
        m = years == yr
        thr = np.quantile(score[m], 1 - q)
        f[m] = score[m] >= thr
    return f


def clean(s):
    return np.nan_to_num(s.astype(np.float64), nan=-1e9, posinf=1e9, neginf=-1e9)


def prep_linear(Xtr, Xte, cols):
    """Median-impute with train medians + missing indicators for sparse columns, then scale."""
    idx = [col_ix[c] for c in cols]
    A, B = Xtr[:, idx].copy(), Xte[:, idx].copy()
    med = np.nanmedian(A, axis=0)
    med = np.where(np.isnan(med), 0, med)
    ind_cols = [j for j, c in enumerate(cols) if c.startswith(("g_", "dz_"))]
    ind_tr = np.isnan(A[:, ind_cols]).astype(np.float32) if ind_cols else np.zeros((len(A), 0), np.float32)
    ind_te = np.isnan(B[:, ind_cols]).astype(np.float32) if ind_cols else np.zeros((len(B), 0), np.float32)
    for j in range(A.shape[1]):
        A[np.isnan(A[:, j]), j] = med[j]
        B[np.isnan(B[:, j]), j] = med[j]
    sc = StandardScaler().fit(A)
    return (np.hstack([np.clip(sc.transform(A), -8, 8), ind_tr]),
            np.hstack([np.clip(sc.transform(B), -8, 8), ind_te]))


def sample_train(tr_idx):
    pos = tr_idx[y[tr_idx] == 1]
    neg = tr_idx[y[tr_idx] == 0]
    if len(neg) > N_NEG_TRAIN:
        neg_s = rng.choice(neg, N_NEG_TRAIN, replace=False)
    else:
        neg_s = neg
    idx = np.concatenate([pos, neg_s])
    w = np.concatenate([np.ones(len(pos)), np.full(len(neg_s), len(neg) / max(len(neg_s), 1))])
    return idx, w


# ------------------------------------------------------------------ run CV
banner("2. CROSS-VALIDATION")
names = (["BASE total Medicare payment", "BASE peer z-score payment/benef",
          "BASE specialty base rate", "Logistic regression", "Isolation Forest"]
         + [f"LightGBM [{k}]" for k in SETS])
flags = {nm: {q: np.zeros(n, dtype=bool) for q in QS} for nm in names}
prauc = {nm: [] for nm in names}
oof_lgb = np.zeros(n, dtype=np.float32)
importance = {c: 0.0 for c in MODEL_COLS}

for k in range(K):
    t0 = time.time()
    te = np.where(fold == k)[0]
    tr = np.where(fold != k)[0]
    yte, yrte = y[te], year[te]
    scores = {}

    # --- baselines (no training except specialty rate)
    scores["BASE total Medicare payment"] = clean(data["pay"][te])
    scores["BASE peer z-score payment/benef"] = clean(X_all[te, col_ix["z_ppb"]])
    rate = {}
    for s in np.unique(spec[tr]):
        m = spec[tr] == s
        rate[s] = y[tr][m].mean()
    base = np.array([rate.get(s, 0.0) for s in spec[te]], dtype=np.float64)
    scores["BASE specialty base rate"] = base + rng.random(len(te)) * 1e-9  # random tie-break

    # --- supervised models on a weighted negative sample
    idx, w = sample_train(tr)

    lr_cols = MODEL_COLS
    A, B = prep_linear(X_all[idx], X_all[te], lr_cols)
    lr = LogisticRegression(C=0.1, max_iter=300, class_weight="balanced")
    lr.fit(A, y[idx], sample_weight=w / w.mean())
    scores["Logistic regression"] = lr.decision_function(B)

    # --- Isolation Forest (unsupervised, ignores labels)
    peer_idx = [col_ix[c] for c in PEER]
    sub = rng.choice(tr, min(200_000, len(tr)), replace=False)
    Asub = np.nan_to_num(X_all[sub][:, peer_idx], nan=0.0)
    iso = IsolationForest(n_estimators=100, max_samples=256, random_state=SEED, n_jobs=-1).fit(Asub)
    scores["Isolation Forest"] = -iso.score_samples(np.nan_to_num(X_all[te][:, peer_idx], nan=0.0))

    # --- LightGBM variants
    for nm, cols in SETS.items():
        ix = [col_ix[c] for c in cols]
        model = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
            random_state=SEED, n_jobs=-1, verbose=-1)
        model.fit(X_all[idx][:, ix], y[idx], sample_weight=w)
        s = model.predict_proba(X_all[te][:, ix])[:, 1]
        scores[f"LightGBM [{nm}]"] = s
        if nm == "full":
            oof_lgb[te] = s
            for c, g in zip(cols, model.booster_.feature_importance("gain")):
                importance[c] += g / K

    for nm, s in scores.items():
        for q in QS:
            flags[nm][q][te] = flags_topq(s, yrte, q)
        prauc[nm].append(average_precision_score(yte, s))
    print(f"  fold {k} done in {time.time() - t0:.0f}s")

# ------------------------------------------------------------------ results
banner("3. RESULTS (out-of-fold, provider-disjoint)")
pos_rows = np.where(y == 1)[0]
pos_npi_u, pos_inv = np.unique(npi[pos_rows], return_inverse=True)
tot_i = np.bincount(pos_inv).astype(np.float64)
base_rate = y.mean()
brng = np.random.default_rng(SEED + 1)
boot_idx = brng.integers(0, len(pos_npi_u), size=(N_BOOT, len(pos_npi_u)))

rows = []
for nm in names:
    rec, ci, ppk = {}, {}, {}
    for q in QS:
        f = flags[nm][q]
        cap_i = np.bincount(pos_inv, weights=f[pos_rows].astype(np.float64), minlength=len(pos_npi_u))
        rec[q] = cap_i.sum() / tot_i.sum()
        boots = cap_i[boot_idx].sum(axis=1) / tot_i[boot_idx].sum(axis=1)
        ci[q] = (np.percentile(boots, 2.5), np.percentile(boots, 97.5))
        ppk[q] = f[pos_rows].sum() / f.sum() * 1e4
    rows.append({
        "model": nm,
        "recall@1%": rec[0.01], "ci_lo@1%": ci[0.01][0], "ci_hi@1%": ci[0.01][1],
        "lift@1%": rec[0.01] / 0.01,
        "recall@5%": rec[0.05], "ci_lo@5%": ci[0.05][0], "ci_hi@5%": ci[0.05][1],
        "pos_per_10k_flagged@1%": ppk[0.01],
        "PR-AUC (mean of folds)": float(np.mean(prauc[nm])),
        "PR-AUC lift vs base rate": float(np.mean(prauc[nm])) / base_rate,
    })

hdr = f"{'model':36s} {'recall@1%':>9s} {'95% CI':>15s} {'lift':>5s} {'recall@5%':>10s} {'PR-AUC':>8s} {'xBase':>6s}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['model']:36s} {r['recall@1%']:9.3f} "
          f"[{r['ci_lo@1%']:.3f},{r['ci_hi@1%']:.3f}] {r['lift@1%']:5.1f} "
          f"{r['recall@5%']:10.3f} {r['PR-AUC (mean of folds)']:8.5f} {r['PR-AUC lift vs base rate']:6.1f}")
print(f"\nBase rate (random PR-AUC) = {base_rate:.6f}.  Random ranking: recall@1% ~ 0.010, recall@5% ~ 0.050.")

OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
keys = list(rows[0].keys())
with open(OUT_RESULTS, "w", encoding="utf-8") as f:
    f.write(",".join(keys) + "\n")
    for r in rows:
        f.write(",".join(f'"{v}"' if isinstance(v, str) else f"{v:.6f}" for v in r.values()) + "\n")

banner("4. FEATURE IMPORTANCE (LightGBM full, mean gain over folds)")
imp = sorted(importance.items(), key=lambda kv: -kv[1])
tot = sum(v for _, v in imp) or 1
for c, v in imp[:15]:
    print(f"  {c:26s} {v / tot:6.1%}")
with open(OUT_IMPORT, "w", encoding="utf-8") as f:
    f.write("feature,share_of_gain\n")
    for c, v in imp:
        f.write(f"{c},{v / tot:.6f}\n")

OUT_OOF.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT_OOF, npi=npi, year=year, y=y, score=oof_lgb, excess=data["excess"],
                    pay=data["pay"], fold=fold)
print(f"\nSaved {OUT_RESULTS}, {OUT_IMPORT}, {OUT_OOF}")
print("""
HOW TO READ
  * A model only 'works' if its CI sits clearly above the BASE rows.
  * LightGBM [full] vs [no_trend]: if full is much better, trend features carry the gain -
    check whether that reflects providers ALREADY being investigated (payment drops).
  * LightGBM [full] vs [raw_only]: the value of peer normalisation (a proposal ablation).
  * Report these as indicative: ~500 positive NPIs -> wide intervals.
""")
banner("DONE")
con.close()