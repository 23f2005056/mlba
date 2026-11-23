"""Build P4 comparison tables, figures, dashboard, and evaluation report.

All test-year panels remain separate. No observed billing is called savings.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
DATA = ROOT / "out" / "p4"
FIGURES = ROOT / "presentation_figures"
MODELS = ("supervised_lgbm", "supervised_logreg", "anomaly_iforest")
BASELINES = ("highest_total_payment", "highest_payment_per_beneficiary")
YEARS = (2022, 2023)
KS = (100, 500, 1000)
NAMES = {
    "supervised_lgbm": "LightGBM",
    "supervised_logreg": "Logistic regression",
    "anomaly_iforest": "Isolation Forest",
    "highest_total_payment": "Highest total payment",
    "highest_payment_per_beneficiary": "Highest payment per beneficiary",
    "random_expected": "Random expectation",
}
COLORS = {
    "supervised_lgbm": "#1769aa",
    "supervised_logreg": "#e07b24",
    "anomaly_iforest": "#238b68",
}


def require_complete(frame: pd.DataFrame, approaches: tuple[str, ...]) -> None:
    expected = {(y, k, a) for y in YEARS for k in KS for a in approaches}
    actual = set(zip(frame["year"], frame["K"], frame["approach"]))
    if actual != expected or frame.duplicated(["year", "K", "approach"]).any():
        raise ValueError("Missing or duplicate approach/year/K result")


def load_results() -> pd.DataFrame:
    model = pd.concat(
        [pd.read_csv(REPORTS / f"p4_topk_metrics_{approach}.csv") for approach in MODELS],
        ignore_index=True,
    ).rename(columns={"common_pool_size": "pool_size", "common_pool_positives": "positive_count"})
    model["estimate_type"] = "observed"
    baseline = pd.read_csv(REPORTS / "p4_baseline_topk_metrics.csv")
    baseline["estimate_type"] = np.where(baseline["approach"].eq("random_expected"), "random_expectation", "observed")
    require_complete(model, MODELS)
    require_complete(baseline, BASELINES + ("random_expected",))
    columns = [
        "year", "approach", "K", "pool_size", "positive_count", "estimate_type",
        "hits_at_k", "precision_at_k", "recall_at_k", "lift_at_k", "audits_per_hit",
    ]
    results = pd.concat([model[columns], baseline[columns]], ignore_index=True)
    for (year, k), group in results.groupby(["year", "K"]):
        if group["pool_size"].nunique() != 1 or group["positive_count"].nunique() != 1:
            raise ValueError(f"Unequal evaluation populations for {year}, K={k}")
    return results.sort_values(["year", "K", "approach"]).reset_index(drop=True)


def save_shortlists() -> None:
    frames = []
    for model in MODELS:
        frame = pd.read_csv(DATA / f"topk_{model}_provisional.csv", dtype={"npi": "string"})
        frame = frame.rename(columns={"score": "ranking_value"})
        frame["selection_type"] = "model"
        frames.append(frame[["year", "approach", "K", "rank", "npi", "ranking_value", "selection_type"]])
    payment = pd.read_csv(DATA / "topk_payment_baselines.csv", dtype={"npi": "string"})
    payment["selection_type"] = "payment_baseline"
    frames.append(payment[["year", "approach", "K", "rank", "npi", "ranking_value", "selection_type"]])
    random = pd.read_csv(DATA / "topk_seeded_random_example.csv", dtype={"npi": "string"})
    random["approach"] = "random_example"
    random["ranking_value"] = np.nan
    random["selection_type"] = "one_seeded_random_draw"
    frames.append(random[["year", "approach", "K", "rank", "npi", "ranking_value", "selection_type"]])
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["year", "approach", "K", "npi"]).any():
        raise ValueError("Duplicate provider within a shortlist")
    for (year, approach, k), group in combined.groupby(["year", "approach", "K"]):
        if len(group) != k or set(group["rank"]) != set(range(1, k + 1)):
            raise ValueError(f"Invalid shortlist: {year}, {approach}, K={k}")
    combined.sort_values(["year", "K", "approach", "rank"]).to_csv(
        REPORTS / "p4_top_k_shortlists.csv", index=False
    )


def chart_lines(results: pd.DataFrame, field: str, title: str, ylabel: str, filename: str, percentage: bool = False) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=False)
    for ax, year in zip(axes, YEARS):
        yearly = results.loc[results["year"].eq(year)]
        for model in MODELS:
            rows = yearly.loc[yearly["approach"].eq(model)].sort_values("K")
            ax.plot(rows["K"], rows[field], marker="o", linewidth=2.2, color=COLORS[model], label=NAMES[model])
        rows = yearly.loc[yearly["approach"].eq("random_expected")].sort_values("K")
        ax.plot(rows["K"], rows[field], linestyle=":", linewidth=1.8, color="#555555", label="Random expectation")
        # Both payment rules have zero hits at each displayed K.
        rows = yearly.loc[yearly["approach"].eq("highest_total_payment")].sort_values("K")
        ax.plot(rows["K"], rows[field], linestyle="--", linewidth=1.5, color="#a458a8", label="Payment rules (both)")
        ax.set_title(str(year), fontsize=13, fontweight="bold")
        ax.set_xticks(KS)
        ax.set_xlabel("Audit shortlist size (K)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(bottom=0)
        if percentage:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))
        else:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    fig.savefig(FIGURES / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def chart_coverage(coverage: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    labels = [str(y) for y in YEARS]
    scored_share = coverage["eval_pool_share_of_upstream_candidates"].to_numpy()
    hit_share = coverage["eval_pool_positive_share_of_upstream"].to_numpy()
    for ax, share, title in (
        (axes[0], scored_share, "Upstream candidates in individual eval pool"),
        (axes[1], hit_share, "Upstream positive records in eval pool"),
    ):
        ax.barh(labels, share, color="#1769aa", label="In evaluation pool")
        ax.barh(labels, 1 - share, left=share, color="#e2e6ec", label="Outside pool")
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set_title(title, fontsize=11)
        for i, value in enumerate(share):
            ax.text(value / 2, i, f"{value:.1%}", ha="center", va="center", color="white", fontweight="bold")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.15)
    fig.suptitle("Individual evaluation pool within upstream candidates", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(FIGURES / "p4_scoring_coverage.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown_report(results: pd.DataFrame, coverage: pd.DataFrame, comparison: pd.DataFrame, paired: pd.DataFrame) -> None:
    lines = [
        "# FraudLens P4 evaluation summary",
        "",
        "**Status: P4 evaluation complete; P1, P2 and P3 handoffs confirmed.** P1 confirmed Scenario A at a two-year horizon, complete uncensored 2022/2023 labels, and an individual-practitioner evaluation pool. P2 confirmed the individual-practitioner feature file and both payment baseline fields. P3 confirmed the prediction handoff, score direction, and 2020-only setting selection. The team supplied no fixed audit budget or cost, so P4 reports audit yield at all three K values without selecting one.",
        "",
        "## Decision evaluated",
        "",
        "At the end of a reporting year, rank eligible providers for audit. A hit is a provider with a later Scenario A LEIE exclusion within the two-year label window (`y_A_h2`). This is a proxy for audit prioritisation, not proven billing fraud or a measured recovery.",
        "",
        "Every model and baseline uses the same P1 evaluation pool within each year: eligible individuals with P2 features. P1's `eval_pool.parquet` was not supplied locally, so P4 reconstructed its published rule from labels and feature keys; the reconstructed keys exactly match P4's test cohorts. All providers in this pool have all three scores. K = 100, 500, and 1,000 are audit-capacity scenarios, not an operationally selected capacity. The 2020 data was used for validation, not for this held-out table.",
        "",
        "## Model scores and tie rule",
        "",
        "LightGBM and logistic regression scores are predicted probabilities of later exclusion. Isolation Forest is fitted separately to each decision year's billing features, which P3 says are available at the decision date, without exclusion labels. Its anomaly score was negated so more unusual billing ranks higher; that score is not a probability. P3 chose supervised settings on 2020 validation only, kept Isolation Forest settings fixed, and used matured labels from 2013–2020 for the supervised 2022 models and 2013–2021 for the supervised 2023 models. P3 confirms neither test year was used to choose model settings.",
        "",
        "Every ranking sorts by score descending, then NPI ascending when scores tie. This fixed tie rule uses no outcome information. P3 reports that final LightGBM has 13 trees, so tied scores can occur. P4 checked the minimum and maximum possible hit counts across cutoff tie orders; all held-out 2022/2023 cutoffs have the same hit count under every tie order. The large 2020 validation ties remain a separate sensitivity issue and did not determine K.",
        "",
        "## Held-out results",
        "",
        "| Year | Method | K=100 hits | K=500 hits | K=1,000 hits |",
        "|---|---|---:|---:|---:|",
    ]
    for year in YEARS:
        for method in MODELS + BASELINES:
            row = results.loc[results["year"].eq(year) & results["approach"].eq(method)].set_index("K")
            lines.append(f"| {year} | {NAMES[method]} | {int(row.loc[100, 'hits_at_k'])} | {int(row.loc[500, 'hits_at_k'])} | {int(row.loc[1000, 'hits_at_k'])} |")
    lines += [
        "",
        "The payment baselines found zero hits at every displayed capacity. The seeded random shortlists also found zero in most individual draws; their expected hit count is below 0.1 even at K=1,000. The results CSV contains precision, recall, lift, and audits per hit for each method and capacity.",
        "",
        "## What the comparison shows",
        "",
        "- In 2022, LightGBM found 3 hits at K=500 versus 2 for logistic regression and 0 for Isolation Forest. Logistic regression's 2 hits were already on LightGBM's list.",
        "- In 2023, Isolation Forest found 1 hit at K=500 while the supervised models found 0. At K=1,000, LightGBM and Isolation Forest each found 1, but they were different providers.",
        "- All 18 paired 95% bootstrap intervals for model hit differences include zero. The smallest exact paired p-value is 0.25. These fixed-shortlist checks do not establish a reliable winner or account for model retraining and future-year drift.",
        "- Across both test years there are 140 positive provider-year records but only 113 distinct positive NPIs. A supplementary pooled comparison resamples whole NPIs and also finds no reliable model difference; the primary metrics remain separate by year.",
        "- At the cutoffs used for the three models in 2022 and 2023, alternative ordering within tied scores would not change the observed hit counts.",
        "",
        "## Evaluation pool and upstream coverage",
        "",
        "| Year | Upstream eligible candidates | Individual evaluation pool (all scored) | Upstream candidates outside pool (unscored) | Positive records in evaluation pool | Positive records outside evaluation pool |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in coverage.itertuples(index=False):
        lines.append(f"| {row.year} | {row.upstream_eligible_candidates:,} | {row.individual_eval_pool:,} ({row.eval_pool_share_of_upstream_candidates:.1%}) | {row.upstream_candidates_outside_eval_pool:,} | {row.eval_pool_later_exclusion_records} | {row.upstream_positive_records_outside_eval_pool} |")
    lines += [
        "",
        "The broader candidate pool includes organisations and is **not** the test denominator. P2 says candidates without features are mainly organisations. The 4 and 3 positive records outside the individual evaluation pool were not scored because they had no P2 feature row; their specific entity types cannot be verified because the local candidate file has no entity-type column. Within the evaluation pool, score coverage is 100%. Shortlist Medicare payment totals describe historical billing exposure only; no audit cost, adjudicated fraud, recoverable payment, or savings measure is available.",
        "",
        "## Provisional recommendation",
        "",
        "Use LightGBM as a **candidate lead ranking for a prospective audit pilot**, because its observed hit counts are at least as high as the other models at most displayed capacities. Keep Isolation Forest as a comparator because it found a different 2023 hit. Do not deploy a combined list or call LightGBM a proven winner: the paired differences are highly uncertain. Report K=100, 500, and 1,000 as alternative workload levels. Select a single operational K only after an audit team states its review capacity or audit cost/value rule; do not tune K on the 2022/2023 test outcomes.",
        "",
        "A prospective pilot should record audited providers, confirmed findings, audit cost, and recoveries. Those observations are needed to estimate actual business value. Historical Medicare payments are not a savings estimate.",
        "",
        "## Presentation figures and source tables",
        "",
        "- [Hits by audit capacity](../presentation_figures/p4_hits_by_k.png)",
        "- [Precision by audit capacity](../presentation_figures/p4_precision_by_k.png)",
        "- [Evaluation-pool coverage within upstream candidates](../presentation_figures/p4_scoring_coverage.png)",
        "- [P4 PowerPoint section](../presentation_figures/FraudLens_P4_Evaluation.pptx)",
        "- [Interactive dashboard](p4_dashboard.html)",
        "- [Consolidated results](p4_results.csv), [top-K shortlists](p4_top_k_shortlists.csv), [model comparison](p4_model_comparison.csv)",
        "- [Per-year paired uncertainty](p4_model_paired_uncertainty.csv), [NPI-clustered pooled check](p4_pooled_npi_uncertainty.csv), [random benchmark](p4_seeded_random_baseline.csv), [coverage](p4_scoring_coverage_summary.csv)",
        "",
        "## Team confirmation before publication",
        "",
        "1. P1's outcome and evaluation-population definitions are confirmed in the team reply. Obtain the actual generated `eval_pool.parquet` if a direct file-to-file key comparison is required; the published construction has already been reconciled exactly.",
        "2. P2 confirmed the final individual-practitioner feature file, `medicare_payment` for total-payment volume and `pymt_per_bene` for payment-per-beneficiary intensity.",
        "3. P3's score direction, 2020-only selection and mature training-year split are confirmed. If predictions change, revalidate that handoff before rerunning P4.",
        "4. The team requested audit-yield reporting without a fixed budget or cost. P4 has completed that comparison; a future operational sponsor would need to state capacity or costs before selecting one K.",
        "",
    ]
    (REPORTS / "p4_evaluation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def dashboard(results: pd.DataFrame, coverage: pd.DataFrame) -> None:
    model_rows = results.loc[results["approach"].isin(MODELS)].copy()
    data = model_rows[["year", "approach", "K", "hits_at_k", "precision_at_k", "recall_at_k", "lift_at_k"]].to_dict(orient="records")
    covered = coverage[["year", "individual_eval_pool", "score_coverage_within_eval_pool", "upstream_candidates_outside_eval_pool", "upstream_positive_records_outside_eval_pool"]].to_dict(orient="records")
    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FraudLens P4 evaluation</title>
<style>body{font:16px/1.5 system-ui,Arial,sans-serif;color:#183047;background:#f4f7fa;margin:0}main{max-width:1100px;margin:auto;padding:28px}h1{margin-bottom:5px}p{max-width:850px}.note{background:#fff4d6;border-left:4px solid #db9b20;padding:12px 16px}.controls{display:flex;gap:16px;margin:24px 0}label{font-weight:600}select{font:inherit;padding:7px;margin-left:8px}table{border-collapse:collapse;width:100%;background:white}th,td{padding:11px;border-bottom:1px solid #dce3eb;text-align:left}th{background:#dceafa}.card{background:white;padding:18px;border-radius:9px;box-shadow:0 1px 4px #ced8e2;margin:18px 0}img{max-width:100%;height:auto}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}@media(max-width:760px){.grid{grid-template-columns:1fr}}</style>
</head><body><main><h1>FraudLens audit-priority evaluation</h1>
<p>Held-out 2022 and 2023 results. A hit means a provider later appeared in the two-year Scenario A exclusion label. It is not confirmed fraud or money saved. Isolation Forest ranks unusual decision-year billing without using exclusion labels.</p>
<div class="note"><strong>P4 evaluation complete:</strong> P1, P2 and P3 handoffs are confirmed. Model differences remain uncertain. With no fixed audit budget or cost, all three K values are shown.</div>
<div class="controls"><label>Year <select id="year"><option>2022</option><option>2023</option></select></label><label>Top K <select id="k"><option>100</option><option selected>500</option><option>1000</option></select></label></div>
<div class="card"><h2>Model comparison</h2><table><thead><tr><th>Model</th><th>Hits</th><th>Precision</th><th>Recall</th><th>Lift</th></tr></thead><tbody id="rows"></tbody></table><p id="coverage"></p></div>
<div class="grid"><div class="card"><h2>Hits by workload</h2><img src="../presentation_figures/p4_hits_by_k.png" alt="Hits by K and year"></div><div class="card"><h2>Precision by workload</h2><img src="../presentation_figures/p4_precision_by_k.png" alt="Precision by K and year"></div></div>
<div class="card"><h2>Evaluation pool</h2><img src="../presentation_figures/p4_scoring_coverage.png" alt="Share of upstream providers and positive records in the individual evaluation pool"></div>
<p>Both payment rules found zero hits at the displayed capacities. Random selection almost always found zero. The full analysis and limitations are in <a href="p4_evaluation_summary.md">the evaluation summary</a>.</p>
</main><script>
const data=__DATA__; const coverage=__COVERAGE__;
const names={supervised_lgbm:'LightGBM',supervised_logreg:'Logistic regression',anomaly_iforest:'Isolation Forest'};
function render(){const year=Number(document.getElementById('year').value),k=Number(document.getElementById('k').value);
 const selected=data.filter(r=>r.year===year&&r.K===k); const order=['supervised_lgbm','supervised_logreg','anomaly_iforest'];
 document.getElementById('rows').innerHTML=order.map(a=>{const r=selected.find(x=>x.approach===a);return `<tr><td>${names[a]}</td><td>${r.hits_at_k}</td><td>${(100*r.precision_at_k).toFixed(2)}%</td><td>${(100*r.recall_at_k).toFixed(2)}%</td><td>${r.lift_at_k.toFixed(1)}×</td></tr>`}).join('');
 const c=coverage.find(x=>x.year===year);document.getElementById('coverage').textContent=`The individual evaluation pool has ${c.individual_eval_pool.toLocaleString()} providers; all have three model scores. Another ${c.upstream_candidates_outside_eval_pool.toLocaleString()} upstream eligible candidates are unscored and outside this test denominator, including ${c.upstream_positive_records_outside_eval_pool} positive records.`}
document.getElementById('year').addEventListener('change',render);document.getElementById('k').addEventListener('change',render);render();
</script></body></html>"""
    page = page.replace("__DATA__", json.dumps(data)).replace("__COVERAGE__", json.dumps(covered))
    (REPORTS / "p4_dashboard.html").write_text(page, encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    results = load_results()
    results.to_csv(REPORTS / "p4_results.csv", index=False)
    save_shortlists()
    comparison = pd.read_csv(REPORTS / "p4_model_shortlist_comparison.csv")
    paired = pd.read_csv(REPORTS / "p4_model_paired_uncertainty.csv")
    pooled = pd.read_csv(REPORTS / "p4_pooled_npi_uncertainty.csv")
    coverage = pd.read_csv(REPORTS / "p4_scoring_coverage_summary.csv")
    if len(comparison) != 6 or len(paired) != 18 or len(pooled) != 9 or len(coverage) != 2:
        raise ValueError("Missing comparison, paired, or coverage results")
    if not ((paired["hit_difference_bootstrap_ci95_low"] <= 0) & (paired["hit_difference_bootstrap_ci95_high"] >= 0)).all():
        raise ValueError("Report text about uncertainty needs revision")
    if not ((pooled["provider_cluster_bootstrap_ci95_low"] <= 0) & (pooled["provider_cluster_bootstrap_ci95_high"] >= 0)).all():
        raise ValueError("Pooled NPI uncertainty text needs revision")
    comparison.to_csv(REPORTS / "p4_model_comparison.csv", index=False)
    chart_lines(results, "hits_at_k", "Later exclusions in top-K shortlists", "Hits", "p4_hits_by_k.png")
    chart_lines(results, "precision_at_k", "Audit-list yield by capacity", "Precision@K", "p4_precision_by_k.png", percentage=True)
    chart_coverage(coverage)
    markdown_report(results, coverage, comparison, paired)
    dashboard(results, coverage)
    print("Saved consolidated P4 outputs, three figures, dashboard, and evaluation report")


if __name__ == "__main__":
    main()
