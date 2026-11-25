"""Build the P4 evaluation from P3's existing scores and the prepared test cohorts."""

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
YEARS = (2022, 2023)
ORIGINAL_KS = (100, 500, 1000)
EXPLORATORY_KS = (2500, 5000)
KS = (*ORIGINAL_KS, *EXPLORATORY_KS)
PRESENTATION_KS = (1000, *EXPLORATORY_KS)
MODELS = ("supervised_lgbm", "supervised_logreg", "anomaly_iforest")
PEERS = {
    "p1_baseline_pymt_per_bene_peer_z": "pymt_per_bene_peer_z",
    "p1_baseline_srvcs_per_bene_peer_z": "srvcs_per_bene_peer_z",
}
SCORES = {
    **{model: f"score_{model}" for model in MODELS},
    "highest_total_payment": "medicare_payment",
    "highest_payment_per_beneficiary": "pymt_per_bene",
    **PEERS,
}
NAMES = dict(zip((*SCORES, "random_expected"), (
    "LightGBM", "Logistic Regression", "Isolation Forest", "Highest total payment",
    "Highest payment per beneficiary", "Peer-adjusted payment per beneficiary",
    "Peer-adjusted services per beneficiary", "Random expectation",
)))
COLOURS = ("#1769aa", "#e07b24", "#238b68")
BOOTSTRAP_DRAWS = 20_000
PAYMENT_COLUMNS = (
    "payment_weighted_recall_at_k",
    "payment_exposure_per_shortlist_place_usd",
    "payment_weighted_lift_at_k",
)


def payment_intervals(
    captured: np.ndarray, pool: np.ndarray, k: int, pool_size: int,
) -> dict[str, float]:
    # Each draw supplies both the captured dollars and the pool denominator.
    valid = pool > 0
    recall = captured[valid] / pool[valid]
    values = (recall, captured / k, recall / (k / pool_size))
    bounds = {}
    for column, samples in zip(PAYMENT_COLUMNS, values):
        low, high = np.quantile(samples, (0.025, 0.975)) if samples.size else (np.nan, np.nan)
        bounds[f"{column}_ci95_low"] = low
        bounds[f"{column}_ci95_high"] = high
    return bounds


def evaluate() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, shortlists = [], []
    for year in YEARS:
        cohort = pd.read_parquet(DATA / f"cohort_{year}.parquet")
        features = pd.read_parquet(
            ROOT / "stubs" / "provider_features.parquet",
            columns=["npi", "year", "medicare_payment", *PEERS.values()],
            filters=[("year", "==", year)],
        ).rename(columns={"medicare_payment": "p2_decision_year_payment"})
        cohort = cohort.merge(features, on=["npi", "year"], how="left", validate="one_to_one")
        if not cohort["medicare_payment"].eq(cohort["p2_decision_year_payment"]).all():
            raise ValueError(f"Cohort payments do not match P2 features for the same NPI and year: {year}")
        labels = cohort["y_A_h2"].to_numpy()
        payments = cohort["medicare_payment"].to_numpy(dtype=np.float64)
        npi = cohort["npi"].astype(str).to_numpy()
        size, positives = len(cohort), int(labels.sum())
        if (
            size < max(KS) or positives == 0 or not np.isin(labels, (0, 1)).all()
            or not cohort["year"].eq(year).all() or cohort["npi"].duplicated().any()
            or not np.isfinite(payments).all()
            or (payments < 0).any()
        ):
            raise ValueError(f"Invalid evaluation inputs for {year}")
        exposure = payments * labels
        pool_exposure = exposure.sum()
        positive_indices = np.flatnonzero(labels == 1)
        positive_payments = payments[positive_indices]
        rng = np.random.default_rng(year)
        draws = rng.integers(0, positives, size=(BOOTSTRAP_DRAWS, positives))
        resampled_pool = positive_payments[draws].sum(axis=1)
        for approach in NAMES:
            random = approach == "random_expected"
            if not random:
                scores = cohort[SCORES[approach]].to_numpy(dtype=np.float64)
                if approach in PEERS and np.isnan(scores).any() and not np.isnan(scores).all():
                    scores = np.where(np.isnan(scores), np.nanmin(scores), scores)
                if not np.isfinite(scores).all():
                    raise ValueError(f"Invalid scores for {year} {approach}")
                order = np.lexsort((npi, -scores))
                kind = "model" if approach in MODELS else "p1_peer_baseline" if approach in PEERS else "payment_baseline"
            for k in KS:
                share = k / size
                if random:
                    hits, captured = share * positives, share * pool_exposure
                    intervals = {f"{column}_ci95_{end}": np.nan
                                 for column in PAYMENT_COLUMNS for end in ("low", "high")}
                else:
                    selected = order[:k]
                    hits, captured = int(labels[selected].sum()), float(exposure[selected].sum())
                    member_payments = positive_payments * np.isin(positive_indices, selected)
                    resampled_captured = member_payments[draws].sum(axis=1)
                    intervals = payment_intervals(resampled_captured, resampled_pool, k, size)
                    shortlists.append(pd.DataFrame({
                        "year": year, "approach": approach, "K": k, "rank": np.arange(1, k + 1),
                        "npi": npi[selected], "ranking_value": scores[selected], "selection_type": kind,
                    }))
                rows.append({
                    "year": year, "approach": approach, "K": k, "pool_size": size, "positive_count": positives,
                    "pool_share_at_k": share,
                    "workload_type": "original" if k in ORIGINAL_KS else "exploratory",
                    "estimate_type": "random_expectation" if random else "observed",
                    "hits_at_k": hits, "precision_at_k": hits / k, "recall_at_k": hits / positives,
                    "lift_at_k": (hits / k) / (positives / size),
                    "captured_payment_exposure_usd": captured, "pool_payment_exposure_usd": pool_exposure,
                    "payment_weighted_recall_at_k": captured / pool_exposure if pool_exposure else np.nan,
                    "payment_exposure_per_shortlist_place_usd": captured / k,
                    "payment_weighted_lift_at_k": captured / (share * pool_exposure) if pool_exposure else np.nan,
                    **intervals,
                })
    return (
        pd.DataFrame(rows).sort_values(["year", "K", "approach"]),
        pd.concat(shortlists, ignore_index=True).sort_values(["year", "K", "approach", "rank"]),
    )


def write_charts(results: pd.DataFrame) -> None:
    for column, title, filename in (
        ("hits_at_k", "Later exclusions captured", "p4_hits_by_k.png"),
        ("precision_at_k", "Precision at each shortlist size", "p4_precision_by_k.png"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for year, ax in zip(YEARS, axes):
            for model, colour in zip(MODELS, COLOURS):
                rows = results.loc[
                    results["year"].eq(year) & results["approach"].eq(model)
                    & results["K"].isin(PRESENTATION_KS)
                ]
                ax.plot(rows["K"], rows[column], marker="o", label=NAMES[model], color=colour)
            ax.set(title=str(year), xlabel="Shortlist size (K)", xticks=PRESENTATION_KS)
            if column == "precision_at_k":
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))
            else:
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(axis="y", alpha=0.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3)
        fig.suptitle(title, fontweight="bold")
        fig.tight_layout(rect=(0, 0.1, 1, 0.95))
        fig.savefig(FIGURES / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)


def write_report(results: pd.DataFrame) -> None:
    lines = [
        "# FraudLens P4 evaluation",
        "",
        "P4 evaluated the saved P3 rankings for 2022 and 2023. The original workloads were K = 100, 500 and 1,000. After examining those results, P4 explored K = 2,500 and 5,000. A hit is a shortlisted individual practitioner who met the two-year Scenario A exclusion outcome.",
        "",
        "LightGBM and Logistic Regression scores rank exclusion risk. They are not calibrated probabilities because training down-sampled negatives and heavily weighted positives. Isolation Forest ranks unusual billing. P4 sorts scores descending and breaks ties by ascending NPI.",
        "",
        "## Metrics",
        "",
        "| Metric | Calculation |",
        "|---|---|",
        "| Hits@K | Later exclusions in the shortlist |",
        "| Precision@K | Hits divided by K |",
        "| Recall@K | Hits divided by later exclusions in the year's full pool |",
        "| Lift@K | Precision divided by the year's pool exclusion rate |",
        "| Payment-Exposure Recall@K | Captured payments divided by all decision-year Medicare payments to later-excluded providers in the year's pool |",
        "| Historical payment exposure per shortlist place | Captured payments divided by K |",
        "| Value-weighted lift@K | Captured payments divided by the exposure expected from K random selections in the same year |",
        "| Net savings and ROI | Pilot measures based on measured financial benefit, review costs, and measurable friction costs |",
        "",
        "Captured payments are Medicare payments to shortlisted providers who were later excluded. P4 verified each payment against P2's `medicare_payment` for the same NPI and decision year as the score.",
        "",
        "The 95% bootstrap intervals use 20,000 resamples of later-excluded NPIs within each year. Each draw is shared across methods, holds the shortlists fixed, and recalculates captured and pool payments. The intervals reflect sensitivity to which later-excluded providers appear in the resample. Zero-hit shortlists yield zero-width intervals under this design. The analytical random expectation has no interval.",
        "",
        "## Results by year and workload",
        "",
        "The table gives the three model rankings. The consolidated CSV also includes the four billing rules and the random expectation.",
        "",
        "| Year | K | Pool reviewed | Model | Hits | Precision | Recall | Lift | Payment exposure | Payment-Exposure Recall@K [95% CI] | Exposure per place [95% CI] | Value-weighted lift [95% CI] |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    model_rows = results.loc[results["approach"].isin(MODELS)]
    for row in model_rows.itertuples(index=False):
        lines.append(
            f"| {row.year} | {row.K:,} | {row.pool_share_at_k:.2%} | {NAMES[row.approach]} | {row.hits_at_k:.0f} | "
            f"{row.precision_at_k:.2%} | {row.recall_at_k:.2%} | {row.lift_at_k:.2f}× | ${row.captured_payment_exposure_usd:,.2f} | "
            f"{row.payment_weighted_recall_at_k:.5%} "
            f"[{row.payment_weighted_recall_at_k_ci95_low:.5%}–{row.payment_weighted_recall_at_k_ci95_high:.5%}] | "
            f"${row.payment_exposure_per_shortlist_place_usd:,.2f} "
            f"[${row.payment_exposure_per_shortlist_place_usd_ci95_low:,.2f}–${row.payment_exposure_per_shortlist_place_usd_ci95_high:,.2f}] | "
            f"{row.payment_weighted_lift_at_k:,.2f}× "
            f"[{row.payment_weighted_lift_at_k_ci95_low:,.2f}–{row.payment_weighted_lift_at_k_ci95_high:,.2f}×] |"
        )
    lines += [
        "",
        "## Pilot decision",
        "",
        "The presentation compares K = 1,000, 2,500 and 5,000. P4 chose 5,000 for the detailed exploratory comparison after seeing the test results. Of these three workloads, 5,000 was the only one where all three models had at least one hit in both years. At K = 2,500, Logistic Regression still had zero hits in 2023. LightGBM had the most observed hits at K = 5,000 in both years. Increasing K from 2,500 to 5,000 added 7 LightGBM hits in 2022 (5 to 12) and 3 in 2023 (2 to 5).",
        "",
        "Pilot LightGBM and Logistic Regression at the same K, set by the audit team's available review capacity. Compare both against the current billing rule and use the pilot results to decide which model to deploy. Payment exposure can be concentrated in a few providers. Track actual recoveries and measured review costs.",
        "",
        "Net benefit is measured financial benefit minus review and measurable friction costs. ROI is net benefit divided by those costs. The historical test data supply billing exposure and later exclusion outcomes.",
        "",
        "## Files",
        "",
        "- [Consolidated metrics](p4_results.csv)",
        "- [Dashboard](p4_dashboard.html)",
        "- [Top-K provider lists for the original workloads](p4_top_k_shortlists.csv). The rebuild writes the full provider-level export, including both exploratory workloads, to the local `out/p4/` folder.",
        "- [Three-slide P4 presentation](../presentation_figures/FraudLens_P4_Evaluation_ThreeSlides.pptx)",
        "",
    ]
    (REPORTS / "p4_evaluation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_dashboard(results: pd.DataFrame) -> None:
    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FraudLens P4 evaluation</title>
<style>body{background:#f4f7fa;color:#183047;font:16px/1.5 system-ui,sans-serif;margin:0}main{max-width:1120px;margin:auto;padding:24px}label{margin-right:1.5em}section{background:#fff;padding:1em;margin:1em 0;overflow:auto}table{border-collapse:collapse;width:100%}th,td{text-align:right;padding:.6em;border-bottom:1px solid #dce6ea}th:first-child,td:first-child{text-align:left}img{width:min(100%,700px);display:block;margin:auto}</style></head>
<body><main><h1>FraudLens audit evaluation</h1>
<p>A hit is a provider who met the two-year exclusion outcome. Exposure uses Medicare payments to later-excluded providers in the same decision year as their scores.</p>
<label>Test year <select id="year"><option>2022</option><option>2023</option></select></label>
<label>Shortlist size <select id="k">__WORKLOADS__</select></label>
<section><h2>Shortlist comparison</h2><p id="pool"></p>
<table><thead><tr><th scope="col">Method</th><th scope="col">Hits</th><th scope="col">Precision</th><th scope="col">Recall</th><th scope="col">Lift</th><th scope="col">Payment exposure</th><th scope="col">Payment-Exposure Recall@K</th><th scope="col">Exposure per shortlist place</th><th scope="col">Value-weighted lift</th></tr></thead><tbody id="rows"></tbody></table>
<p>Exposure intervals use 20,000 bootstrap resamples of later-excluded NPIs within each year and hold the shortlists fixed. At small K, a handful of providers can drive the results.</p></section>
<section><h2>Hits and precision at K = 1,000, 2,500 and 5,000</h2><img src="../presentation_figures/p4_hits_by_k.png" alt="Model hits by K and test year"><img src="../presentation_figures/p4_precision_by_k.png" alt="Model precision by K and test year"></section>
<section><h2>Pilot recommendation</h2><p>We chose K = 5,000 for the detailed exploratory comparison after seeing the test results. LightGBM had the most observed hits in both years at that workload. Pilot LightGBM and Logistic Regression against the current billing rule at the same K, based on the audit team’s capacity. Measure actual recoveries per review, net savings and ROI using audit outcomes and costs.</p><a href="p4_results.csv">Download results</a> · <a href="p4_evaluation_summary.md">Read the evaluation report</a></section>
<script>
const data=__DATA__;
const names=__NAMES__;
const year=document.getElementById("year"), k=document.getElementById("k");
const money=new Intl.NumberFormat("en-US",{style:"currency",currency:"USD"});
const percent=(n,d=2)=>n===null?"—":(100*n).toFixed(d)+"%";
const lift=n=>n===null?"—":n.toFixed(2)+"×";
const withInterval=(r,key,format)=>{
  const low=r[key+"_ci95_low"], high=r[key+"_ci95_high"];
  const point=r[key]===null?"—":format(r[key]);
  return low===null?" "+point:point+"<br><small>95% CI "+format(low)+"–"+format(high)+"</small>";
};
function render(){
  const matches=Object.keys(names).map(name=>data.find(r=>r.year===+year.value&&r.K===+k.value&&r.approach===name));
  const context=matches[0];
  document.getElementById("pool").textContent=context.pool_size.toLocaleString()+" providers in the year’s pool · "+context.positive_count+" later exclusions · "+percent(context.pool_share_at_k)+" reviewed · "+context.workload_type+" workload.";
  document.getElementById("rows").innerHTML=matches.map(r=>[
    names[r.approach],r.estimate_type==="observed"?r.hits_at_k:r.hits_at_k.toFixed(3),
    percent(r.precision_at_k),percent(r.recall_at_k),lift(r.lift_at_k),money.format(r.captured_payment_exposure_usd),
    withInterval(r,"payment_weighted_recall_at_k",n=>percent(n,5)),
    withInterval(r,"payment_exposure_per_shortlist_place_usd",n=>money.format(n)),
    withInterval(r,"payment_weighted_lift_at_k",lift)
  ].map(value=>"<td>"+value+"</td>").join("")).map(row=>"<tr>"+row+"</tr>").join("");
}
year.addEventListener("change",render); k.addEventListener("change",render); render();
</script></main></body></html>"""
    workloads = "".join(
        f'<option value="{k}"{" selected" if k == max(PRESENTATION_KS) else ""}>'
        f'{k:,}{" (exploratory)" if k in EXPLORATORY_KS else ""}</option>'
        for k in KS
    )
    page = (page.replace("__DATA__", results.to_json(orient="records", double_precision=15))
            .replace("__NAMES__", json.dumps(NAMES)).replace("__WORKLOADS__", workloads))
    (REPORTS / "p4_dashboard.html").write_text(page, encoding="utf-8")


def main() -> None:
    results, shortlists = evaluate()
    results.to_csv(REPORTS / "p4_results.csv", index=False)
    shortlists.loc[shortlists["K"].isin(ORIGINAL_KS)].to_csv(REPORTS / "p4_top_k_shortlists.csv", index=False)
    shortlists.to_csv(DATA / "top_k_shortlists.csv", index=False)
    write_charts(results)
    write_report(results)
    write_dashboard(results)
    print(f"Built P4 outputs with {len(results)} result rows.")


if __name__ == "__main__":
    main()
