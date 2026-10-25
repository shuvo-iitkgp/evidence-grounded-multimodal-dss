"""Generate actual measured pilot tables/figures from audited val/test scores."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .fusion import CONFIGS, Fusion
from .metrics import metrics, paired, holm, PAIRS, LAMBDAS, choose_review_fraction, review_errors, reliability


def validate(frame):
    required = {"dataset","seed","split","id","label","early_logit","hier_logit",
                "dtf","domain_n","p_sup","p_ref","p_nei","s_max","evidence_available"}
    if required-set(frame):
        raise ValueError(f"Missing columns: {sorted(required-set(frame))}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Empty data or missing values")
    if not set(frame.split) <= {"val","test"} or set(frame.split) != {"val","test"}:
        raise ValueError("Both val and test splits are required")
    if not set(frame.label) <= {0,1} or not set(frame.evidence_available) <= {0,1}:
        raise ValueError("Labels and availability must be binary")
    numeric = ["early_logit","hier_logit","dtf","domain_n","p_sup","p_ref","p_nei","s_max"]
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise ValueError("Nonfinite expert scores")
    if (frame.domain_n < 0).any() or (~frame.s_max.between(-1,1)).any():
        raise ValueError("Invalid counts or similarities")
    probs = frame[["p_sup","p_ref","p_nei"]].to_numpy()
    if (probs < 0).any() or (probs > 1).any() or not np.allclose(probs.sum(1),1,atol=1e-5):
        raise ValueError("Verifier probabilities must form a distribution")
    if frame.duplicated(["dataset","seed","split","id"]).any():
        raise ValueError("Duplicate prediction IDs")
    for dataset, data in frame.groupby("dataset"):
        reference = None
        for seed, run in data.groupby("seed"):
            if set(run.split) != {"val","test"} or run.groupby("id").split.nunique().max() > 1:
                raise ValueError("Missing split or validation/test ID overlap")
            if "image_hash" in run and run.groupby("image_hash").split.nunique().max() > 1:
                raise ValueError("Validation/test image overlap")
            signature = run.sort_values(["split","id"])[["split","id","label"]].reset_index(drop=True)
            if reference is not None and not signature.equals(reference):
                raise ValueError(f"Different splits/labels across seeds for {dataset}")
            reference = signature


def savefig(fig, root, name):
    fig.tight_layout()
    for extension in ("png","pdf","svg"):
        fig.savefig(root/(name+"."+extension),dpi=200,bbox_inches="tight")
    plt.close(fig)


def generate(scores_path, out, draws=500):
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    frame = pd.read_csv(scores_path,dtype={"id":str})
    validate(frame)
    coverage = []
    for (dataset,seed,split), group in frame.groupby(["dataset","seed","split"]):
        coverage.append({"dataset":dataset,"seed":seed,"split":split,"n":len(group),
            "domain_available":int((group.domain_n > 0).sum()),
            "evidence_available":int(group.evidence_available.sum()),
            "domain_coverage":float((group.domain_n > 0).mean()),
            "evidence_coverage":float(group.evidence_available.mean())})
    pd.DataFrame(coverage).to_csv(out/"context_coverage.csv",index=False)
    rows, predictions, comparisons, mcnemar, states, stress, costs, rel_rows, domain_stress = [], [], [], [], [], [], [], [], []
    for (dataset,seed), run in frame.groupby(["dataset","seed"]):
        val = run[run.split == "val"].sort_values("id").reset_index(drop=True)
        test = run[run.split == "test"].sort_values("id").reset_index(drop=True)
        y = test.label.to_numpy(int)
        probs, fitted = {}, {}
        for config in CONFIGS:
            model = Fusion(config).fit(val)
            p = model.predict(test)
            probs[config], fitted[config] = p, model
            states.append({"dataset":dataset,"seed":int(seed),**model.state()})
            predictions.append(pd.DataFrame({"dataset":dataset,"seed":seed,"config":config,"id":test.id,
                                            "label":y,"p_real":p}))
            for subset, mask in (("all",np.ones(len(test),bool)),
                                 ("context",(test.evidence_available.astype(bool)|(test.domain_n > 0)).to_numpy()),
                                 ("domain",(test.domain_n > 0).to_numpy()),
                                 ("evidence",test.evidence_available.to_numpy(bool))):
                if mask.any():
                    rows.append({"dataset":dataset,"seed":seed,"config":config,"subset":subset,
                                 **metrics(y[mask],p[mask])})
            for rel in reliability(y,p):
                rel_rows.append({"dataset":dataset,"seed":seed,"config":config,**rel})
            vp = model.predict(val)
            for penalty in LAMBDAS:
                fraction = choose_review_fraction(val.label.to_numpy(),vp,penalty)
                count = int(np.ceil(fraction*len(test)))
                residual = review_errors(y,p)[count]
                cost = count+penalty*residual
                baseline = penalty*((p >= .5) != y).sum()
                costs.append({"dataset":dataset,"seed":seed,"config":config,"lambda":penalty,
                              "validation_review_fraction":fraction,"test_review_count":count,
                              "cost":cost,"no_review_cost":baseline,
                              "cost_reduction":1-cost/baseline if baseline else np.nan})
        for better, baseline in PAIRS:
            print(f"bootstrap {dataset}/{seed} {better}-{baseline}",flush=True)
            intervals, test_result = paired(y,probs[better],probs[baseline],draws=draws,
                clusters=test.image_hash.to_numpy() if "image_hash" in test else None)
            keys = {"dataset":dataset,"seed":seed,"comparison":better+"-"+baseline}
            comparisons.extend([{**keys,**interval} for interval in intervals])
            mcnemar.append({**keys,**test_result})
        support_bins = {"none":test.domain_n == 0,"1_to_5":test.domain_n.between(1,5),
                        "6_to_20":(test.domain_n > 5)&(test.domain_n <= 20),"over_20":test.domain_n > 20}
        for variant, disable in (("gated",False),("same_fit_without_domain_gate",True)):
            p = fitted["A6"].predict(test,disable_domain_gate=disable)
            for bucket,mask in support_bins.items():
                if mask.any():
                    domain_stress.append({"dataset":dataset,"seed":seed,"variant":variant,
                                          "domain_support":bucket,**metrics(y[mask],p[mask])})
        # Controlled score degradation; does not pretend to rerun corrupted passages.
        no_evidence = test.copy()
        no_evidence["evidence_available"] = 0
        fallback_f1 = metrics(y,fitted["A6"].predict(no_evidence))["macro_f1"]
        for severity in np.linspace(0,1,6):
            corrupted = test.copy()
            corrupted["s_max"] = test.s_max-severity*(test.s_max+1)
            corrupted["p_sup"] = test.p_sup*(1-severity)
            corrupted["p_ref"] = test.p_ref*(1-severity)
            corrupted["p_nei"] = 1-corrupted.p_sup-corrupted.p_ref
            for variant, disable in (("gated",False),("same_fit_without_evidence_gate",True)):
                p = fitted["A6"].predict(corrupted,disable_evidence_gate=disable)
                stress.append({"dataset":dataset,"seed":seed,"severity":severity,"variant":variant,
                               "a6_without_evidence_macro_f1":fallback_f1,
                               **metrics(y,p)})
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out/"metrics_per_seed.csv",index=False)
    summary = per_seed.groupby(["dataset","subset","config"]).agg(
        {col:["mean","std","count"] for col in per_seed.select_dtypes("number") if col != "seed"})
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(out/"metrics_summary.csv",index=False)
    pd.concat(predictions).to_csv(out/"predictions.csv",index=False)
    pd.DataFrame(comparisons).to_csv(out/"paired_bootstrap.csv",index=False)
    mc = pd.DataFrame(mcnemar)
    mc["p_holm"] = holm(mc.p_value.to_numpy())
    mc.to_csv(out/"mcnemar.csv",index=False)
    pd.DataFrame(costs).to_csv(out/"decision_costs.csv",index=False)
    pd.DataFrame(rel_rows).to_csv(out/"reliability_bins.csv",index=False)
    pd.DataFrame(stress).to_csv(out/"score_stress.csv",index=False)
    pd.DataFrame(domain_stress).to_csv(out/"domain_support.csv",index=False)
    (out/"fitted_parameters.json").write_text(json.dumps(states,indent=2),encoding="utf-8")
    for dataset in frame.dataset.unique():
        table = summary.loc[(dataset,"all")]
        display = pd.DataFrame(index=table.index)
        for name in ("macro_f1","roc_auc","pr_auc","ece","brier","nll"):
            display[name] = [f"{row[name+'_mean']:.4f} +/- {row[name+'_std']:.4f}" for _,row in table.iterrows()]
        display.to_csv(out/f"{dataset}_table.csv")
        display.to_latex(out/f"{dataset}_table.tex",escape=True,
            caption=f"Local pilot only: {dataset}, mean and sample standard deviation over seeds.",label=f"tab:pilot_{dataset}")
        fig,ax = plt.subplots(figsize=(7,4))
        wide = per_seed[(per_seed.dataset == dataset)&(per_seed.subset == "all")].pivot(index="seed",columns="config",values="macro_f1")
        gains = wide.subtract(wide.A1,axis=0)
        ax.bar(gains.columns,gains.mean(),yerr=gains.std(ddof=1),capsize=3,color="#286580")
        ax.axhline(0,color="black",lw=.7)
        ax.set(ylabel="Macro F1 change vs A1",title=f"{dataset}: pilot component gains (seed SD)")
        savefig(fig,out,f"{dataset}_component_gain")
        fig,ax = plt.subplots(figsize=(5,5))
        ax.plot([.5,1],[.5,1],"--",color="gray")
        rel = pd.DataFrame(rel_rows)
        for config in ("A4","A5","A6"):
            selected = rel[(rel.dataset == dataset)&(rel.config == config)]
            bins = pd.DataFrame([{
                "confidence":np.average(g.confidence,weights=g.n),
                "accuracy":np.average(g.accuracy,weights=g.n)} for _,g in selected.groupby("bin")])
            ax.plot(bins.confidence,bins.accuracy,"o-",label=config)
        ax.set(xlabel="Mean predicted confidence",ylabel="Observed accuracy",title=f"{dataset}: pilot calibration")
        ax.legend()
        savefig(fig,out,f"{dataset}_reliability")
        fig,ax = plt.subplots(figsize=(7,4))
        sdf = pd.DataFrame(stress)
        for variant, group in sdf[sdf.dataset == dataset].groupby("variant"):
            agg = group.groupby("severity").macro_f1.agg(["mean","std"])
            label = "Gated" if variant == "gated" else "Evidence gate disabled (same fit)"
            ax.errorbar(agg.index,agg["mean"],yerr=agg["std"],marker="o",capsize=3,label=label)
        ax.axhline(table.loc["A1","macro_f1_mean"],color="gray",ls="--",label="A1 (raw)")
        ax.axhline(sdf[sdf.dataset == dataset].a6_without_evidence_macro_f1.mean(),
                   color="black",ls=":",label="A6 with evidence removed")
        ax.set(xlabel="Score degradation severity",ylabel="Macro F1",title=f"{dataset}: pilot score stress (not passage corruption)")
        ax.legend(fontsize=8)
        savefig(fig,out,f"{dataset}_score_stress")
    latency_path = Path(scores_path).parent/"head_latency.csv"
    if latency_path.exists() and frame.dataset.nunique() == 1:
        latency = pd.read_csv(latency_path)
        efficiency = latency.groupby(["model","seed"]).head_ms.agg(
            mean_ms="mean",median_ms="median",p95_ms=lambda x: x.quantile(.95))
        efficiency["head_only_throughput_per_second"] = 1000/efficiency.mean_ms
        efficiency.to_csv(out/"head_only_efficiency.csv")
        fig,ax = plt.subplots(figsize=(6,4))
        for model, config in (("early","A0"),("hier","A1")):
            x = efficiency.loc[model,"p95_ms"].mean()
            y = per_seed[(per_seed.config == config)&(per_seed.subset == "all")].macro_f1.mean()
            ax.scatter([x],[y],s=70)
            ax.annotate(config,(x,y),xytext=(5,5),textcoords="offset points")
        ax.set(xlabel="Fusion-head p95 latency (ms; encoders excluded)",ylabel="Macro F1",
               title="Pilot head-only performance/cost (not end-to-end Pareto)")
        savefig(fig,out,"head_only_performance_latency")
    manifest = {"mode":"PILOT", "input_sha256":hashlib.sha256(Path(scores_path).read_bytes()).hexdigest(),
                "bootstrap_draws":draws,"bootstrap":"paired image-cluster when hashes supplied; within-seed intervals",
                "mcnemar":"Exact per seed; Holm across all exported comparisons; nominal post-level test ignores image clustering",
                "positive_class":"fake (stored label 0)","pr_auc":"average precision", "ece":"15 equal-width confidence bins",
                "missing_outputs":["End-to-end latency/Pareto: head-only measurements are not end-to-end latency",
                    "Retrieval precision/recall/MRR and stance F1: no relevance or stance ground truth",
                    "E4 coreference: no real resolver available", "Passage-level corruption: not implemented by score stress"],
                "cost":"Review fraction selected on validation; test cost evaluated at fixed policy; perfect-review assumption",
                "test_selection":"No test labels used by calibration, gate search or review policy fitting"}
    (out/"report_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    (out/"README.md").write_text("# Local pilot results\n\nMeasured subset results, not final manuscript evidence. "
        "See run_manifest.json in the parent run directory for architecture and data provenance. "
        "A0-A6 here are pilot analogues. Metrics use fake as the positive class; p_real stores P(real). "
        "Tables contain sample SD across training seeds. Bootstrap CIs are conditional on each fitted seed, "
        "clustered by exact image hash when available. McNemar p-values are nominal post-level checks and "
        "should not support significance claims when posts share images/events.\n\n"
        "Empty contextual subsets have no rows; missing evidence is never reported as measured retrieval quality. "
        "See report_manifest.json for unmeasured outputs and limitations.\n",encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--bootstrap",type=int,default=500)
    args = ap.parse_args()
    if args.bootstrap < 100:
        raise ValueError("At least 100 bootstrap draws required")
    generate(args.scores,args.out,args.bootstrap)


if __name__ == "__main__":
    main()
