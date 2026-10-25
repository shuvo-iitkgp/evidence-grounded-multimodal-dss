"""Package measured reports, without publishing downloaded text/images/weights."""
from pathlib import Path
import json
import shutil
import zipfile
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    target = Path("pilot_results")
    target.mkdir(exist_ok=True)
    tables = []
    seed_counts = {}
    sample_counts = {}
    devices = {}
    for dataset in ("mediaeval","fakeddit","weibo"):
        run = Path("outputs/pilot")/dataset
        if not (run/"report/metrics_summary.csv").exists():
            raise FileNotFoundError(f"Missing measured report: {dataset}")
        dest = target/dataset
        dest.mkdir(exist_ok=True)
        for path in (run/"report").iterdir():
            if path.is_file():
                shutil.copy2(path,dest/path.name)
                if path.suffix == ".svg":
                    copied = dest/path.name
                    copied.write_text("\n".join(line.rstrip() for line in copied.read_text(encoding="utf-8").splitlines())+"\n",encoding="utf-8")
        for name in ("run_manifest.json","encoder_provenance.json","verifier_provenance.json"):
            if (run/name).exists():
                shutil.copy2(run/name,dest/name)
        run_metadata = json.loads((run/"run_manifest.json").read_text())
        devices[dataset] = run_metadata["device"]
        sample_counts[dataset] = {split: stats["n"] for split,stats in run_metadata["data_manifest"]["splits"].items()}
        # Numeric expert outputs allow independently recomputing every report.
        scores = pd.read_csv(run/"scores.csv",dtype={"id":str})
        seed_counts[dataset] = int(scores.seed.nunique())
        cols = ["dataset","seed","split","id","label","image_hash","early_logit","hier_logit",
                "dtf","domain_n","p_sup","p_ref","p_nei","s_max","evidence_available"]
        scores[[c for c in cols if c in scores]].to_csv(dest/"expert_scores.csv",index=False)
        split_ids = []
        for split in ("train","val","test"):
            source = pd.read_csv(Path("data/pilot")/dataset/(split+".csv"),dtype={"id":str})
            split_ids.append(source[["id","label","image_hash"]].assign(split=split))
        pd.concat(split_ids).to_csv(dest/"subset_ids.csv",index=False)
        table = pd.read_csv(dest/"metrics_summary.csv")
        tables.append(table[table.subset == "all"])
    summary = pd.concat(tables)
    summary.to_csv(target/"combined_summary.csv",index=False)
    fig,axes = plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,(dataset,table) in zip(axes,summary.groupby("dataset",sort=True)):
        ax.bar(table.config,table.macro_f1_mean,yerr=table.macro_f1_std,capsize=3,color="#286580")
        ax.set_title(dataset+(" (English-model probe)" if dataset == "weibo" else ""))
        ax.set_ylim(0,1)
        ax.set_xlabel("Pilot configuration")
        ax.grid(axis="y",alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Macro F1 (mean +/- seed SD)")
    fig.suptitle("Real-data local pilot: fixed subsets and repeated training seeds",fontsize=13)
    fig.tight_layout()
    for extension in ("png","pdf","svg"):
        path = target/("pilot_macro_f1."+extension)
        fig.savefig(path,dpi=200,bbox_inches="tight")
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())+"\n",encoding="utf-8")
    plt.close(fig)
    lines = ["# Measured local ablation pilot", "", "These are real subset measurements, not final paper evidence. "
             f"Training seeds per dataset: {seed_counts}. The model uses frozen CLIP features, small fusion heads, "
             "train-caption retrieval and MiniLM NLI; it does not reproduce the manuscript's full architecture.","",
             "## Macro F1 (mean ± sample SD)","", "| Configuration | Fakeddit | MediaEval | Weibo (English-model probe) |",
             "|---|---:|---:|---:|"]
    for config in ["A"+str(i) for i in range(7)]:
        values = []
        for dataset in ("fakeddit","mediaeval","weibo"):
            row = summary[(summary.dataset == dataset)&(summary.config == config)].iloc[0]
            values.append(f"{row.macro_f1_mean:.4f} ± {row.macro_f1_std:.4f}")
        lines.append("| "+config+" | "+" | ".join(values)+" |")
    lines.extend(["", "![Pilot macro F1](pilot_macro_f1.png)", "", "## What to use", "",
        f"Retained subset sizes: {sample_counts}. Execution devices: {devices}.", "",
        "Each dataset folder has CSV/LaTeX tables, per-seed metrics, numeric expert scores, predictions, "
        "paired bootstrap intervals, nominal McNemar tests with Holm correction, calibration plots, "
        "score-stress curves, cost policies, context coverage, and head-only efficiency. "
        "Figures are supplied as PNG, PDF and SVG. Manifests and subset IDs record provenance.", "",
        "## Interpretation limits", "",
        "- A0 is early fusion; A1 hierarchical attention; A2 adds DTF; A3 adds evidence; "
        "A4 combines raw experts; A5 calibrates; A6 gates. A2/A3 are uncalibrated/ungated.",
        "- These measurements do not automatically support the proposed hypotheses. Negative changes are retained.",
        "- MediaEval repeats images across posts; clustered intervals account for exact-image repetition. "
        "Check the recorded unique-image counts and validation/test gap before drawing transfer conclusions.",
        "- Weibo uses English CLIP/NLI solely to exercise the pipeline. It cannot support multilingual performance claims.",
        "- Retrieved training captions are not independent fact-checked evidence. Availability is candidate availability, "
        "not annotated relevance. Retrieval relevance/stance scores and E0–E4 are not established by this pilot.",
        "- Timings are fusion-head only, affected by concurrent local workload; not end-to-end latency or retrieval cost. "
        "Hardware is recorded per run; see the protocol for the development laptop's CUDA-driver limitation.",
        "- Score degradation is not passage corruption. Domain support strata do not substitute for controlled history corruption.",
        "- The original paper's cost claims at lambda <= 1 contradict its stated perfect-review cost model; "
        "the implemented validation-selected policy correctly chooses zero review there.",
        "", "See [the full protocol and remaining work](../docs/ablation_protocol.md) before using values in a paper."])
    (target/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    archive = Path.cwd().parent/"ablation_pilot_results.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for path in target.rglob("*"):
            if path.is_file():
                z.write(path,path.relative_to(target.parent))
        z.write("docs/ablation_protocol.md","docs/ablation_protocol.md")
    print(archive)


if __name__ == "__main__":
    main()
