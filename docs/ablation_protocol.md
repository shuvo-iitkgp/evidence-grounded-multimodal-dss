# Controlled ablation: local pilot and full-study boundary

Status: implementation and real-data feasibility pilot. No prior manuscript values
are treated as observed results. The pasted design is incomplete after H1.

## Reproduce locally

From the repository root (Python 3.10+, Windows tar required for MediaEval RAR):

```powershell
python -m pip install -e .
python scripts/download_pilot.py mediaeval
python scripts/download_pilot.py fakeddit
python scripts/download_pilot.py weibo
mmfnd-pilot --data data/pilot/mediaeval --out outputs/pilot/mediaeval
mmfnd-ablation-report --scores outputs/pilot/mediaeval/scores.csv --out outputs/pilot/mediaeval/report
python -m unittest discover -s tests
```

Repeat the last two experiment commands with each successfully downloaded dataset.
Downloads use official metadata and original image URLs. Unavailable images are
excluded; manifests report actual retained sizes and class counts. The seed is
fixed before training. All five training seeds reuse identical selected splits.
Run with `--train`, `--val`, and `--test` on the download script to increase sizes.
Image-URL availability is a selection bias; these subsets are not full benchmarks.

The runner uses CUDA automatically when the installed PyTorch supports it, otherwise
CPU. `--device cuda` explicitly requires CUDA. `--encoder-batch-size 1` reduces
encoder memory; `--batch-size 8` reduces fusion-head memory. Frozen feature caching
avoids repeating expensive encoder passes for each ablation/seed. Do not compare
CPU pilot latency with the manuscript's A100 latency. A CUDA PyTorch build compatible
with the MX350's Pascal architecture is needed before this laptop can use CUDA;
the tested installation is CPU-only. The runner does not modify the system install.

On the development laptop, a separate PyTorch 2.7.1 + CUDA 11.8 environment was
installed on D: because C: lacked space. A real CUDA tensor check failed: the
installed NVIDIA driver exposes CUDA 10.2 and is too old for that runtime. Therefore
the measured pilot uses CPU. A compatible NVIDIA driver must be installed before
claiming GPU execution. No graphics-driver installation or reboot was performed.

The convenience wrapper accepts an explicit Python interpreter:

```powershell
.\scripts\run_local_pilot.ps1 -Dataset fakeddit -SkipDownload -Device cpu
# On a GPU machine with a working CUDA PyTorch environment:
.\scripts\run_local_pilot.ps1 -Dataset fakeddit -Python .venv\Scripts\python.exe -Device cuda -Train 10000 -Validation 2000 -Test 2000
```

## Fixed pilot protocol

- Five seeds: 11, 22, 33, 44, 55. Fixed subset seed 2026.
- Frozen CLIP ViT-B/32 text/vision encoders; max 77 tokens, standard 224px CLIP image
  transform. Same encoders/features for all configurations.
- A0: projected, mean-pooled text/image concatenation and classifier.
- A1: within-modality attention pooling followed by two-modality attention.
- 64 hidden units, AdamW 0.001, weight decay 0.01, dropout 0.2, batch 16, at most
  25 epochs, validation macro-F1 early stopping with patience 5, train-only inverse
  class-frequency weights. No augmentation, no event adversary in this pilot.
- A2 adds DTF; A3 adds evidence; A4 adds both with equal raw weights; A5 calibrates
  experts; A6 adds reliability gates. A2/A3 are **uncalibrated and ungated**, resolving
  the contradiction between the supplied prose and its sample LaTeX table in favor
  of clean incremental comparisons. They are not the table's calibrated/gated variants.
- DTF is computed from unique domains per training post only, using the manuscript
  formula with ParaMax=ParaMin=10, fixed before testing. Multiple domains are averaged.
  Domain support is their mean observation count. Unknown domains get no influence.
  Shorteners/platform/CDN hosts are excluded; image-host domains are not publisher trust.
  Fakeddit uses original submission-domain metadata, not the thumbnail URL. Other
  datasets extract actual URLs from post text. This preserves publisher domains
  that Fakeddit's cleaned title has removed.
- Evidence uses train-only CLIP image top-5 and CLIP text cosine reranking, then
  MiniLM NLI support/refute/neutral probabilities. No training-post veracity labels
  enter retrieval or verification. These captions are **not independent verified
  external sources**, CLIP reranking is **not DPR**, and MiniLM is **not T5**.
- Positive temperature/offset calibration minimizes validation NLL. A4/A5/A6 retain
  equal base weights to isolate the changed component. A6 searches the same fixed
  gate grid for each seed, selecting validation NLL. Threshold stays 0.5.
- Calibration of the evidence expert is an added implementation choice beyond the
  manuscript's conditional support/refute ratio. Its fitted values are exported.
- Missing context gets neutral probability and zero weight in every variant, giving
  exact fallback to the active multimodal expert. Calibration may change that expert.
- English CLIP/NLI on Chinese Weibo is only a plumbing stress test. Do not use these
  results to claim multilingual performance. Use suitable multilingual encoders and
  a validated multilingual verifier for the full study.

## Outputs and interpretation

`metrics_per_seed.csv`, `metrics_summary.csv`, dataset CSV/LaTeX tables, and
PNG/PDF/SVG component-gain, reliability, and score-stress figures contain measured
results. Metrics are on all posts and fixed domain/evidence/union-context subsets.
Standard deviation is sample SD across seeds. PR-AUC is average precision. Fake is
the operational positive class while stored labels are 0=fake, 1=real and probabilities
are P(real). ECE uses 15 equal-width confidence bins. Undefined metrics remain blank.

`paired_bootstrap.csv` gives within-seed paired effect differences and 95% intervals.
It clusters by image hash where available (no pooling repeated seed predictions as
independent posts). Seed SD and bootstrap intervals measure different uncertainty.
`mcnemar.csv` contains exact paired post-level tests and Holm correction across all
exported seed/comparison pairs. With shared images/events, those p-values are nominal;
the clustered intervals are more appropriate and significance claims need an
event-aware full-study analysis. These are exploratory pilot tests, not preregistration.

`decision_costs.csv` chooses review fraction on validation, then applies it to test.
It assumes perfect review and cost = reviewed_count + lambda * residual_errors.
For lambda <= 1, positive review cannot beat zero review under this equation:
one review costs 1 and corrects at most one error worth lambda. The manuscript's
claimed positive optimum and savings at these values cannot follow from that model.
At lambda=1 ties choose the smallest review count. Test labels never select policy.

`score_stress.csv` increases NEI mass and lowers similarity without changing claim
direction. It is explicitly a **score-level** stress test, not replaced/removed/
irrelevant passage evaluation. Gated versus gate-disabled uses the same fitted A6
parameters. `head_latency.csv` is warmed-up batch-one fusion-head timing only; it
does not support end-to-end latency, retrieval overhead, or an accuracy/latency Pareto claim.

## What remains necessary for the full paper

1. Replace pilot backbones with the exact appendix-selected checkpoints, retain
   token/region attention, and run the specified fine-tuning/optimizer protocol.
2. Supply a provenance-audited independent evidence corpus, a real DPR implementation,
   trained T5 verifier, and actual coreference resolver. Original repository modules
   use TF-IDF, a one-token support proxy, and pronoun deletion; those do not meet the design.
3. Implement E0-E4 with frozen components and annotated relevance/stance ground truth.
   Coverage, relevance, recall, MRR, and stance F1 cannot be derived from retrieval
   similarity or veracity labels alone. No such values are invented here.
4. Rerun actual passage corruption, domain-history resampling/corruption and counts
   0/1-5/6-20/>20; export availability and group sizes.
5. Measure complete warmed-up latency/throughput and GPU memory on the same device
   and instances, with explicit caching and retrieval accounting.
6. Run at least five seeds on full official datasets with recorded model revisions,
   data hashes, validation-only decisions, and an event/near-duplicate leakage audit.
   Exact-image deduplication here is not a near-duplicate or event-disjoint guarantee.

Original training/evaluation CLIs also import absent `mmfnd.models` and `mmfnd.datasets`
packages. The pilot is self-contained and does not claim to repair/reproduce those
missing originals. The unanchored `models/` and `datasets/` ignore rules could have
excluded them; recover the original implementations or explicitly implement a new
full-study model before claiming full manuscript reproduction.

## Sources

- Fakeddit official: https://github.com/entitize/Fakeddit (v2 multimodal official splits).
- MediaEval 2015: https://github.com/MKLab-ITI/image-verification-corpus.
- Weibo: https://github.com/wangzhuang1911/Weibo-dataset (original image URLs may expire).
- CLIP: https://huggingface.co/openai/clip-vit-base-patch32.
- NLI class order: https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768.
