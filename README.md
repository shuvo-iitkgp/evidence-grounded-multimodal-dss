## Multi-Modal Fake News Detection with Trust-Aware Triage

### Controlled local ablation pilot

See [the reproducible pilot protocol](docs/ablation_protocol.md) for official
subset downloads, five-seed local runs, validation-only calibration/gating,
CSV/LaTeX tables, figures, and paired uncertainty analysis. The pilot uses frozen
CLIP features and a small NLI verifier to fit local hardware. It is **not the full
paper experiment**. The protocol lists missing components and unmeasured outputs.
The original training CLIs currently reference model/data-loader modules absent
from this repository; use `mmfnd-pilot` for the self-contained pilot.

This repository contains a research prototype for multimodal fake news detection with textual, visual, and contextual signals, uncertainty scoring, and cost-aware verification triage. The self-contained local pilot is runnable; the original full-study implementation is incomplete as described above.

The codebase is designed for research reproducibility, ablation studies, and real-world deployment experiments.

### Key Contributions

Multimodal classifier combining text + image encoders with hierarchical fusion

Event-adversarial training to reduce topic leakage

Hyperlink trust modeling via Domain Trust Factor (DTF)

External evidence retrieval:

Reverse image search

Text retrieval

Optional verifier model

Three decision signals:

PTS (Preliminary Trustworthiness Score)

CMC (Combined Metric of Classification)

CMPU (Combined Metric of Prediction Uncertainty)

Cost-aware triage for selective human verification

The pilot supports subset preparation, frozen-feature fusion training, expert-score ablations, and measured report generation. The original full-study CLIs require the missing model/data-loader implementations.

### Repository Structure
```
├── configs/                # YAML experiment configs
├── data/                   # ignored by git (datasets, images, indices)
├── scripts/                # dataset preparation & utilities
├── src/mmfnd/              # core library
│   ├── cli/                # train / eval / predict CLIs
│   ├── datasets/           # CSV + image loaders
│   ├── models/             # text, vision, fusion, adversary
│   ├── evidence/           # retrieval, verifier, caching
│   ├── trust/              # DTF computation
│   └── scoring/            # PTS, CMC, CMPU, triage
├── outputs/                # ignored (runs, checkpoints, reports)
├── .gitignore
├── pyproject.toml
└── README.md
```
### Citation

If you use this codebase, cite the corresponding paper:
```
@article{yourpaper2025,
  title={Multimodal Fake News Detection with Trust-Aware Triage},
  author={...},
  year={2025}
}
```

