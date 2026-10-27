# Evaluation Practices and Benchmarks

This file reproduces **Table 2** of the survey and the discussion in **Section 5**: how models fine-tuned on the datasets in [datasets.md](datasets.md) are evaluated. Evaluation should answer two questions. Did the targeted spatial skill improve? Were general vision–language abilities preserved?

## Table 2 — Evaluation practices

| Citation | Method | Answer type / scoring | Retent. | Independ. | Robust. | Human |
|---|---|---|:---:|:---:|:---:|:---:|
| Song et al. (2025) | RoboSpatial | cat, act / acc, geom | ✓ | ✓ | env | – |
| Chen et al. (2025a) | Robo2VLM | cat / acc | – | – | – | ✓ |
| Ning et al. (2025) | 3D ReasonSeg | cat, seg / acc, iou | – | – | – | – |
| Liu et al. (2025b) | SSR | cat, met, open / acc, judge | ✓ | – | – | – |
| Deng et al. (2026) | InternSpatial | cat, met, open / acc, judge | ✓ | – | – | ✓ |
| Ranasinghe et al. (2024) | Localize | cat, act / acc, geom | ✓ | ✓ | – | – |
| Zhang et al. (2025b) | Open3D-VQA | cat, met, open / acc, tol | – | – | s2r | – |
| Yuan et al. (2024) | RoboPoint | act / geom, task | ✓ | ✓ | s2r | – |
| Ogezi and Shi (2025) | SpaRE | cat / acc | ✓ | ✓ | – | – |
| Tang et al. (2025) | Sparkle | cat, met / acc | ✓ | – | – | – |
| Huang et al. (2026) | Spatial-DISE | cat / acc | – | – | – | ✓ |
| Ouyang et al. (2025) | SpaceR | cat, met / acc, mra | ✓ | ✓ | mod | – |
| Batra et al. (2025) | SpatialThinker | cat / acc | ✓ | – | – | – |
| Shen et al. (2025) | fDPO | cat, met / acc, judge, tol | – | – | – | – |
| Wu et al. (2025) | VILASR | cat, met / acc, mra | – | ✓ | – | – |
| Sarch et al. (2025) | ViGoRL | cat / acc | ✓ | ✓ | – | – |
| Chen et al. (2025b) | Perception before reasoning | cat / acc | ✓ | ✓ | – | – |
| Zhao et al. (2025) | Embodied-R | cat / acc | ✓ | – | – | – |
| Chen et al. (2026) | Think with 3D | cat, met / acc, mra | ✓ | – | – | – |
| Cai et al. (2025) | SpatialBot | cat, met, act / acc, task | ✓ | – | – | – |
| Li et al. (2026) | SpatialLadder | cat, met / acc, mra | ✓ | – | – | – |
| Cheng et al. (2024) | SpatialRGPT | cat, met / acc, tol | ✓ | – | para | – |
| Chen et al. (2024) | SpatialVLM | cat, met / acc, tol | ✓ | – | – | – |
| Wang et al. (2025) | Video-STR | cat, met / acc, mra | ✓ | ✓ | – | – |
| Sun et al. (2026) | SpaceVista | cat, met / acc, mra | – | – | mod | ✓ |
| Zhang et al. (2025a) | SPAR | cat, met / acc, mra | ✓ | – | – | ✓ |
| Lee et al. (2025) | SpatialMosaic | cat, met / acc, tol | – | – | env | ✓ |
| Cai et al. (2026) | SenseNova-SI | cat, met, open / acc, judge | ✓ | ✓ | dbias | – |
| **Count** | | | **20/28** | **10/28** | **8/28** | **6/28** |

The *Method* column is added here for convenience and is not part of the published table.

### Legend

- **Answer type.** Categorical (`cat`), metric (`met`), open-ended (`open`), action or point (`act`), segmentation (`seg`). A dash (–) denotes not reported, here and in every column.
- **Scoring.**
  - `acc`: accuracy or exact match.
  - `tol`: relative tolerance band, such as ±25% of the ground-truth value.
  - `mra`: mean relative accuracy.
  - `judge`: LLM-as-judge for semantic correctness (not only to extract an answer).
  - `geom`: geometric criterion, such as convex-hull containment.
  - `iou`: intersection-over-union.
  - `task`: downstream task success.
- **Retent.** General-capability retention reported on non-spatial vision–language benchmarks with a matched control. A checkmark denotes an assessment, not the absence of a decline in performance.
- **Independ.** The primary evaluation set is independent of the authors' own training-data construction pipeline. For entries that also serve as benchmarks, independence is assessed against the accompanying training split.
- **Robust.** A probe beyond headline accuracy: paraphrased questions (`para`), held-out environments (`env`), modality shift (`mod`), sim-to-real transfer (`s2r`), debiasing or shortcut probes such as no-vision and circular testing (`dbias`).
- **Human.** A human task-performance baseline is reported. This is distinct from human-annotated ground truth, data verification, rating model outputs, and baselines taken from existing benchmarks.

## What is measured

- **Target skill, broken down by task type.** A single overall score can average away specific failures.
  - SPAR-Bench reports depth and distance estimation, position matching, view-change inference, object relations, and spatial imagination separately.
  - Spatial-DISE crosses *intrinsic vs. extrinsic* with *static vs. dynamic*. Models fall below the human baseline in categories such as intrinsic-dynamic (mentally rotating or folding one object).
- **General-capability retention.** SpaRE reports MMMU, MMBench, and HallusionBench; fDPO reports MME and POPE. Retention is not consistent:
  - SPAR: MME drops from 2186 to 2163 vs. its matched control, and from 2323 to 2163 vs. the untrained base.
  - RoboSpatial: MME-cognition drops from 305.4 to 293.2; other general scores stay the same or increase.
  - fDPO compares against SpatialRGPT rather than its own initial model, so it does not show retention.
- **Transfer depends on the training mix.** In Spatial-DISE, fine-tuning on one quadrant (e.g., only viewpoint transformation) transfers "largely minimal or even detrimental[ly]" to other tasks. Fine-tuning on the full dataset improves all external tasks.

**Takeaway:** a general-benchmark result shows retention only relative to the right initial checkpoint under the same protocol. Comparisons to other models do not show what the adapted model gained or lost.

## How it is scored

| Answer type | Scoring | Examples |
|---|---|---|
| Categorical (left/right, yes/no, multiple choice) | Accuracy / exact match | SPAR, Spatial-DISE |
| Metric (distance, size, area) | ±25% relative tolerance; mean relative accuracy over thresholds 0.50–0.95 | SpatialRGPT (tol); SpaceR, VILASR (mra) |
| Open-ended | LLM-as-judge | SSR, InternSpatial |
| Mixed | LLM extracts the number or helps judge, then tolerance scoring | SpatialRGPT, fDPO, Open3D-VQA |
| Action / point | Geometric correctness, e.g., predicted 3D point inside the convex hull of reference points (described by its authors as deliberately conservative) | RoboSpatial |
| Action / point | Downstream robot task success | RoboPoint, SpatialBot |
| Any | Human baseline for calibration | Robo2VLM, Spatial-DISE |

## How generalization is tested

- **In-domain vs. out-of-domain benchmarks.** These reduce reliance on memorized answer formats and phrasing.
  - SpatialLadder reports gains on CV-Bench, SPAR-Bench, and ViewSpatial-Bench. This shows some cross-benchmark transfer but cannot rule out shared annotation practices or data sources.
  - VILASR evaluates across task categories with different input types and spatial demands.
- **Sim-to-real.** Open3D-VQA and RoboPoint test synthetic-trained models on real-world scenes.
- **Robustness probes.**
  - Paraphrased questions: SpatialRGPT. Rephrasing SpatialRGPT-Bench with GPT-4 lowers the qualitative score from 91.8 to 90.0.
  - Held-out environments: RoboSpatial.
  - Excluding training videos that overlap the evaluation set: SpaceR.

## Benchmarks referenced in the survey

| Benchmark | Role in the survey |
|---|---|
| SpatialRGPT-Bench | Spatial benchmark accompanying OSD; used for the paraphrase robustness probe |
| SPAR-Bench | Spatial benchmark accompanying SPAR-7M; per-skill breakdown |
| SpatialBench | Spatial benchmark accompanying SpatialBot; depth-input comparison |
| Spatial-DISE | Benchmark with an intrinsic/extrinsic × static/dynamic taxonomy and a human baseline |
| VSI-Bench | Used to compare SpatialLadder training orders (43.9 / 42.7 / 40.7) |
| CV-Bench, ViewSpatial-Bench | Out-of-domain spatial benchmarks used by SpatialLadder |
| SIBench | Suite consolidating ~20 open-source datasets (Yu et al. 2025) |
| MME, MMMU, MMBench, HallusionBench, POPE | General vision–language benchmarks used for retention checks |

## In this repo

Loaders for these benchmarks are in [`examples/extend_existing/`](../examples/extend_existing/):

- [VSI-Bench](../examples/extend_existing/VSI-Bench/)
- [BLINK](../examples/extend_existing/BLINK/)
- [SAT](../examples/extend_existing/3_SAT/)
- [3DSRBench](../examples/extend_existing/4_3DSRBech/)
- [Spatial-DISE](../examples/extend_existing/2_SPATIAL_DISE/)
