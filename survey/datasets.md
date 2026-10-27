# Spatial Reasoning Post-Training Datasets

This file reproduces **Table 1** of the survey and the dataset-building discussion of **Section 4**. It lists the training datasets (and dataset collections) used to fine-tune pretrained MLLMs/VLMs for spatial reasoning. Benchmarks are included only when a training split accompanies them.

## Table 1 — Datasets

| # | Citation | Name | Method / paper | Scale | View | Modality | Annotation |
|---|---|---|---|---|---|---|---|
| 1 | Song et al. (2025) | RoboSpatial | RoboSpatial | 4M<sup>Σ</sup> | single, object, ego, world | rgb, bbox, dmap, ptcl, cpose, nl | llm, proc, tmpl |
| 2 | Chen et al. (2025a) | Robo2VLM-1 | Robo2VLM | 685k | multi, ego, world | rgb, seg, nl, traj | tmpl, semi |
| 3 | Ning et al. (2025) | 3D ReasonSeg | Reasoning-based segmentation | 29.2k | world | seg, ptcl, nl | llm, tmpl, semi |
| 4 | Liu et al. (2025b) | SSR-COT | SSR | 1.2M | single, world | rgb, dmap, nl | llm, tmpl, other |
| 5 | Deng et al. (2026) | InternSpatial | InternSpatial | 12M | single, multi, ego | rgb, bbox, seg, nl | llm, proc, tmpl, semi |
| 6 | Ranasinghe et al. (2024) | Localize<sup>†</sup> | Learning to localize objects | 200k+ | single | vid, rgb, bbox, nl | llm, tmpl, semi |
| 7 | Zhang et al. (2025b) | Open3D-VQA | Open3D-VQA | 74.2k<sup>Σ</sup> | single, topdown, object, ego, world | rgb, bbox, seg, dmap, ptcl, cpose, nl | llm, proc, tmpl, semi |
| 8 | Yuan et al. (2024) | RoboPoint-1432K | RoboPoint | 1432k | single, ego, world | rgb, bbox, kpt, dmap, nl | proc, tmpl |
| 9 | Ogezi and Shi (2025) | SpaRE | SpaRE | 3.8M<sup>Σ</sup> | single, object | rgb, nl | llm, proc, semi |
| 10 | Tang et al. (2025) | Sparkle | Sparkle | 180k<sup>Σ</sup> | single, topdown, world | rgb, nl | proc, tmpl |
| 11 | Huang et al. (2026) | Spatial-DISE-12K | Spatial-DISE | 12.4k | single, multi, object | rgb, nl | proc, tmpl, semi |
| 12 | Ouyang et al. (2025) | SpaceR-151k | SpaceR | 151k | single, topdown, ego | vid, rgb, bbox, ptcl, nl | proc, tmpl |
| 13 | Batra et al. (2025) | STVQA-7K | SpatialThinker | 7.6k | single, ego | rgb, bbox, nl | llm, full |
| 14 | Shen et al. (2025) | Open Spatial Reasoning | fDPO | 900k<sup>Σ</sup> | single, object | rgb, seg, dmap, nl | llm, proc |
| 15 | Wu et al. (2025) | VILASR<sup>†</sup> | VILASR | 80.8k<sup>Σ</sup> | single, multi | vid, rgb, bbox, nl | llm, proc, tmpl |
| 16 | Sarch et al. (2025) | ViGoRL | ViGoRL | 86.6k<sup>Σ</sup> | single, object | rgb, nl | llm, other |
| 17 | Chen et al. (2025b) | – | Perception before reasoning | 409k<sup>Σ</sup> | single, ego | rgb, nl | llm, tmpl, semi, full |
| 18 | Zhao et al. (2025) | – | Embodied-R | 5.4k | ego, world | vid, rgb, kpt, nl, traj | – |
| 19 | Chen et al. (2026) | – | Think with 3D | 247k<sup>Σ</sup> | multi, ego | rgb, nl | llm |
| 20 | Cai et al. (2025) | SpatialQA<sup>†</sup> | SpatialBot | 746k | single, object, ego | rgb, bbox, dmap, nl, traj | llm, proc, tmpl, semi |
| 21 | Li et al. (2026) | SpatialLadder-26K | SpatialLadder | 26.6k | single, multi, ego, world | vid, rgb, bbox, nl | proc, tmpl |
| 22 | Cheng et al. (2024) | OSD | SpatialRGPT | 8.7M | single, ego, world | rgb, bbox, seg, dmap, nl | llm, proc, tmpl |
| 23 | Chen et al. (2024) | – | SpatialVLM | 2B | single, ego, world | rgb, bbox, seg, dmap, ptcl, nl | proc, tmpl |
| 24 | Wang et al. (2025) | STV-205k | Video-STR | 205k | multi | vid, rgb, bbox, nl | proc, tmpl, full |
| 25 | Sun et al. (2026) | SpaceVista-1M | SpaceVista | 1M | multi, ego, world | vid, rgb, bbox, kpt, seg, nl | llm, proc, tmpl, semi |
| 26 | Zhang et al. (2025a) | SPAR-7M | SPAR (From flatland to space) | 7M | single, multi, ego | vid, rgb, bbox, kpt | llm, proc, tmpl |
| 27 | Lee et al. (2025) | SpatialMosaic | SpatialMosaic | 2M | multi, ego | rgb, bbox, dmap, ptcl, cpose, nl | proc, tmpl |
| 28 | Cai et al. (2026) | SenseNova-SI-8M | SenseNova-SI | 8.5M | single, multi, topdown, object, ego, world | vid, rgb, bbox, ptcl, nl | proc, other |

The *Method / paper* column is added here for convenience. It is the name each work goes by in the survey text, not part of the published table.

### Legend

- **Name.** <sup>†</sup> marks dataset families reported under a single name: Localize (Localize-Instruct-200K, Localize-ActivityNet); VILASR (VILASR-ColdStart-33k, VILASR-RRS-8k, VILASR-RL-40k); SpatialQA (SpatialQA, SpatialQA-E). A dash (–) denotes not reported, here and in every column.
- **Scale.** Rounded QA/example counts unless marked. B/M/k: billions/millions/thousands. +: lower bound. <sup>Σ</sup>: survey-computed sum across visual/annotation records or stage collections.
- **View.** Combines viewpoint with frame of reference.
  - Viewpoint: single-view (`single`), multi-view (`multi`).
  - Frame: object-centric (`object`), egocentric (`ego`), world-centric (`world`).
  - Top-down (`topdown`): a flat 2D map representation.
- **Modality.**
  - Vision: video (`vid`), RGB (`rgb`), bounding box (`bbox`), keypoint (`kpt`), segmentation (`seg`).
  - Geometry: depth map (`dmap`), point cloud (`ptcl`), camera pose (`cpose`).
  - Text: natural language (`nl`), trajectory (`traj`).
- **Annotation.** How the fixed training corpus was annotated.
  - Automated: LLM-based (`llm`), procedural or 3D reconstruction (`proc`), template- or rule-based (`tmpl`), supervision derived from auxiliary vision models or from the model's own outputs (`other`).
  - Human involvement: partial (`semi`), full (`full`).

### Observations from the table

- Single-view settings dominate over multi-view settings.
- Egocentric framing is the most common frame of reference.
- Four datasets adopt a top-down representation (Open3D-VQA, Sparkle, SpaceR-151k, SenseNova-SI-8M).

## Construction strategies (§4.1)

- **Reformatting existing datasets** (most common). General-purpose vision datasets are extended and their annotations rewritten as spatial QA, which scales supervision without new data collection.
  - Ranasinghe et al. (2024) convert bounding boxes to instruction QA with coordinate answers.
  - RoboSpatial adds automatically computed robot-relevant signals (e.g., free space) to existing 2D/3D datasets.
  - SPAR-7M reformats richly annotated 3D indoor scenes into 7M QA pairs across 33 task types (single-view, multi-view, video).
- **Mixing sources.**
  - SpaceVista-1M combines indoor scans, outdoor scenes, and aerial videos.
  - SenseNova-SI aggregates 4M samples from open-source datasets under one spatial-capability taxonomy, then generates 4.5M more from 3D sources (ScanNet, ScanNet++, SUN RGB-D, Matterport3D, ...) to cover under-represented skills such as perspective-taking.
- **Built from scratch** (rare; usually task-specific or benchmark-oriented).
  - Sparkle: small synthetic dataset for basic skills (direction, distance, localization) that generalize to harder tasks.
  - Spatial-DISE: hybrid benchmark mixing real-world problems with synthetic scenes.

## Annotation approaches (§4.2)

Approaches are not mutually exclusive and are often combined in one pipeline. Most start with auxiliary perception models (detectors, depth/geometry estimators) that extract object- and geometry-level structure.

| Approach | How it works | Examples | Limitation |
|---|---|---|---|
| Template / rule-based | A rule computes a relation over a structured scene; a template verbalizes it. The answer exists before the question. | SpatialVLM (38 question types), SpatialRGPT (3D scene graphs → region QA), InternSpatial, SpatialLadder, Sparkle | Linguistic inflexibility; only covers relations present in the source representation |
| LLM-based | An LLM generates QA pairs, rationales, or reasoning traces from facts or captions; can also review samples. | SpatialRGPT (Llama), SpaRE (Qwen on spatial captions), 3D ReasonSeg (question–reasoning–answer triples), SSR (rationales distilled to latents), SpatialThinker (Claude Sonnet 4 generates, GPT-4o checks consistency) | Hallucination or misreading of visual content |
| Procedural / 3D reconstruction | Labels computed directly from reconstructed or simulated geometry. | SpatialVLM (metric point clouds from internet images), RoboPoint (simulation + camera pose), RoboSpatial (real images + 3D scans), SpatialMosaic (occlusion across views from ScanNet++ and Waymo) | Only properties computable from geometry; 3D estimation errors propagate |
| Model-generated | Model outputs become preference data or reward signals; no annotated reasoning chains needed, but answers still need references or a verifier. | SpatialThinker (dense RL rewards), SpaceR (map reward), Embodied-R (correctness + logic + format), fDPO (M3CTS reasoning traces + preference pairs) | Depends on reward/verifier design |
| Human supervision | Humans steer design or review at strategic points rather than labeling everything. | Robo2VLM (expert review before full generation), Spatial-DISE (hand-defined cognitive taxonomy), SpatialThinker (Visual Genome ground truth), RoboSpatial (hand-annotated eval set) | Does not scale to millions of pairs |

## Data modalities (§4.3)

- **Vision and language.** RGB + question + answer is the shared baseline; VQA is model-agnostic. Some language-only datasets (Li et al. 2024b) separate perception failures from reasoning failures.
- **Bounding boxes and segmentation masks.** Structure, not a new sensor signal. They can be inputs or *outputs*:
  - RoboPoint and Ranasinghe et al. encode coordinates as text tokens.
  - SpatialThinker uses them as target outputs.
  - SpatialLadder learns them in its first curriculum stage.
- **Depth maps and point clouds.** Two ways to use geometry:
  - *At construction time only.* SpatialVLM, SpatialLadder (ScanNet), and InternSpatial use 3D to generate labels; the model sees only RGB at inference. No runtime depth cost.
  - *As a model input.* SpatialBot feeds RGB-D: over 99% on the SpatialBench depth task vs. ~84–90% for RGB-only, and about 2× pick-and-place success. Untrained models do not benefit: GPT-4o zero-shot with depth drops from 70.6 to 61.8 on object position.
  - *Middle path.* SSR estimates monocular depth and compresses it into latent tokens, so no depth sensor is needed at inference.
  - Evidence across domains is mixed: 3D helps for robotics (RoboSpatial); for aerial scenes (Open3D-VQA) a 3D-LLM is better at distance but the advantage is not consistent.
- **Video and trajectories.** Needed for dynamic relations such as "moving into". "Dynamic" has two senses:
  - *Dynamic input* (the model sees video): SpaceR, Video-STR, Embodied-R.
  - *Dynamic reasoning on static images* (e.g., predict the scene after a rotation): Spatial-DISE, Open3D-VQA.

## Source environments (§4.4)

- **Robotics / tabletop / indoor:** RoboSpatial, Robo2VLM. RoboPoint uses almost exclusively procedurally generated simulated indoor scenes.
- **Indoor + outdoor driving:** SpatialMosaic (ScanNet++ and Waymo Open Dataset), evaluated separately to avoid indoor over-specialization.
- **General purpose:** InternSpatial, SpatialRGPT, covering everyday indoor and outdoor scenes.
- **Multi-scale (mm to km):** SpaceVista, covering indoor, outdoor, and aerial (drone-view) video.

## In this repo

Loaders and samples for two Table 1 datasets are in [`examples/extend_existing/`](../examples/extend_existing/):

- Robo2VLM-1 → [`1_Robo2VLM/`](../examples/extend_existing/1_Robo2VLM/)
- Spatial-DISE → [`2_SPATIAL_DISE/`](../examples/extend_existing/2_SPATIAL_DISE/)
