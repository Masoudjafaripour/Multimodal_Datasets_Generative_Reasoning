# Taxonomy of Data-Centric Spatial Reasoning

This file reproduces the survey's taxonomy (**Figure 1**) and the design dimensions of **Section 3**. The taxonomy has three branches: **dataset design**, **dataset building**, and **evaluation**.

## Figure 1 — Taxonomy

### 1. Dataset design dimensions

- **Ground truth**
  - Human supervision pipeline (Chen et al. 2025a; Huang et al. 2026)
  - Autonomous generation pipeline (Chen et al. 2024; Deng et al. 2026)
- **Patterns of tasks**
  - Perceptual reasoning, e.g., identifying an object (Li et al. 2026)
  - Relational reasoning, e.g., left-of, larger than (Cheng et al. 2024)
  - Temporal reasoning, e.g., "into" or "onto" (Ouyang et al. 2025; Wang et al. 2025)
  - Metric reasoning, e.g., estimating object size (Cai et al. 2025)
  - Occlusion-aware reasoning, e.g., targets not visible in the query view (Lee et al. 2025)
  - Planning, e.g., action sequences to achieve spatial objectives (Sun et al. 2026)
  - Perspective-transformation reasoning, e.g., matching objects across camera poses (Deng et al. 2026)
- **Spatial viewpoints**
  - Single-view: inferring a mug behind a laptop from one 2D scene (Chen et al. 2024)
  - Multi-view: an implicit mental model from multiple observations (Deng et al. 2026)
- **Frames of reference**
  - Egocentric, e.g., "to my left" from the observer's view (Zhao et al. 2025)
  - Object-centric, e.g., a bowl is "in front of" a car (Song et al. 2025)
  - World-centric, e.g., something "north of" another (Chen et al. 2024)
  - Top-down representation, e.g., a flat 2D map view (Ouyang et al. 2025)

### 2. Dataset building

- **Construction strategies**
  - Ground-up: build the corpus from scratch (Tang et al. 2025; Huang et al. 2026)
  - Reformatting: turn existing vision datasets into spatial QA pairs (Ranasinghe et al. 2024; Song et al. 2025)
  - Mixing sources: combine indoor scans, outdoor scenes, and aerial videos (Sun et al. 2026)
- **Annotation approaches**
  - Template- and rule-based (Chen et al. 2024; Cheng et al. 2024)
  - LLM-based (Ogezi and Shi 2025; Ning et al. 2025)
  - Procedural generation and 3D reconstruction (Chen et al. 2024; Yuan et al. 2024)
  - Model-generated (Batra et al. 2025; Shen et al. 2025)
  - Human supervised (Chen et al. 2025a; Huang et al. 2026)
- **Data modalities**
  - Vision and language (Song et al. 2025; Chen et al. 2025a)
  - Depth and 3D information (Liu et al. 2025b; Zhang et al. 2025b)
  - Bounding boxes and segmentation masks (Cheng et al. 2024; Ogezi and Shi 2025)
  - Video, trajectories, and keypoints (Wang et al. 2025; Yang et al. 2025)
- **Source environment**
  - Robotics and autonomous driving (Song et al. 2025; Yuan et al. 2024)
  - General purpose (Deng et al. 2026; Cheng et al. 2024)
  - Simulated indoor environments (Yuan et al. 2024)
  - Aerial and drone-view video (Sun et al. 2026; Zhang et al. 2025b)

### 3. Evaluation

- **What is measured**
  - Target-skill performance broken down by task type (Zhang et al. 2025a; Huang et al. 2026)
  - General-capability retention on vision–language benchmarks (Ogezi and Shi 2025; Li et al. 2026)
- **Metrics by answer type**
  - Accuracy for categorical answers, e.g., left/right, yes/no (Zhang et al. 2025a; Huang et al. 2026)
  - Tolerance-based scoring for metric answers, e.g., distance, size (Cheng et al. 2024; Ouyang et al. 2025)
  - LLM-as-judge for open-ended answers (Liu et al. 2025b; Deng et al. 2026)
  - Geometric and end-to-end task success for action answers (Song et al. 2025; Yuan et al. 2024)
  - Human baselines for calibration (Chen et al. 2025a; Huang et al. 2026)
- **Generalization testing**
  - In-domain vs. out-of-domain benchmarks (Li et al. 2026; Wu et al. 2025)
  - Robustness probes (Cheng et al. 2024; Song et al. 2025; Ouyang et al. 2025)
- **Open problems**
  - LLM judges unvalidated on spatial content (Liu et al. 2025b; Cheng et al. 2024)
  - Conventional, unjustified tolerance thresholds, e.g., ±25% (Cheng et al. 2024)
  - Shared train–eval construction pipelines (Liu et al. 2025b; Deng et al. 2026)

The building branch is detailed in [datasets.md](datasets.md), the evaluation branch in [benchmarks.md](benchmarks.md), and the open problems in [gaps_directions.md](gaps_directions.md).

## Key definitions (§2.1, §3.1)

- **Spatial awareness:** perceptual and cognitive ability to derive spatial information from visual input and keep an internal representation of the scene's spatial configuration (object locations, relative positions).
- **Spatial cognition:** manipulating or mentally simulating that internal representation. Some works use "spatial cognition" for what the survey calls spatial awareness.
- **Visual reasoning:** high-level inference built on spatial awareness: drawing conclusions, making comparisons, and answering questions beyond direct observation.
- **Spatial ground truth:** structured reference specifications of spatial information (object relations and locations, depth, compositional plans) used to train or evaluate a model on visual or multimodal input. Most spatial attributes are implicit, so ground truth has to be produced by a designed pipeline, either autonomously or with human supervision.

## Task patterns: perception to reasoning (§3.2)

| Task type | What the model must do | Typical ground-truth source | Awareness or cognition |
|---|---|---|---|
| Perceptual | Identify, localize, or detect presence of objects; SpatialLadder makes this level 1 of a curriculum | Detection / box annotations | Awareness |
| Relational | Qualitative (left-of, above, behind) or comparative (closer to, larger than) relations between object pairs | 3D scene metadata or object depth | Straddles both |
| Perspective-transformation | Reason under camera motion or match objects across camera poses | Multi-view geometry, camera poses | Cognition |
| Temporal / dynamic | Relations that need motion ("into", "onto") | Video; also hypothetical transformations of static images (Spatial-DISE) | Cognition |
| Metric | Estimate absolute distances or sizes | Depth sensors, or depth maps estimated during generation | Awareness |
| Occlusion-aware | Reason about targets not visible in the query view | Visibility masks | Cognition |
| Planning | Produce action sequences that reach spatial goals | Task / trajectory specifications | Cognition |

Relational reasoning straddles the divide. Reading a relation from an observed pair is perceptual; comparing objects across a transformation is not. Most datasets include tasks from both sides. For dynamic tasks, the survey recommends stating clearly whether the change comes from object motion, camera motion, or a hypothetical transformation.

## Viewpoints and frames of reference (§3.3)

- **Viewpoint**
  - *Single-view:* infer 3D relations from one 2D observation (SpatialVLM, SpatialRGPT).
  - *Multi-view:* combine several observations of the same scene (InternSpatial).
- **Frame of reference** (the frame changes which answer is correct)
  - *Egocentric:* relative to the observer; especially relevant to embodied agents (Embodied-R).
  - *World-centric:* in world coordinates, independent of the observer (SpatialVLM).
  - *Object-centric:* relative to the object itself. In RoboSpatial, a bowl can be *in front of* a car from a third-person view but *behind* it egocentrically.
  - *Top-down:* a flat 2D map of the environment (SpaceR-151k).
- **Multiple frames.** Datasets that cover several frames (e.g., SenseNova-SI-8M) teach that the same relation can be described differently depending on the frame.
