# Insights, Open Problems, and Future Directions

This file reproduces **Section 6** of the survey: patterns that only emerge across datasets, open problems in evaluation, and directions for future datasets.

## Cross-dataset observations (§6.1)

1. **Perception before reasoning helps, but strict ordering is not proven.**
   - SpatialLadder and Chen et al. (2025b) explicitly train perceptual tasks (spatial awareness) before reasoning-oriented phases (spatial cognition).
   - SpatialThinker and Sparkle provide supporting evidence while exploring different training orders.
   - The convergence suggests value in foundational spatial awareness, but not that a strict order is required.
2. **Staged training improves results at typical corpus sizes.** A common staging is perception → supervised spatial reasoning → RL.
   - Chen et al. (2025b): two-stage RL, perception first, then reasoning.
   - SpatialLadder: localization → multi-dimensional spatial understanding → RL with verifiable rewards.
   - SpatialLadder on VSI-Bench: **43.9** perception-then-spatial vs. **42.7** spatial-only vs. **40.7** mixed.
3. **Supervision design can partly substitute for scale.**
   - Large synthetic corpora work (SpatialVLM, InternSpatial).
   - SpatialThinker, trained on only ~7,000 QA samples with verifiable spatial rewards, beats both SFT and conventional RL on fourteen tasks.
   - This does not mean rewards always replace data, but carefully designed rewards can reduce reliance on very large static corpora.
4. **The value of RL depends on where it sits in the pipeline.**
   - SpaceR: from the same starting point, reward-based training outperforms SFT on all tested tasks.
   - ViGoRL: GRPO applied directly to the base model gives less spatially grounded reasoning than a warm start.
   - Compare RL and SFT only with the prior supervision and the RL insertion point held constant.
5. **Depth is a reliable supervision signal but a mixed runtime input.**
   - Large datasets use depth during construction and train RGB-only (SpatialVLM, InternSpatial).
   - At inference: geometry helps in robotics (RoboSpatial). In aerial scenes (Open3D-VQA) a 3D-LLM wins on distance estimation, while fine-tuned Qwen2-VL-7B has the higher overall score.
6. **Reference frames and viewpoints remain hard, and terminology is inconsistent.**
   - Models struggle with frame transformations and perspective-taking in RoboSpatial (ego/world/object), Open3D-VQA (allocentric–egocentric), and InternSpatial (multi-view).
   - Terms like "multi-view", "egocentric", and "object-centric" are used inconsistently across papers.

## Evaluation open problems (§6.2)

There is no unified way to evaluate spatial reasoning. Four issues make results hard to compare across the works in [benchmarks.md](benchmarks.md):

1. **Unvalidated LLM judges.** Open-ended answers are judged by LLMs (SSR, InternSpatial) that can misread spatial content. Agreement with human spatial judgments is not reported. SSR compares a second judge (GPT-4o-mini) with its original evaluator, which says nothing about agreement with humans.
2. **Per-paper tolerance thresholds.** SpatialRGPT uses ±25% relative error; VILASR uses mean relative accuracy over ten thresholds from 0.50 to 0.95. A looser threshold alone can raise accuracy, so numbers are not directly comparable.
3. **Shared train–test construction.** When the training and test sets come from the same pipeline, gains may partly reflect shared construction artifacts rather than spatial ability.
4. **Transfer depends on configuration.** High scores may reflect narrow task-specific performance. Accuracy on a single benchmark gives an incomplete view of spatial reasoning.

## Future directions (§6.3)

- **Unified dynamic annotation.**
  - Motion datasets are split between dynamic input (video) and dynamic reasoning over static images.
  - Motion, trajectory, visibility, and affordance annotations come from separate, task-specific systems (Video-STR, RoboPoint, SpatialMosaic, Embodied-R).
  - Needed: one annotation system that captures spatial, temporal, and affordance information together, plus the physics linking them.
- **Efficient reasoning.** RL improves multi-step spatial reasoning but can produce long chains (SpaceR) that are expensive to decode, and high-resolution images raise visual-token costs (VisionThink). Future datasets could include reasoning chains of varied length so that models reason longer only when needed.
- **More comparable evaluation.** Shared tasks, agreed tolerance ranges, common baselines, clearly reported scoring, and LLM judges validated against human spatial judgments.
- **Spatial scale and geospatial data.**
  - Most datasets cover Montello's (1993) *figural* and *vista* scales: small objects, tabletops, rooms.
  - Exceptions: SpaceVista (mm to km) and Open3D-VQA (aerial).
  - Geospatial datasets exist separately: remote sensing (GeoChat, VRSBench) and geospatial chain-of-thought (Liu et al. 2026b).
  - How spatial reasoning transfers between these scales is open.
- **Human-centred settings.** Spatial understanding matters for safe human–robot interaction (Sisbot et al. 2007), yet current datasets rarely capture a robot reasoning about space *jointly with a person*, where human intention and two-way interaction shape the spatial context.

## Conclusion (§7)

Future progress depends not only on larger datasets but on more consistent task definitions, reference frames, supervision strategies, and evaluation protocols.
