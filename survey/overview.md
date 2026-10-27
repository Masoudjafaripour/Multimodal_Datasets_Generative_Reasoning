# Towards a Data-Centric Understanding of Spatial Reasoning in Multimodal Large Language Models

This repository accompanies our survey **“Towards a Data-Centric Understanding of Spatial Reasoning in Multimodal Large Language Models”** ([arXiv:2610.19108](https://arxiv.org/abs/2610.19108)) by Mahdieh Mallahnezhad, Masoud Jafaripour, Mohamadali Tofigh, Sheila Schoepp, and Osmar R. Zaiane (University of Alberta, Amii). The survey provides a **data-centric view** of how spatial reasoning capabilities in Vision-Language Models (VLMs) and Multimodal Large Language Models (MLLMs) are learned, evaluated, and limited by dataset design.

## What this survey is about

Spatial reasoning—understanding object relations, geometry, distance, perspective, and spatial dynamics—is essential for embodied AI, robotics, navigation, and real-world decision making. Despite strong performance in visual recognition and language understanding, current VLMs struggle with even basic spatial inferences. Our survey argues that **datasets, not architectures alone, are the primary bottleneck**.

We systematically analyze how existing datasets encode spatial supervision, how they are constructed, and how they shape model behavior.

## Key contributions

* **Dataset-centric taxonomy** of spatial reasoning fine-tuning datasets, covering scale, modalities, viewpoints, annotation methods, and task types.
* **Unified view of dataset construction pipelines**, from template-based and LLM-generated annotations to procedural 3D reconstruction and reinforcement-learning-based supervision.
* **Analysis of spatial task patterns**, spanning perception, relational reasoning, metric reasoning, perspective transformation, temporal reasoning, and planning.
* **Evaluation practices and metrics** used to measure spatial generalization, robustness, and downstream embodied performance.
* **Cross-dataset insights and gaps**, highlighting trade-offs between scale and precision, the role of curricula and RL, and missing directions such as efficient reasoning and human-centered spatial interaction.

## Scope and focus

* We focus on **fine-tuning datasets** (not only benchmarks) used to improve spatial reasoning in pretrained VLMs/MLLMs.
* Fine-tuning is used broadly, including **SFT, PEFT/LoRA, preference optimization, and RL with spatially grounded rewards**.
* Benchmarks are discussed only when they illuminate **dataset design choices** rather than standalone evaluation.

## Why this matters

Spatial intelligence is foundational for embodied agents. Without reliable spatial grounding, language-level reasoning cannot translate into safe or effective action. By organizing the dataset landscape and clarifying design choices, this survey aims to:

* Guide **future dataset construction** for spatial reasoning,
* Support **data-centric model improvement**, and
* Enable more **robust, generalizable, and efficient spatial reasoning systems**.

## Intended audience

Researchers and practitioners working on:

* Vision-Language Models and Multimodal LLMs
* Spatial and geometric reasoning
* Robotics and embodied AI
* Dataset design, annotation pipelines, and evaluation

## Files in this folder

* [tasks_taxonomy.md](tasks_taxonomy.md): the full taxonomy (Figure 1), key definitions, task patterns, viewpoints and frames of reference (§2–3)
* [datasets.md](datasets.md): the 28 surveyed post-training datasets (Table 1), construction strategies, annotation approaches, modalities and source environments (§4)
* [benchmarks.md](benchmarks.md): evaluation practices (Table 2), scoring by answer type, generalization testing and referenced benchmarks (§5)
* [gaps_directions.md](gaps_directions.md): cross-dataset observations, evaluation open problems and future directions (§6)
* [papers.md](papers.md): the full reference list, grouped by role
