# Understanding Spatial Reasoning in Multimodal Large Language Models

This repository accompanies our survey **“Understanding Spatial Reasoning in Multimodal Large Language Models through Data and Evaluation” (submitted to IJCAI 2026)**. The survey provides a **data-centric view** of how spatial reasoning capabilities in Vision-Language Models (VLMs) and Multimodal Large Language Models (MLLMs) are learned, evaluated, and limited by dataset design.

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


