# Multimodal_Datasets_Generative_Reasoning

This repository provides a **minimal, data-centric reference** for building, extending, and analyzing datasets for generative reasoning in multimodal large language models (MLLMs).

## Purpose

The goal is not to introduce a new benchmark, but to **operationalize common dataset construction patterns** observed across recent literature—including synthetic generation, automatic annotation, curation, and evaluation—especially for spatial and visual reasoning tasks.

## What’s Inside

* **data/**: raw assets, generated QA, curated datasets, and train/val/test splits
* **prompts/**: reusable LLM prompt templates for VQA generation, spatial relations, and quality checks
* **scripts/**: lightweight Python utilities for auto-generation, annotation, filtering, merging, and splitting
* **notebooks/**: exploratory analysis, prompt iteration, and dataset quality inspection
* **eval/**: sanity checks and simple baseline evaluations

## Scope

This repo serves as a **companion artifact** to a survey on datasets and benchmarks for multimodal reasoning. It is intentionally modular, lightweight, and model-agnostic, designed to help researchers translate survey insights into reproducible dataset pipelines.

## Intended Use

* Building datasets from scratch (synthetic or simulated)
* Curating or extending existing multimodal datasets
* Prototyping spatial or reasoning-focused VQA data

This repository is educational and illustrative by design, emphasizing clarity and reproducibility over scale.
