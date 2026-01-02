import os

ROOT = "Multimodal_Datasets_Generative_Reasoning"

structure = {
    "": ["README.md", "requirements.txt"],

    "survey": [
        "papers.md",
        "datasets.md",
        "benchmarks.md",
        "tasks_taxonomy.md",
        "gaps_directions.md",
    ],

    "examples/from_scratch": [],
    "examples/extend_existing": [],
    "examples/curate_existing": [],

    "data/raw": [],
    "data/generated": [],
    "data/curated": [],
    "data/splits": [],

    "prompts": [
        "vqa_generation.txt",
        "spatial_reasoning.txt",
        "quality_control.txt",
    ],

    "pipelines": [
        "generate_vqa.py",
        "auto_annotate.py",
        "curate_filter.py",
        "merge_extend.py",
        "split_dataset.py",
    ],

    "notebooks": [
        "01_dataset_overview.ipynb",
        "02_prompt_iteration.ipynb",
        "03_quality_analysis.ipynb",
    ],

    "evaluation": [
        "sanity_checks.py",
        "baseline_eval.py",
    ],
}

def create_repo(root, structure):
    for folder, files in structure.items():
        folder_path = os.path.join(root, folder)
        os.makedirs(folder_path, exist_ok=True)

        for f in files:
            file_path = os.path.join(folder_path, f)
            if not os.path.exists(file_path):
                with open(file_path, "w") as fp:
                    fp.write("")  # empty placeholder

if __name__ == "__main__":
    create_repo(ROOT, structure)
    print(f"Repository structure '{ROOT}' created successfully.")
