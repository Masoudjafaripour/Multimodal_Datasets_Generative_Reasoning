# vqa_prompt_templates.py
# Clean, minimal VQA prompt templates

from string import Template


TEMPLATES = {
    "basic_vqa": """You are given an image.
Question: ${QUESTION}
Answer the question based only on what is visible in the image.""",

    "grounded_vqa": """Look carefully at the image.
If the answer cannot be determined from the image alone, say "Not answerable from the image."
Question: ${QUESTION}""",

    "spatial_vqa": """Given the image, answer the question by reasoning about spatial relationships
(left/right, above/below, distance, containment).
Question: ${QUESTION}
Answer concisely.""",

    "reasoning_vqa": """Analyze the image step by step.
First describe the relevant visual elements.
Then reason about the question.
Finally, give the answer in one sentence.
Question: ${QUESTION}""",

    "structured_vqa": """Image → Objects → Relationships → Answer
Identify relevant objects and their relations before answering.
Question: ${QUESTION}""",

    "action_planning_vqa": """The image shows an environment.
Based on the visual scene, answer the question related to actions or outcomes.
Question: ${QUESTION}
Answer using only visual evidence.""",

    "yes_no_vqa": """Answer the following question with Yes or No only.
Question: ${QUESTION}""",

    "mcq_vqa": """Choose the correct answer based on the image.
Question: ${QUESTION}
Options:
A) ${A}
B) ${B}
C) ${C}
D) ${D}
Answer with the letter only.""",

    "calibrated_vqa": """Answer the question based on the image.
Then report your confidence as a number between 0 and 1.
Question: ${QUESTION}
Format: Answer | Confidence""",

    "json_vqa": """Given the image, generate a VQA example in JSON format:
{
  "question": "${QUESTION}",
  "answer": "...",
  "reasoning_type": "spatial / counting / attribute / action"
}"""
}


def render(template_name, **kwargs):
    """
    Render a VQA prompt by name.

    Example:
        render(
            "mcq_vqa",
            QUESTION="What color is the cube?",
            A="Red",
            B="Blue",
            C="Green",
            D="Yellow"
        )
    """
    if template_name not in TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}")

    try:
        return Template(TEMPLATES[template_name]).substitute(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing placeholder: {e.args[0]}")


# from vqa_prompt_templates import render

prompt = render(
    "spatial_vqa",
    QUESTION="Is the red cube to the left of the blue sphere?"
)

print(prompt)
