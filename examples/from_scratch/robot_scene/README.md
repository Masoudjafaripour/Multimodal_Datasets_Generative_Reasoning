# Robot Scene VQA — Segmentation Step

Open-vocabulary segmentation of `robot_scene.png` as the first stage of a VQA
data-generation pipeline: detect objects by text prompt, mask them, and save
labeled bounding boxes for downstream question/answer generation.

## Pipeline (`segment.py`)

1. **[OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble)** — zero-shot object detection from a text label list (`LABELS`).
2. **[SAM](https://huggingface.co/facebook/sam-vit-base)** — converts each detected box into a precise segmentation mask.
3. **Dedup** — per-label NMS + containment check to drop overlapping duplicate boxes.
4. **Save outputs**:
   - `segmented_overlay.png` — colored mask overlay per object
   - `segmented_bbox.png` — bounding boxes + labels drawn on the image
   - `segments.json` — `[{label, score, bbox_xyxy}, ...]`, input for VQA generation

## Run

```bash
source ../../../mm_data_venv_linux/bin/activate
pip install -q scipy   # first run only
python segment.py
```

## Config

Edit the top of `segment.py`:
- `IMG_PATH` — input image
- `LABELS` — candidate object names to detect (edit for your scene)
- `SCORE_THRESH` — min detection confidence (default `0.18`)
- `IOU_THRESH` — NMS overlap threshold (default `0.3`)

## Next step

`vqa_gen.py` (not yet implemented) will consume `segments.json` to generate
Q&A pairs about object identity, position, and spatial relations.
