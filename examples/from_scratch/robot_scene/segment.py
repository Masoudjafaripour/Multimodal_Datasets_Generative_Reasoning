"""Open-vocabulary segmentation: OWLv2 (text-prompt boxes) + SAM (box -> mask)."""
import json
import numpy as np
import torch
from torchvision.ops import nms
from PIL import Image, ImageDraw
from transformers import pipeline, SamModel, SamProcessor

IMG_PATH = "robot_scene.png"
LABELS = ["robot arm", "banana", "apple", "orange", "toy block", "plastic container",
          "sheet of paper", "table", "wall", "cable"]
SCORE_THRESH = 0.18
IOU_THRESH = 0.3

device = "cuda" if torch.cuda.is_available() else "cpu"
image = Image.open(IMG_PATH).convert("RGB")

detector = pipeline("zero-shot-object-detection", model="google/owlv2-base-patch16-ensemble",
                     device=0 if device == "cuda" else -1)
raw = [d for d in detector(image, candidate_labels=LABELS) if d["score"] >= SCORE_THRESH]

def dedupe(group):
    boxes = torch.tensor([[d["box"]["xmin"], d["box"]["ymin"], d["box"]["xmax"], d["box"]["ymax"]] for d in group], dtype=torch.float32)
    scores = torch.tensor([d["score"] for d in group])
    keep = nms(boxes, scores, IOU_THRESH).tolist()
    # drop boxes that are mostly contained inside a higher-scoring kept box (catches nested duplicates NMS misses)
    order = sorted(keep, key=lambda i: -scores[i])
    final = []
    for i in order:
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        contained = False
        for j in final:
            ix0, iy0 = max(boxes[i, 0], boxes[j, 0]), max(boxes[i, 1], boxes[j, 1])
            ix1, iy1 = min(boxes[i, 2], boxes[j, 2]), min(boxes[i, 3], boxes[j, 3])
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            if inter / area_i > 0.8:
                contained = True
                break
        if not contained:
            final.append(i)
    return [group[i] for i in final]


detections = []
for label in {d["label"] for d in raw}:
    detections.extend(dedupe([d for d in raw if d["label"] == label]))

sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

boxes = [[d["box"]["xmin"], d["box"]["ymin"], d["box"]["xmax"], d["box"]["ymax"]] for d in detections]
masks = []
if boxes:
    inputs = sam_processor(image, input_boxes=[boxes], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = sam_model(**inputs)
    pp_masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )[0]  # (num_boxes, 3, H, W) bool
    best = outputs.iou_scores[0].argmax(dim=-1).cpu()  # best of 3 masks per box
    masks = [pp_masks[i, best[i]].numpy() for i in range(len(boxes))]

rng = np.random.default_rng(0)
overlay = image.copy()
bbox_img = image.copy()
draw_bbox = ImageDraw.Draw(bbox_img)

segments = []
for det, mask in zip(detections, masks):
    color = tuple(int(c) for c in rng.integers(0, 255, 3))
    color_layer = Image.new("RGB", image.size, color)
    overlay = Image.composite(color_layer, overlay, Image.fromarray(mask))

    b = det["box"]
    draw_bbox.rectangle([b["xmin"], b["ymin"], b["xmax"], b["ymax"]], outline=color, width=3)
    draw_bbox.text((b["xmin"] + 2, max(0, b["ymin"] - 12)), det["label"], fill=color)

    segments.append({
        "label": det["label"],
        "score": float(det["score"]),
        "bbox_xyxy": [b["xmin"], b["ymin"], b["xmax"], b["ymax"]],
    })

Image.blend(image, overlay, alpha=0.5).save("segmented_overlay.png")
bbox_img.save("segmented_bbox.png")
with open("segments.json", "w") as f:
    json.dump(segments, f, indent=2)

print(f"Found {len(segments)} segments -> segmented_overlay.png, segmented_bbox.png, segments.json")
