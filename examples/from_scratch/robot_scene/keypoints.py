"""Keypoint visualizations built on top of segment.py's output.

1. Generic corner keypoints (OpenCV Shi-Tomasi) over the whole image.
2. Per-object keypoints (centroid + extreme points) from each object's SAM mask.
3. Robot-arm "joint" keypoints via skeletonization of its mask (no pose model
   exists for this robot, so joints are approximated as skeleton endpoints /
   branch points of the arm's silhouette).
"""
import json
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.morphology import skeletonize
from transformers import SamModel, SamProcessor

IMG_PATH = "robot_scene.png"
SEGMENTS_PATH = "segments.json"

image = Image.open(IMG_PATH).convert("RGB")
segments = json.load(open(SEGMENTS_PATH))

# ---- recompute masks for the saved boxes (segment.py doesn't persist them) ----
device = "cuda" if torch.cuda.is_available() else "cpu"
sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

boxes = [s["bbox_xyxy"] for s in segments]
inputs = sam_processor(image, input_boxes=[boxes], return_tensors="pt").to(device)
with torch.no_grad():
    outputs = sam_model(**inputs)
pp_masks = sam_processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)[0]  # (num_boxes, 3, H, W) bool
best = outputs.iou_scores[0].argmax(dim=-1).cpu()
masks = [pp_masks[i, best[i]].numpy() for i in range(len(boxes))]

# ---- 1. generic corner keypoints ----
gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
corners = cv2.goodFeaturesToTrack(gray, maxCorners=150, qualityLevel=0.02, minDistance=8)
corner_img = image.copy()
draw = ImageDraw.Draw(corner_img)
for c in corners.reshape(-1, 2):
    x, y = c
    draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline="red", width=2)
corner_img.save("keypoints_corners.png")

# ---- 2. per-object keypoints: centroid + 4 extreme points ----
obj_img = image.copy()
draw = ImageDraw.Draw(obj_img)
for seg, mask in zip(segments, masks):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        continue
    cx, cy = xs.mean(), ys.mean()
    top = (xs[ys.argmin()], ys.min())
    bottom = (xs[ys.argmax()], ys.max())
    left = (xs.min(), ys[xs.argmin()])
    right = (xs.max(), ys[xs.argmax()])
    draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill="yellow", outline="black")
    for px, py in [top, bottom, left, right]:
        draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill="cyan", outline="black")
    draw.text((cx + 6, cy - 6), seg["label"], fill="yellow")
obj_img.save("keypoints_objects.png")

# ---- 3. robot-arm joint keypoints via skeletonization ----
arm_img = image.copy()
draw = ImageDraw.Draw(arm_img)
for seg, mask in zip(segments, masks):
    if seg["label"] != "robot arm":
        continue
    skeleton = skeletonize(mask)
    ys, xs = np.where(skeleton)
    pts = np.stack([xs, ys], axis=1)
    for x, y in pts:
        draw.point((x, y), fill="lime")
    # neighbor count per skeleton pixel -> endpoints (==1) and branch points (>=3)
    sk = skeleton.astype(np.uint8)
    neighbor_count = cv2.filter2D(sk, -1, np.ones((3, 3), np.uint8)) * sk - sk
    for y, x in zip(*np.where((neighbor_count == 1) | (neighbor_count >= 3))):
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], outline="red", width=2)
arm_img.save("keypoints_robot_arm.png")

print("Saved keypoints_corners.png, keypoints_objects.png, keypoints_robot_arm.png")
