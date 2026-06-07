"""
SFT fine-tuning of Qwen2.5-VL-3B-Instruct on 3DSRBench
Uses LoRA + 4-bit quantization (QLoRA) for memory efficiency.
Includes: loss plotting + before/after SFT evaluation.
"""

import gc
import re
from io import BytesIO

import requests
import torch
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

# ============================================================
# Config
# ============================================================
MODEL_ID        = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR      = "./qwen25vl_3dsrbench_lora"
MAX_SAMPLES     = None       # None = full dataset
EVAL_SAMPLES    = 200        # samples used for before/after eval
MAX_SEQ_LEN     = 512
EPOCHS          = 3
BATCH_SIZE      = 4
GRAD_ACCUM      = 4
LR              = 2e-4
WARMUP_RATIO    = 0.05
SAVE_STEPS      = 200
LOG_STEPS       = 10
MAX_NEW_TOKENS  = 16

# LoRA
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
LORA_TARGET     = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]

# ============================================================
# Quantization (QLoRA)
# ============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# ============================================================
# Shared helpers (identical to eval script)
# ============================================================
def load_image_from_url(url: str) -> Image.Image:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")

def get_options(row):
    opts = {}
    for letter in ["A", "B", "C", "D"]:
        val = row.get(letter)
        if val is not None and str(val).strip():
            opts[letter] = str(val).strip()
    return opts

def build_prompt(question: str, options: dict) -> str:
    opt_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    return (
        "You are solving a 3D spatial reasoning multiple-choice question from an image.\n"
        "Carefully reason about real-world 3D relations such as depth, height, orientation, "
        "and object-relative position. Do not rely only on 2D image layout.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{opt_text}\n\n"
        "Answer with only the single correct option letter."
    )

def parse_letter(text: str, valid_letters):
    text = text.strip().upper()
    if text in valid_letters:
        return text
    m = re.search(r"\b([ABCD])\b", text)
    if m and m.group(1) in valid_letters:
        return m.group(1)
    m = re.search(r"ANSWER\s*[:\-]?\s*([ABCD])", text)
    if m and m.group(1) in valid_letters:
        return m.group(1)
    return None

# ============================================================
# Dataset
# ============================================================
class SRBenchDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row     = self.ds[idx]
        options = get_options(row)
        prompt  = build_prompt(row["question"], options)
        answer  = str(row["answer"]).strip().upper()
        try:
            image = load_image_from_url(row["image_url"])
        except Exception:
            image = Image.new("RGB", (224, 224))
        return {"image": image, "prompt": prompt, "answer": answer}


# ============================================================
# Collator
# ============================================================
class QwenSFTCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        from qwen_vl_utils import process_vision_info

        texts, images = [], []
        for item in batch:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["image"]},
                        {"type": "text",  "text": item["prompt"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": item["answer"]}],
                },
            ]
            full_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            input_text = self.processor.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            )
            texts.append((full_text, input_text))
            images.append(item["image"])

        full_texts  = [t[0] for t in texts]
        input_texts = [t[1] for t in texts]

        all_messages = [
            [{"role": "user",
              "content": [{"type": "image", "image": img},
                          {"type": "text",  "text": ""}]}]
            for img in images
        ]
        image_inputs, _ = process_vision_info(all_messages)

        encoding = self.processor(
            text=full_texts,
            images=image_inputs,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )
        input_encoding = self.processor(
            text=input_texts,
            images=image_inputs,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )

        labels = encoding["input_ids"].clone()
        input_lens = input_encoding["attention_mask"].sum(dim=1)
        for i, ilen in enumerate(input_lens):
            labels[i, :ilen] = -100
        labels[encoding["attention_mask"] == 0] = -100

        encoding["labels"] = labels
        return encoding


# ============================================================
# Evaluation helper — works for both base and fine-tuned model
# ============================================================
@torch.no_grad()
def run_eval(model, processor, eval_ds, n_samples, label=""):
    """
    Returns (accuracy, per_category_acc dict).
    Mirrors the predict logic from QwenVLEvaluator in the eval script.
    """
    from qwen_vl_utils import process_vision_info

    model.eval()
    correct = 0
    total   = 0
    per_cat = {}   # category -> [correct, total]

    n = min(n_samples, len(eval_ds))
    print(f"\n--- Evaluating [{label}] on {n} samples ---")

    for i in range(n):
        row = eval_ds[i]
        try:
            image    = load_image_from_url(row["image_url"])
            options  = get_options(row)
            gold     = str(row["answer"]).strip().upper()
            prompt   = build_prompt(row["question"], options)
            category = row.get("category", "unknown")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text",  "text": prompt},
                    ],
                }
            ]
            chat_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[chat_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            generated_ids = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
            )
            new_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
            output_text = processor.batch_decode(
                new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            pred       = parse_letter(output_text, list(options.keys()))
            is_correct = int(pred == gold) if pred is not None else 0

            correct += is_correct
            total   += 1
            if category not in per_cat:
                per_cat[category] = [0, 0]
            per_cat[category][0] += is_correct
            per_cat[category][1] += 1

        except Exception as e:
            print(f"  Eval error on sample {i}: {e}")
            total += 1
            if row.get("category", "unknown") not in per_cat:
                per_cat[row.get("category", "unknown")] = [0, 0]
            per_cat[row.get("category", "unknown")][1] += 1

        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"  [{i+1}/{n}] running acc = {correct/total:.4f}")

    acc = correct / total if total > 0 else 0.0
    print(f"  >> [{label}] Final accuracy: {acc:.4f}")
    return acc, {cat: v[0]/v[1] for cat, v in per_cat.items() if v[1] > 0}


# ============================================================
# Plotting helpers
# ============================================================
def plot_loss_curves(train_losses, val_losses, save_path="loss_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Train loss (per log step) ---
    axes[0].plot(train_losses, color="#2196F3", linewidth=1.5, label="Train loss")
    axes[0].set_xlabel("Optimizer step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Val loss (per epoch) ---
    epochs = list(range(1, len(val_losses) + 1))
    axes[1].plot(epochs, val_losses, color="#F44336", linewidth=2,
                 marker="o", markersize=7, label="Val loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Validation Loss per Epoch")
    axes[1].set_xticks(epochs)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss plot saved → {save_path}")


def plot_performance_comparison(before_acc, after_acc,
                                before_cat, after_cat,
                                save_path="performance_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Overall accuracy bar ---
    labels = ["Before SFT", "After SFT"]
    values = [before_acc * 100, after_acc * 100]
    colors = ["#90CAF9", "#1565C0"]
    bars   = axes[0].bar(labels, values, color=colors, width=0.4, edgecolor="black")
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Overall Accuracy: Before vs After SFT")
    delta = values[1] - values[0]
    sign  = "+" if delta >= 0 else ""
    axes[0].text(0.5, 0.92, f"Δ = {sign}{delta:.1f}%",
                 transform=axes[0].transAxes,
                 ha="center", fontsize=12, color="green" if delta >= 0 else "red",
                 fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    # --- Per-category grouped bar ---
    categories = sorted(set(list(before_cat.keys()) + list(after_cat.keys())))
    x          = range(len(categories))
    w          = 0.35
    b_vals     = [before_cat.get(c, 0) * 100 for c in categories]
    a_vals     = [after_cat.get(c, 0)  * 100 for c in categories]

    axes[1].bar([i - w/2 for i in x], b_vals, width=w, label="Before SFT",
                color="#90CAF9", edgecolor="black")
    axes[1].bar([i + w/2 for i in x], a_vals, width=w, label="After SFT",
                color="#1565C0", edgecolor="black")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(categories, rotation=30, ha="right", fontsize=9)
    axes[1].set_ylim(0, 110)
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Per-Category Accuracy: Before vs After SFT")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Performance comparison plot saved → {save_path}")


# ============================================================
# Load dataset
# ============================================================
print("Loading dataset...")
raw   = load_dataset("ccvl/3DSRBench")
split = "test" if "test" in raw else list(raw.keys())[0]
ds    = raw[split]
if MAX_SAMPLES:
    ds = ds.select(range(min(MAX_SAMPLES, len(ds))))

split_ds   = ds.train_test_split(test_size=0.1, seed=42)
train_data = split_ds["train"]
val_data   = split_ds["test"]

# Hold out a fixed eval slice (from val, never seen during training)
eval_n  = min(EVAL_SAMPLES, len(val_data))
eval_ds = val_data.select(range(eval_n))

# ============================================================
# Load model + LoRA
# ============================================================
print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=bnb_config,
)

# ============================================================
# *** BEFORE-SFT EVALUATION ***
# ============================================================
print("\n" + "="*60)
print("  PRE-SFT EVALUATION")
print("="*60)
before_acc, before_cat = run_eval(model, processor, eval_ds, eval_n, label="Before SFT")

# ============================================================
# Attach LoRA for training
# ============================================================
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET,
    task_type=TaskType.CAUSAL_LM,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.enable_input_require_grads()
model.gradient_checkpointing_enable()

# ============================================================
# DataLoaders
# ============================================================
collator     = QwenSFTCollator(processor)
train_loader = DataLoader(SRBenchDataset(train_data), batch_size=BATCH_SIZE,
                          shuffle=True,  collate_fn=collator, num_workers=2)
val_loader   = DataLoader(SRBenchDataset(val_data),   batch_size=BATCH_SIZE,
                          shuffle=False, collate_fn=collator, num_workers=2)

# ============================================================
# Optimizer + scheduler
# ============================================================
optimizer    = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
total_steps  = (len(train_loader) // GRAD_ACCUM) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

# ============================================================
# Training loop
# ============================================================
device       = next(model.parameters()).device
global_step  = 0
train_losses = []   # logged every LOG_STEPS optimizer steps
val_losses   = []   # logged every epoch
running_loss = 0.0

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        batch   = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        outputs = model(**batch)
        loss    = outputs.loss / GRAD_ACCUM
        loss.backward()
        running_loss += loss.item()

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % LOG_STEPS == 0:
                avg = running_loss * GRAD_ACCUM / LOG_STEPS
                train_losses.append(avg)
                print(f"Epoch {epoch+1} | step {global_step} | "
                      f"loss {avg:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")
                running_loss = 0.0

            if global_step % SAVE_STEPS == 0:
                model.save_pretrained(f"{OUTPUT_DIR}/step-{global_step}")
                processor.save_pretrained(f"{OUTPUT_DIR}/step-{global_step}")

    # --- Validation loss ---
    model.eval()
    v_loss, v_steps = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch   = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            v_loss += model(**batch).loss.item()
            v_steps += 1
    epoch_val_loss = v_loss / v_steps
    val_losses.append(epoch_val_loss)
    print(f">>> Epoch {epoch+1} val loss: {epoch_val_loss:.4f}")

# ============================================================
# Save final adapter + loss plot
# ============================================================
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"\nSaved LoRA adapter → {OUTPUT_DIR}")

plot_loss_curves(train_losses, val_losses, save_path=f"{OUTPUT_DIR}/loss_curves.png")

# ============================================================
# *** AFTER-SFT EVALUATION ***
# ============================================================
print("\n" + "="*60)
print("  POST-SFT EVALUATION")
print("="*60)
after_acc, after_cat = run_eval(model, processor, eval_ds, eval_n, label="After SFT")

# ============================================================
# Performance comparison plot + summary
# ============================================================
plot_performance_comparison(
    before_acc, after_acc,
    before_cat, after_cat,
    save_path=f"{OUTPUT_DIR}/performance_comparison.png",
)

print("\n" + "="*60)
print("  FINAL SUMMARY")
print("="*60)
print(f"  Before SFT accuracy : {before_acc:.4f}  ({before_acc*100:.1f}%)")
print(f"  After  SFT accuracy : {after_acc:.4f}  ({after_acc*100:.1f}%)")
delta = after_acc - before_acc
sign  = "+" if delta >= 0 else ""
print(f"  Delta               : {sign}{delta:.4f}  ({sign}{delta*100:.1f}%)")
print()
print("  Per-category breakdown:")
all_cats = sorted(set(list(before_cat.keys()) + list(after_cat.keys())))
print(f"  {'Category':<30} {'Before':>8} {'After':>8} {'Δ':>8}")
print(f"  {'-'*56}")
for cat in all_cats:
    b = before_cat.get(cat, 0.0)
    a = after_cat.get(cat, 0.0)
    d = a - b
    s = "+" if d >= 0 else ""
    print(f"  {cat:<30} {b*100:>7.1f}% {a*100:>7.1f}% {s}{d*100:>6.1f}%")
print("="*60)