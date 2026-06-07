"""
SFT fine-tuning of Qwen2.5-VL-3B-Instruct on 3DSRBench
Uses LoRA + 4-bit quantization (QLoRA) for memory efficiency.
"""

import gc
import re
from io import BytesIO

import requests
import torch
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

# ============================================================
# Config
# ============================================================
MODEL_ID        = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR      = "./qwen25vl_3dsrbench_lora"
MAX_SAMPLES     = None       # None = full dataset
MAX_SEQ_LEN     = 512
EPOCHS          = 3
BATCH_SIZE      = 4          # effective; use GRAD_ACCUM to scale
GRAD_ACCUM      = 4
LR              = 2e-4
WARMUP_RATIO    = 0.05
SAVE_STEPS      = 200
LOG_STEPS       = 10

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
# Dataset
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
            image = Image.new("RGB", (224, 224))  # fallback blank image
        return {"image": image, "prompt": prompt, "answer": answer}


# ============================================================
# Collator — builds chat-formatted input + labels
# ============================================================
class QwenSFTCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
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
            # Full conversation (input + target)
            full_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            # Input only (to find the answer span)
            input_text = self.processor.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            )
            texts.append((full_text, input_text))
            images.append(item["image"])

        # Tokenize full conversations
        full_texts   = [t[0] for t in texts]
        input_texts  = [t[1] for t in texts]

        from qwen_vl_utils import process_vision_info
        # Build message lists for process_vision_info (need images attached)
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

        # Build labels: mask everything up to (and including) the assistant prompt token
        input_encoding = self.processor(
            text=input_texts,
            images=image_inputs,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )

        labels = encoding["input_ids"].clone()
        input_lens = input_encoding["attention_mask"].sum(dim=1)  # actual token count
        for i, ilen in enumerate(input_lens):
            labels[i, :ilen] = -100   # mask prompt tokens
        labels[encoding["attention_mask"] == 0] = -100   # mask padding

        encoding["labels"] = labels
        return encoding


# ============================================================
# Model + LoRA
# ============================================================
print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=bnb_config,
)

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

# Required for gradient checkpointing with PEFT + quantization
model.enable_input_require_grads()
model.gradient_checkpointing_enable()

# ============================================================
# Data
# ============================================================
raw = load_dataset("ccvl/3DSRBench")
split = "test" if "test" in raw else list(raw.keys())[0]
ds = raw[split]
if MAX_SAMPLES:
    ds = ds.select(range(min(MAX_SAMPLES, len(ds))))

# 90/10 train/val split
split_ds  = ds.train_test_split(test_size=0.1, seed=42)
train_ds  = SRBenchDataset(split_ds["train"])
val_ds    = SRBenchDataset(split_ds["test"])

collator   = QwenSFTCollator(processor)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collator, num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collator, num_workers=2)

# ============================================================
# Optimizer + scheduler
# ============================================================
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
total_steps   = (len(train_loader) // GRAD_ACCUM) * EPOCHS
warmup_steps  = int(total_steps * WARMUP_RATIO)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

# ============================================================
# Training loop
# ============================================================
device = next(model.parameters()).device
global_step = 0

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    running_loss = 0.0

    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
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
                print(f"Epoch {epoch+1} | step {global_step} | "
                      f"loss {running_loss * GRAD_ACCUM / LOG_STEPS:.4f} | "
                      f"lr {scheduler.get_last_lr()[0]:.2e}")
                running_loss = 0.0

            if global_step % SAVE_STEPS == 0:
                model.save_pretrained(f"{OUTPUT_DIR}/step-{global_step}")
                processor.save_pretrained(f"{OUTPUT_DIR}/step-{global_step}")

    # ----- Validation -----
    model.eval()
    val_loss, val_steps = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            val_loss += model(**batch).loss.item()
            val_steps += 1
    print(f">>> Epoch {epoch+1} val loss: {val_loss / val_steps:.4f}")

# ============================================================
# Save final adapter
# ============================================================
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"Saved LoRA adapter to {OUTPUT_DIR}")