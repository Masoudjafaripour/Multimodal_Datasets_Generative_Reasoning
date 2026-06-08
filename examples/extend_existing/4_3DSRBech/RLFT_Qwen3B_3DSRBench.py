"""
GRPO fine-tuning of Qwen2.5-VL-3B-Instruct + SFT LoRA on 3DSRBench
Starts from SFT checkpoint, uses binary correctness reward.
Includes before/after evaluation and performance plots.
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
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig

# ============================================================
# Config
# ============================================================
BASE_MODEL_ID   = "Qwen/Qwen2.5-VL-3B-Instruct"
SFT_LORA_DIR    = "./qwen25vl_3dsrbench_lora"   # your SFT checkpoint
OUTPUT_DIR      = "./qwen25vl_3dsrbench_grpo"
MAX_SAMPLES     = 100      # None = full dataset
EVAL_SAMPLES    = 200
MAX_NEW_TOKENS  = 64        # longer than SFT to allow short CoT before answer
MAX_SEQ_LEN     = 2048

# GRPO hyperparams
EPOCHS          = 2
LR              = 5e-6
NUM_GENERATIONS = 8         # G rollouts per prompt
BATCH_SIZE      = 1         # per-device; GRPO is memory-heavy
GRAD_ACCUM      = 8
WARMUP_RATIO    = 0.05

# Rewards
REWARD_CORRECT    =  1.0
REWARD_WRONG      =  0.0
REWARD_MALFORMED  = -0.5
REWARD_FORMAT_BON =  0.2    # bonus if output is a clean single letter

# ============================================================
# Quantization
# ============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# ============================================================
# Shared helpers (identical to eval + SFT scripts)
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
        "Think step by step, then end your response with 'Answer: X' "
        "where X is the single correct option letter."
    )

def parse_letter(text, valid_letters):
    # Guard: ensure text is always a string
    if not isinstance(text, str):
        text = str(text)
    text_up = text.strip().upper()
    m = re.search(r"ANSWER\s*[:\-]?\s*([ABCD])", text_up)
    if m and m.group(1) in valid_letters:
        return m.group(1)
    m = re.search(r"\b([ABCD])\b", text_up)
    if m and m.group(1) in valid_letters:
        return m.group(1)
    if text_up in valid_letters:
        return text_up
    return None
# ============================================================
# Reward functions
# ============================================================
def correctness_reward(predictions: list[str], golds: list[str],
                       valid_letters_list: list[list[str]]) -> list[float]:
    rewards = []
    for pred_text, gold, valid in zip(predictions, golds, valid_letters_list):
        pred = parse_letter(pred_text, valid)
        if pred is None:
            rewards.append(REWARD_MALFORMED)
        elif pred == gold.strip().upper():
            rewards.append(REWARD_CORRECT)
        else:
            rewards.append(REWARD_WRONG)
    return rewards

def format_reward(predictions: list[str]) -> list[float]:
    """Bonus for clean 'Answer: X' format."""
    rewards = []
    for text in predictions:
        if re.search(r"ANSWER\s*[:\-]?\s*[ABCD]", text.strip().upper()):
            rewards.append(REWARD_FORMAT_BON)
        else:
            rewards.append(0.0)
    return rewards

# ============================================================
# Dataset — build HF-compatible format for GRPOTrainer
# GRPOTrainer expects: {"prompt": [...chat messages...], + any extra fields}
# We store gold + valid_letters as metadata for reward computation
# ============================================================
def build_grpo_dataset(hf_ds, processor):
    """Convert 3DSRBench rows → list of dicts with chat-format prompts."""
    samples = []
    for row in hf_ds:
        options = get_options(row)
        prompt  = build_prompt(row["question"], options)
        try:
            image = load_image_from_url(row["image_url"])
        except Exception:
            image = Image.new("RGB", (224, 224))

        # GRPOTrainer expects "prompt" as a list of chat messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": prompt},
                ],
            }
        ]
        samples.append({
            "prompt"        : messages,
            "gold"          : str(row["answer"]).strip().upper(),
            "valid_letters" : list(options.keys()),
            "category"      : row.get("category", "unknown"),
        })
    return samples

# ============================================================
# Evaluation helper
# ============================================================
@torch.no_grad()
def run_eval(model, processor, hf_ds, n_samples, label=""):
    from qwen_vl_utils import process_vision_info

    model.eval()
    correct, total = 0, 0
    per_cat = {}
    n = min(n_samples, len(hf_ds))
    print(f"\n--- Evaluating [{label}] on {n} samples ---")

    for i in range(n):
        row = hf_ds[i]
        try:
            image    = load_image_from_url(row["image_url"])
            options  = get_options(row)
            gold     = str(row["answer"]).strip().upper()
            prompt   = build_prompt(row["question"], options)
            category = row.get("category", "unknown")

            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt},
            ]}]
            chat_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[chat_text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            generated_ids = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
            )
            new_ids     = generated_ids[:, inputs["input_ids"].shape[1]:]
            output_text = processor.batch_decode(
                new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            pred       = parse_letter(output_text, list(options.keys()))
            is_correct = int(pred == gold) if pred is not None else 0
            correct   += is_correct
            total     += 1
            per_cat.setdefault(category, [0, 0])
            per_cat[category][0] += is_correct
            per_cat[category][1] += 1

        except Exception as e:
            print(f"  Eval error on sample {i}: {e}")
            total += 1
            cat = row.get("category", "unknown")
            per_cat.setdefault(cat, [0, 0])
            per_cat[cat][1] += 1

        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"  [{i+1}/{n}] running acc = {correct/total:.4f}")

    acc = correct / total if total > 0 else 0.0
    print(f"  >> [{label}] Final accuracy: {acc:.4f}")
    return acc, {c: v[0]/v[1] for c, v in per_cat.items() if v[1] > 0}

# ============================================================
# Plotting
# ============================================================
def plot_reward_curves(mean_rewards, save_path):
    plt.figure(figsize=(9, 4))
    plt.plot(mean_rewards, color="#E91E63", linewidth=1.5, label="Mean reward")
    # Rolling average
    window = min(20, len(mean_rewards) // 4 or 1)
    if len(mean_rewards) >= window:
        rolled = [
            sum(mean_rewards[max(0,i-window):i+1]) / len(mean_rewards[max(0,i-window):i+1])
            for i in range(len(mean_rewards))
        ]
        plt.plot(rolled, color="#880E4F", linewidth=2.5, linestyle="--", label=f"Rolling avg (w={window})")
    plt.xlabel("GRPO step")
    plt.ylabel("Mean reward")
    plt.title("GRPO Training — Mean Reward per Step")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Reward plot saved → {save_path}")

def plot_performance_comparison(accs: dict, cats: dict, save_path: str):
    """accs = {"SFT": 0.78, "GRPO": 0.84}  cats = {"SFT": {...}, "GRPO": {...}}"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    labels = list(accs.keys())
    values = [v * 100 for v in accs.values()]
    colors = ["#1565C0", "#E91E63"]

    bars = axes[0].bar(labels, values, color=colors, width=0.4, edgecolor="black")
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
    if len(values) == 2:
        delta = values[1] - values[0]
        sign  = "+" if delta >= 0 else ""
        axes[0].text(0.5, 0.92, f"Δ = {sign}{delta:.1f}%",
                     transform=axes[0].transAxes, ha="center", fontsize=12,
                     color="green" if delta >= 0 else "red", fontweight="bold")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Overall Accuracy: SFT vs GRPO")
    axes[0].grid(axis="y", alpha=0.3)

    all_cats = sorted(set(c for d in cats.values() for c in d))
    x, w = range(len(all_cats)), 0.35 / max(len(labels)-1, 1) * 2
    offsets = [-w/2, w/2] if len(labels) == 2 else [0]
    for idx, (label, color) in enumerate(zip(labels, colors)):
        vals = [cats[label].get(c, 0)*100 for c in all_cats]
        axes[1].bar([i + offsets[idx] for i in x], vals, width=w,
                    label=label, color=color, edgecolor="black", alpha=0.85)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(all_cats, rotation=30, ha="right", fontsize=9)
    axes[1].set_ylim(0, 110)
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Per-Category Accuracy: SFT vs GRPO")
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
eval_ds    = val_data.select(range(min(EVAL_SAMPLES, len(val_data))))

# ============================================================
# Load base model + SFT LoRA
# ============================================================
print(f"\nLoading base model: {BASE_MODEL_ID}")
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config,
)

print(f"Loading SFT LoRA from: {SFT_LORA_DIR}")
model = PeftModel.from_pretrained(base_model, SFT_LORA_DIR, is_trainable=True)
model.print_trainable_parameters()

# ============================================================
# PRE-GRPO evaluation (= post-SFT baseline)
# ============================================================
print("\n" + "="*60)
print("  PRE-GRPO EVALUATION  (SFT checkpoint)")
print("="*60)
sft_acc, sft_cat = run_eval(model, processor, eval_ds, EVAL_SAMPLES, label="SFT")

# ============================================================
# Build GRPO dataset
# ============================================================
print("\nBuilding GRPO dataset...")
grpo_train = build_grpo_dataset(train_data, processor)
grpo_val   = build_grpo_dataset(val_data,   processor)
print(f"  Train: {len(grpo_train)}  |  Val: {len(grpo_val)}")

# ============================================================
# Reward wrapper for GRPOTrainer
# GRPOTrainer calls: reward_fn(prompts, completions, **batch_extras)
# batch_extras contains all extra keys from dataset ("gold", "valid_letters")
# ============================================================
def combined_reward_fn(prompts, completions, gold, valid_letters, **kwargs):
    """
    GRPOTrainer passes completions as List[List[str]] — one inner list per prompt,
    with NUM_GENERATIONS completions each. We need to flatten + tile gold/valid_letters.
    """
    # Flatten completions: [[g1,g2,...], [g1,g2,...]] → [g1,g2,...,g1,g2,...]
    flat_completions = []
    flat_gold        = []
    flat_valid       = []

    for i, comp_group in enumerate(completions):
        # comp_group may itself be a list of dicts with a "content" key (newer trl)
        # or a plain list of strings — handle both
        for comp in comp_group:
            if isinstance(comp, dict):
                text = comp.get("content", "") or ""
            elif isinstance(comp, list):
                # list of content blocks
                text = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in comp
                )
            else:
                text = str(comp)

            flat_completions.append(text)
            flat_gold.append(gold[i])
            flat_valid.append(valid_letters[i])

    c_rewards = correctness_reward(flat_completions, flat_gold, flat_valid)
    f_rewards = format_reward(flat_completions)
    return [c + f for c, f in zip(c_rewards, f_rewards)]
# ============================================================
# GRPO training
# ============================================================
grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    warmup_ratio=WARMUP_RATIO,
    num_generations=NUM_GENERATIONS,
    max_completion_length=MAX_NEW_TOKENS,
    logging_steps=10,
    save_steps=100,
    bf16=True,
    remove_unused_columns=False,    # keep "gold" and "valid_letters" in batch
    report_to="none",
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=combined_reward_fn,
    args=grpo_config,
    train_dataset=grpo_train,
    eval_dataset=grpo_val,
    processing_class=processor,
)

print("\n" + "="*60)
print("  STARTING GRPO TRAINING")
print("="*60)
train_result = trainer.train()

# Save final model
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"\nSaved GRPO model → {OUTPUT_DIR}")

# ============================================================
# Extract reward history from trainer logs for plotting
# ============================================================
mean_rewards = []
if hasattr(trainer, "state") and trainer.state.log_history:
    for entry in trainer.state.log_history:
        if "reward" in entry:
            mean_rewards.append(entry["reward"])

if mean_rewards:
    plot_reward_curves(mean_rewards, save_path=f"{OUTPUT_DIR}/reward_curve.png")

# ============================================================
# POST-GRPO evaluation
# ============================================================
print("\n" + "="*60)
print("  POST-GRPO EVALUATION")
print("="*60)
grpo_acc, grpo_cat = run_eval(model, processor, eval_ds, EVAL_SAMPLES, label="GRPO")

# ============================================================
# Performance comparison plot + summary
# ============================================================
plot_performance_comparison(
    accs={"SFT": sft_acc, "GRPO": grpo_acc},
    cats={"SFT": sft_cat, "GRPO": grpo_cat},
    save_path=f"{OUTPUT_DIR}/performance_comparison.png",
)

print("\n" + "="*60)
print("  FINAL SUMMARY")
print("="*60)
print(f"  SFT  accuracy  : {sft_acc:.4f}  ({sft_acc*100:.1f}%)")
print(f"  GRPO accuracy  : {grpo_acc:.4f}  ({grpo_acc*100:.1f}%)")
delta = grpo_acc - sft_acc
sign  = "+" if delta >= 0 else ""
print(f"  Delta          : {sign}{delta:.4f}  ({sign}{delta*100:.1f}%)")
print()
print(f"  {'Category':<30} {'SFT':>8} {'GRPO':>8} {'Δ':>8}")
print(f"  {'-'*56}")
all_cats = sorted(set(list(sft_cat.keys()) + list(grpo_cat.keys())))
for cat in all_cats:
    b = sft_cat.get(cat,  0.0)
    a = grpo_cat.get(cat, 0.0)
    d = a - b
    s = "+" if d >= 0 else ""
    print(f"  {cat:<30} {b*100:>7.1f}% {a*100:>7.1f}% {s}{d*100:>6.1f}%")
print("="*60)