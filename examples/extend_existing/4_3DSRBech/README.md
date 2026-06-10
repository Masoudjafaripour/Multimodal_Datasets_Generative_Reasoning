# Qwen2.5-VL-3B Fine-tuning on 3DSRBench

Fine-tuning pipeline for **3D spatial reasoning** using Qwen2.5-VL-3B-Instruct on [3DSRBench](https://huggingface.co/datasets/ccvl/3DSRBench), combining Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO).

---

## Results

| Stage | Accuracy | Δ |
|---|---|---|
| Base model | 55.5% | — |
| After SFT | 78.0% | +22.5% |
| After GRPO | 90.0% | +12.0% |

---

## Stage 1 — Supervised Fine-Tuning (SFT)

**Script:** `SFT_Qwen3B_3DSRBench.py`

### Method

QLoRA fine-tuning: the base model is frozen in 4-bit NF4 quantization, and only the LoRA adapter weights $\Delta W = BA$ are trained, where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, rank $r \ll \min(d,k)$.

The training objective is standard causal language modelling loss over the **answer token only** (prompt tokens are masked with $-100$):

$$\mathcal{L}_{\text{SFT}} = -\log p_\theta(y^* \mid x, v)$$

where $y^*$ is the correct option letter, $x$ is the question + options text, and $v$ is the image.

### Key config

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Quantization | 4-bit NF4 (bfloat16 compute) |
| LoRA rank $r$ | 16 |
| LoRA $\alpha$ | 32 |
| LoRA targets | `q,k,v,o,gate,up,down_proj` |
| Learning rate | 1e-4 (cosine decay) |
| Batch size | 4 × grad accum 4 = 16 effective |
| Epochs | 3 |

### Output

Adapter saved to `./qwen25vl_3dsrbench_lora/`

---

## Stage 2 — Reinforcement Learning via GRPO

**Script:** `RLFT_Qwen3B_3DSRBench.py`

### Method

GRPO ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)) samples $G$ completions per prompt from the current policy $\pi_\theta$, scores them with a reward function $r$, and updates the policy by maximising the clipped group-relative advantage — without a separate critic model.

**Advantage estimate** for completion $i$ in group $g$:

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G)}$$

**GRPO objective** (with KL penalty against the SFT reference policy $\pi_{\text{ref}}$):

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\!\left[\min\!\left(\rho_i \hat{A}_i,\; \text{clip}(\rho_i, 1{-}\epsilon, 1{+}\epsilon)\hat{A}_i\right)\right] + \beta\, D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

where $\rho_i = \pi_\theta(o_i \mid q) / \pi_{\text{ref}}(o_i \mid q)$ is the importance ratio.

**Reward function** combines two signals:

$$r(o, y^*) = r_{\text{correct}}(o, y^*) + r_{\text{format}}(o)$$

$$r_{\text{correct}} = \begin{cases} +1.0 & \text{if } \texttt{parse}(o) = y^* \\ \phantom{+}0.0 & \text{if wrong but parseable} \\ -0.5 & \text{if unparseable} \end{cases}$$

$$r_{\text{format}} = \begin{cases} +0.2 & \text{if output matches } \texttt{Answer: X} \\ \phantom{+}0.0 & \text{otherwise} \end{cases}$$

### Key config

| Parameter | Value |
|---|---|
| Init checkpoint | `./qwen25vl_3dsrbench_lora/` (SFT adapter) |
| Generations $G$ | 8 per prompt |
| Learning rate | 5e-6 |
| Max new tokens | 64 (allows short CoT) |
| Batch size | 1 × grad accum 8 |
| Epochs | 2 |

### Output

Model saved to `./qwen25vl_3dsrbench_grpo/`

---

## Pipeline

```
Qwen2.5-VL-3B-Instruct (base)
        │
        ▼  Stage 1: SFT (QLoRA)
        │  • learns task format
        │  • loss on answer token only
        │
qwen25vl_3dsrbench_lora/
        │
        ▼  Stage 2: GRPO (RL)
        │  • explores reasoning traces
        │  • reward = correctness + format
        │
qwen25vl_3dsrbench_grpo/
```

---

## Requirements

```bash
pip install transformers peft trl datasets qwen-vl-utils bitsandbytes matplotlib
```

## Usage

```bash
# Stage 1 — SFT
python SFT_Qwen3B_3DSRBench.py

# Stage 2 — GRPO (requires SFT checkpoint)
python RLFT_Qwen3B_3DSRBench.py
```