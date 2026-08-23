# 🌿 SkinSense AI — Complete Technical Project Documentation & Architecture Guide

---

## 📌 Executive Summary

**SkinSense AI** is an end-to-end, domain-adapted LLM application designed to provide reliable, educational skincare guidance, active ingredient pairing analysis, and symptom-to-condition differential analysis. 

The project addresses two critical challenges in consumer AI applications:
1. **Domain Hallucination:** Standard general-purpose LLMs frequently substitute ingredients or misstate safety guidelines when handling active compounds (e.g., combining Retinol and Glycolic Acid).
2. **Edge Deployment Efficiency:** Running large language models on local hardware with low latency and minimal RAM overhead.

---

## 🏛️ System Architecture Overview

The system follows a 4-tier decoupled architecture:

```
┌────────────────────────────────────────────────────────────────────────┐
│                          1. Frontend Tier                              │
│         HTML5 / TailwindCSS / ES6 JavaScript / Google OAuth 2.0         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / JSON
┌───────────────────────────────────▼────────────────────────────────────┐
│                       2. Backend & API Tier                            │
│           FastAPI (Python) / Session Management / CORS                 │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
                    ▼                                ▼
┌──────────────────────────────────────┐ ┌───────────────────────────────┐
│ 3. Automated Guardrails Layer        │ │ 4. LLM Inference Server       │
│ validate_response.py + skincare_kb   │ │ mlx_lm.server                 │
│ Ground-Truth Verification & Filter   │ │ Qwen2.5-3B-Instruct (4-bit)   │
└──────────────────────────────────────┘ └───────────────────────────────┘
```

---

## 🔬 Phase 1: Knowledge Base & Dataset Engineering

### 1. Ground-Truth Knowledge Base (`skincare_kb.json`)
Before fine-tuning, a curated knowledge base was constructed covering 20 canonical skincare active ingredients (e.g., *Niacinamide, Salicylic Acid, Retinol, Glycolic Acid, Ascorbic Acid, Azelaic Acid, Ceramides, Centella Asiatica*). 

Each entry maps:
- **Canonical Name & Aliases** (e.g., "vitamin c", "l-ascorbic acid")
- **Mechanism of Action ("does")** and **Target Skin Concerns ("helps")**
- **Conflict Classes:** `gentle_universal`, `strong_exfoliant_night`, `ph_sensitive_am`.
- **Explicit Pair Relationships:** Rules specifying `safe_together`, `alternate_nights`, or `sequential` (AM vs. PM).

### 2. Dataset Synthesis & Preparation Pipeline
- **`generate_from_kaggle.py`**: Extracted real-world product usage distributions across skincare product types.
- **`prepare_dataset.py`**: Formatted multi-turn chat dialogues into Qwen chat templates (`<|im_start|>user ... <|im_start|>assistant ...`) and converted raw dialogues into tokenized JSONL training files (`data/train.jsonl` and `data/valid.jsonl`).

---

## 🧠 Phase 2: Domain Adaptation via MLX LoRA Fine-Tuning

### 1. Model Selection
- **Base Model:** `mlx-community/Qwen2.5-3B-Instruct-4bit`
- **Why Qwen2.5 3B?** Outstanding reasoning performance at a compact 3-billion parameter footprint, making it ideal for real-time local inference on Apple Silicon hardware (M-series chips).

### 2. Parameter-Efficient Fine-Tuning (LoRA)
Rather than updating all 3 billion parameters (which is computationally expensive), **Low-Rank Adaptation (LoRA)** was used:
- **Rank ($r$):** 8
- **Scale ($\alpha$):** 20.0
- **Max Sequence Length:** 512 tokens
- **Target Layers:** 8 attention projection layers
- **Batch Size:** 1 with gradient accumulation

### 3. Model Fusing (`skincare-model-fused-v6`)
After training LoRA adapters, the low-rank weight matrices were merged directly back into the base model weights using `mlx_lm.fuse`. This created a standalone, fully fused model directory ready for high-speed inference without runtime adapter loading overhead.

---

## 🛡️ Phase 3: Automated Validation Guardrails Layer

To ensure safety and reliability in medical/skincare advice, an independent verification script (`validate_response.py`) acts as a real-time guardrail:

1. **Ingredient Substitution Detection:**
   - Extracts all ingredients mentioned in the user prompt ($Q_{ings}$) and assistant response ($A_{ings}$).
   - Catches if the model substitutes a queried ingredient with a different, similar-sounding ingredient from training data.

2. **Relationship Contradiction Check:**
   - Evaluates the model's advice against `skincare_kb.json`.
   - If the knowledge base marks two actives as `alternate_nights` but the model output claims `safe_together`, the backend flags a warning.

3. **Response Deduplication:**
   - In `backend/app.py`, an automated text cleaner strips repetitive disclaimer sentences, keeping output crisp and direct.

---

## 💻 Phase 4: Backend & Frontend Web Integration

1. **FastAPI Backend (`backend/app.py`):**
   - Exposes `/api/chat` and `/api/ingredients` endpoints.
   - Forwards prompt payloads to `mlx_lm.server` (`http://localhost:8080/v1/chat/completions`).
   - Verifies Google OAuth 2.0 ID tokens cryptographically using `google.oauth2.id_token.verify_oauth2_token`.

2. **Responsive Web UI (`frontend/`):**
   - Built with HTML5, TailwindCSS, and ES6 JavaScript.
   - Provides real-time chat feeds, single/multi-turn conversation management in `localStorage`, and interactive ingredient lookup pages.

---

## 🎯 Interview Q&A Guide ("If Someone Asks What You Built")

### 1. The 30-Second Elevator Pitch
> *"I built SkinSense AI, a domain-adapted LLM application for safe skincare guidance. I fine-tuned a Qwen2.5 3B model locally on Apple Silicon using MLX and LoRA adapters on a custom skincare knowledge base. To prevent AI hallucinations, I built an automated guardrail layer in Python that verifies generated advice against ground-truth safety rules before serving it through a FastAPI backend."*

### 2. Technical Deep-Dive Questions & Answers

* **Q: Why did you fine-tune a model instead of just using Prompt Engineering / RAG?**
  * **A:** Fine-tuning via LoRA allowed the model to internalize specific response structures, direct actionability, and precise domain terminology. We also combined it with a ground-truth validation layer to get both speed and accuracy.

* **Q: How did you handle LLM hallucination?**
  * **A:** I built `validate_response.py`, which parses canonical ingredient entities in both the user query and model answer. It cross-references stated claims against `skincare_kb.json` to catch substitution bugs or conflicting routine recommendations.

* **Q: Why MLX over PyTorch / Ollama?**
  * **A:** MLX is Apple's native machine learning framework designed specifically for Apple Silicon unified memory architecture. It achieved over **55 tokens/sec** generation speed on 4-bit quantized weights with under 2.7 GB RAM footprint.

---

## 📌 Summary of Project Artifacts

| Component | File Path | Description |
| :--- | :--- | :--- |
| **Knowledge Base** | `skincare_kb.json` | 20 canonical ingredients & explicit pair rules |
| **Dataset Generator** | `generate_from_kaggle.py` | Data mining script from Skinsort product dataset |
| **Dataset Pipeline** | `prepare_dataset.py` | Tokenizer & Chat-template JSONL builder |
| **LoRA Config** | `adapters-v4/adapter_config.json` | Fine-tuning hyperparameters |
| **Guardrails** | `validate_response.py` | Ground-truth verification engine |
| **Backend API** | `backend/app.py` | FastAPI server & Google Auth handler |
| **Frontend UI** | `frontend/chat.html` | Responsive chat interface |
