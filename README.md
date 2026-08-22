# 🌿 SkinSense AI — Fine-Tuned Domain LLM for Skincare Guidance

SkinSense AI is an end-to-end domain-adapted LLM application designed to provide educational skincare guidance, active ingredient compatibility advice, and symptom differential analysis.

## 🚀 Key Features
- **Local LoRA Fine-Tuning:** Fine-tuned `Qwen2.5-3B-Instruct` on a custom skincare dataset using Apple Silicon **MLX**.
- **Response Validation Guardrails:** Integrated ground-truth verification layer (`validate_response.py`) to prevent active ingredient hallucinations.
- **Symptom & Compatibility Guidance:** Educational analysis for ingredient pairings (e.g., Retinol + Glycolic Acid) and symptom queries (e.g., Rosacea, acne spot treatment, barrier repair).
- **Full-Stack Architecture:** FastAPI REST API backend with Google OAuth 2.0 authentication and a responsive UI.

## 🏗️ Architecture
1. **Dataset Pipeline:** Ground-truth knowledge base (`skincare_kb.json`) $\rightarrow$ Chat-template formatter (`prepare_dataset.py`) $\rightarrow$ JSONL dataset.
2. **Fine-Tuning:** MLX-LM LoRA training on Apple Silicon (`adapters-v4`).
3. **Backend & Guardrails:** FastAPI server (`backend/app.py`) executing real-time output validation (`validate_response.py`).
4. **Frontend:** Custom HTML5 / JS / TailwindCSS UI with Google OAuth 2.0 login.

## 🛠️ Tech Stack
- **AI / ML:** MLX, MLX-LM, LoRA, Qwen2.5-3B-Instruct, HuggingFace Transformers
- **Backend:** Python 3.10+, FastAPI, Uvicorn, Pydantic, Google Auth API
- **Frontend:** HTML5, TailwindCSS, JavaScript (ES6+), Google Identity Services

## ⚙️ Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/skin-care.git
cd skin-care
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run MLX Model Server
```bash
mlx_lm.server --model skincare-model-fused-v6 --port 8080
```

### 4. Run FastAPI Web Server
```bash
python3 backend/app.py
```

Open `http://localhost:8000/chat` in your browser to interact with the application.
