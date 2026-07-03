# TutorBuddy — Project Context for OpenCode

> Read this file at the start of every session. This is the single source of truth for the project. Do not ask for context that is already here.

---

## What This Project Is

TutorBuddy is an AI education agent for Grade 4–8 students. It helps with homework questions, lesson summaries, quiz generation, and optional worksheet OCR. It supports both English and Urdu output.

This is a Kaggle 5-Day AI Intensive capstone submission. The final deliverable is a polished Kaggle notebook with a working Gradio demo, RAG pipeline, evaluation results, and a clear writeup.

---

## Tech Stack

| Layer | Tool / Library                              |
|---|---------------------------------------------|
| LLM | Gemini 2.5 Flash via `google-genai`         |
| Embeddings | `sentence-transformers` — model `all-MiniLM-L6-v2` |
| Vector store | `faiss-cpu`                                 |
| OCR | `pytesseract` + `opencv-python` + `Pillow`  |
| UI | `Gradio`                                    |
| Data | `pandas`, `numpy`                           |
| Evaluation | `scikit-learn`, custom scripts              |
| Env management | `python-dotenv`                             |
| Testing | `pytest`                                    |
| Dev environment | Local machine — Arch Linux, Python venv     |
| Final submission | Kaggle notebook                             |

---

## Project Structure

```
TutorBuddy/
├── Capstone_TutorBuddy.ipynb   ← final Kaggle submission
├── AGENT.md                    ← this file
├── README.md
├── requirements.txt
├── Dockerfile
├── .env                        ← GEMINI_API_KEY lives here, never commit this
├── .gitignore
├── assets/
│   ├── cover.png
│   └── demo.mp4
├── app/
│   └── main.py                 ← Gradio UI
├── src/
│   ├── agent/
│   │   ├── orchestrator.py     ← TutorAgent class, calls Gemini
│   │   └── retriever.py        ← CurriculumRetriever class, FAISS search
│   ├── ocr/
│   │   └── ocr_pipeline.py     ← extract_text_from_image()
│   └── evaluation/
│       └── metrics.py          ← eval runner, saves results.csv
├── data/
│   ├── curriculum/
│   │   └── notes.json          ← knowledge base (30 notes)
│   ├── eval/
│   │   ├── eval_questions.json ← 10 test Q&A pairs
│   │   └── results.csv         ← generated after eval run
│   └── faiss_index.bin         ← saved FAISS index (auto-generated)
└── tests/
    ├── test_retriever.py
    ├── test_agent.py
    └── test_ocr.py
```

---

## Key Classes and Functions

### `CurriculumRetriever` — `src/agent/retriever.py`
- Loads `data/curriculum/notes.json`
- Embeds with `all-MiniLM-L6-v2`
- Stores/loads FAISS index at `data/faiss_index.bin`
- `retrieve(query, top_k=3)` → returns top 3 relevant notes as plain text

### `TutorAgent` — `src/agent/orchestrator.py`
- Loads API key from `.env`
- Uses `gemini-1.5-flash`
- Accepts a `CurriculumRetriever` instance
- `answer_question(question, language='english')` → retrieves context + calls Gemini
- `summarize_lesson(text, language='english')` → summarizes lesson text
- `generate_quiz(topic, num_questions=3)` → returns MCQ quiz with answers and explanations

### `extract_text_from_image` — `src/ocr/ocr_pipeline.py`
- Opens image with Pillow
- Preprocesses with OpenCV (grayscale + threshold)
- Runs pytesseract
- Returns cleaned text string

### Gradio App — `app/main.py`
- Four tabs: Ask a Question, Summarize Lesson, Generate Quiz, Worksheet OCR
- Language dropdown (English / Urdu) on the Ask tab
- Runs on `localhost:7860`

---

## Guardrails — Always Follow These

### Safety
- Every Gemini prompt must begin with this system instruction:
  ```
  You are TutorBuddy, a friendly AI tutor for Grade 4–8 students.
  Always use simple, clear language appropriate for children aged 9–14.
  Never produce adult content, harmful content, or anything unrelated to education.
  Stay focused on the student's question only.
  ```
- If `language='urdu'`, append `Respond in Urdu.` to the system instruction.
- Do not remove or weaken this instruction under any circumstance.

### Code Style
- All Python files use type hints.
- Every class and function has a short docstring.
- No hardcoded API keys anywhere — always load from `.env` using `python-dotenv`.
- All file paths use `pathlib.Path`, not raw strings.
- Keep functions small and single-purpose.

### Error Handling
- Wrap all Gemini API calls in try/except. On failure, return a user-friendly error string, never crash.
- Wrap all OCR operations in try/except. If OCR fails, return `"Could not read image. Please try a clearer photo."`.
- Wrap FAISS index loading in try/except. If index file missing, rebuild it automatically.

### Kaggle Compatibility
- No relative imports that break inside a notebook. Use absolute imports or inline all code.
- API key in Kaggle must use `kaggle_secrets`:
  ```python
  from kaggle_secrets import UserSecretsClient
  api_key = UserSecretsClient().get_secret("GEMINI_API_KEY")
  ```
- All pip installs go in the first notebook cell with `!pip install -q ...`.
- Avoid writing to paths outside the notebook working directory on Kaggle.

---

## What Is Already Done

Check this list before starting any task. Do not re-create files that exist.

- [ ] Phase 0 — Project folder + venv + packages installed
- [ ] Phase 1 — `data/curriculum/notes.json` + `data/eval/eval_questions.json`
- [ ] Phase 2 — `src/agent/retriever.py` + FAISS index working
- [ ] Phase 3 — `src/agent/orchestrator.py` + agent tests passing
- [ ] Phase 4 — `src/ocr/ocr_pipeline.py` + OCR test passing
- [ ] Phase 5 — `app/main.py` Gradio UI running
- [ ] Phase 6 — `src/evaluation/metrics.py` + results.csv generated
- [ ] Phase 7 — `Capstone_TutorBuddy.ipynb` complete and tested on Kaggle

Update this list manually as you finish each phase.

---

## How to Run Things

```bash
# Activate venv (always do this first)
source venv/bin/activate

# Test retriever
python tests/test_retriever.py

# Test agent
python tests/test_agent.py

# Test OCR
python tests/test_ocr.py

# Run all tests
pytest tests/

# Launch Gradio UI
python app/main.py

# Run evaluation
python src/evaluation/metrics.py
```

---

## Current Phase

> **Update this line manually when you move to a new phase.**

Currently on: **Phase 0 — Setup**

---

## Notes and Decisions

- OCR is optional for the Kaggle demo but included in the full local build.
- Urdu support is prompt-level only — no separate translation model needed.
- FAISS is used over Chroma for simplicity and Kaggle compatibility.
- Gradio is preferred over FastAPI for the demo because it works inline in notebooks.
- `gemini-1.5-flash` is chosen for speed and free-tier availability.
- Evaluation is keyword-overlap based (not semantic) to keep it simple and reproducible.
