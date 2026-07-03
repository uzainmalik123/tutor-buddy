# TutorBuddy

An AI education agent for Grade 4-8 students built with Gemini, RAG, and Gradio. Answers homework questions, summarizes lessons, generates quizzes, and extracts text from worksheets.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Gemini API](https://img.shields.io/badge/Gemini-2.5_Flash-8E75B2)](https://ai.google.dev/)
[![FAISS](https://img.shields.io/badge/FAISS-cpu-4B8BBE)](https://github.com/facebookresearch/faiss)
[![Gradio](https://img.shields.io/badge/Gradio-6.x-F97316)](https://gradio.app/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## What it does

- Answers curriculum-grounded questions by retrieving relevant notes via FAISS and generating responses with Gemini 2.5 Flash.
- Summarizes lesson text into clear, age-appropriate language for students aged 9-14.
- Generates multiple-choice quizzes with answer explanations on any topic, configurable from 1 to 5 questions.
- Extracts text from printed worksheet images using Tesseract OCR with OpenCV preprocessing.

## Architecture

```
                          +------------------+
                          |   Student Input   |
                          +--------+---------+
                                   |
                    +--------------+--------------+
                    |              |              |
            +-------v------+  +---v----+   +-----v-----+
            |  Text Input  |  |  Quiz  |   |  Upload   |
            |  (Question)  |  | Topic  |   |  Image    |
            +------+-------+  +---+----+   +-----+-----+
                   |              |               |
                   |       +------v-------+       |
                   |       |  TutorAgent  |       |
                   |       |  (Gemini)    |       |
                   |       +------+-------+       |
                   |              |               |
            +------v-------+      |        +------v-------+
            | Curriculum-  |      |        |  OCR Pipe-   |
            | Retriever    |<-----+        |  line        |
            | (FAISS)      |               |  (Tesseract) |
            +------+-------+               +------+-------+
                   |                              |
            +------v-------+               +------v-------+
            | all-MiniLM-  |               |  OpenCV +    |
            | L6-v2        |               |  Pillow      |
            +--------------+               +--------------+
                   |
            +------v-------+
            |  notes.json  |
            |  (30 notes)  |
            +--------------+
```

## Tech Stack

| Component | Tool |
|-----------|------|
| LLM | Gemini 2.5 Flash (`google-genai`) |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector store | FAISS (cpu) |
| OCR | Tesseract + OpenCV + Pillow |
| UI | Gradio |
| Evaluation | Jaccard keyword overlap |
| Environment | Python 3.11+, dotenv |

## Getting Started

```bash
git clone https://github.com/your-username/TutorBuddy.git
cd TutorBuddy
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

Get a free API key from [aistudio.google.com](https://aistudio.google.com/).

Run the app:

```bash
python app/main.py
```

Open `http://localhost:7860` in your browser.

## Project Structure

```
TutorBuddy/
├── Capstone_TutorBuddy.ipynb   -- Kaggle submission notebook
├── AGENT.md                    -- Project context for AI coding sessions
├── README.md
├── requirements.txt
├── Dockerfile
├── .env                        -- API key (never committed)
├── .gitignore
├── assets/
│   ├── cover.png
│   └── demo.mp4
├── app/
│   └── main.py                 -- Gradio UI
├── src/
│   ├── agent/
│   │   ├── orchestrator.py     -- TutorAgent (Gemini calls)
│   │   └── retriever.py        -- CurriculumRetriever (FAISS search)
│   ├── ocr/
│   │   └── ocr_pipeline.py     -- extract_text_from_image()
│   └── evaluation/
│       └── metrics.py          -- Keyword overlap evaluation
├── data/
│   ├── curriculum/
│   │   └── notes.json          -- 30 curriculum notes
│   ├── eval/
│   │   ├── eval_questions.json -- 10 QA pairs
│   │   └── results.csv         -- Evaluation results
│   └── faiss_index.bin         -- Built FAISS index
└── tests/
    ├── test_retriever.py
    ├── test_agent.py
    └── test_ocr.py
```

## Kaggle Submission

TutorBuddy was built for the [Kaggle 5-Day Generative AI Intensive](https://www.kaggle.com/learn/generative-ai-intensive) capstone. The notebook inlines all code for Kaggle compatibility and includes a live Gradio demo with `share=True`.

Notebook: [KAGGLE_NOTEBOOK_URL]

## Demo

*Screenshot or GIF of the Gradio interface goes here.*

## Evaluation

We evaluate on 10 held-out questions across math, science, and English. The metric is **Jaccard keyword overlap**: both the expected answer and model response are tokenized, stopwords are removed, and the intersection-over-union of their keyword sets is computed. A score of 1.0 means every keyword in the expected answer appeared in the response; 0.0 means no overlap. The pass rate is the fraction of questions scoring >= 0.3.

This metric is intentionally simple and deterministic -- it trades nuance for reproducibility. It measures keyword coverage, not semantic correctness, but it provides a consistent baseline for tracking improvement.

## Roadmap

- **Semantic evaluation**: Replace Jaccard overlap with sentence-BERT cosine similarity or LLM-as-judge scoring for more accurate quality measurement.
- **Scaled curriculum**: Expand from 30 notes to hundreds, covering more subjects and grade levels to improve answer breadth.
- **Multi-turn conversation**: Add chat memory so students can ask follow-up questions and the agent maintains context across turns.

## License

MIT. See [LICENSE](LICENSE).

---

Built by [your-name](https://github.com/your-username) for the Kaggle 5-Day Generative AI Intensive capstone.
