import csv
import re
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

venv_site = root / "venv" / "lib"
if venv_site.exists():
    site_packages = sorted(venv_site.rglob("site-packages"))
    if site_packages:
        sys.path.insert(0, str(site_packages[0]))

from src.agent.retriever import CurriculumRetriever
from src.agent.orchestrator import TutorAgent


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or",
    "nor", "not", "so", "yet", "both", "either", "neither", "each",
    "every", "all", "any", "few", "more", "most", "other", "some", "such",
    "no", "only", "own", "same", "than", "too", "very", "just", "because",
    "about", "if", "then", "else", "when", "where", "why", "how", "which",
    "who", "whom", "what", "this", "that", "these", "those", "it", "its",
    "you", "your", "they", "their", "we", "our", "i", "me", "my", "he",
    "she", "him", "her", "his",
}


def tokenize(text: str) -> set[str]:
    """Lowercase, split on non-alpha, remove stopwords and short tokens."""
    tokens = re.findall(r"[a-zA-Z\u0600-\u06FF]+", text.lower())
    return {t for t in tokens if t not in STOPWORDS and len(t) > 1}


def keyword_overlap(expected: str, actual: str) -> float:
    """Jaccard similarity between keyword sets of expected and actual."""
    exp_tokens = tokenize(expected)
    act_tokens = tokenize(actual)
    if not exp_tokens or not act_tokens:
        return 0.0
    intersection = exp_tokens & act_tokens
    union = exp_tokens | act_tokens
    return len(intersection) / len(union)


def run_evaluation(pass_threshold: float = 0.3) -> None:
    """Run all eval questions, compute scores, save results, print summary."""
    eval_path = Path("data/eval/eval_questions.json")
    results_path = Path("data/eval/results.csv")

    with open(eval_path) as f:
        import json
        questions = json.load(f)

    retriever = CurriculumRetriever()
    agent = TutorAgent(retriever)

    rows: list[dict] = []
    scores: list[float] = []
    lengths: list[int] = []

    for q in questions:
        question = q["question"]
        expected = q["expected_answer"]
        subject = q["subject"]

        print(f"[{subject}] {question}")
        response = agent.answer_question(question)
        score = keyword_overlap(expected, response)
        length = len(response.split())

        rows.append({
            "question": question,
            "expected": expected,
            "response": response,
            "subject": subject,
            "score": round(score, 3),
            "length": length,
        })
        scores.append(score)
        lengths.append(length)
        print(f"  score={score:.3f}  length={length}\n")
        time.sleep(1)

    avg_score = sum(scores) / len(scores)
    pass_count = sum(1 for s in scores if s >= pass_threshold)
    pass_rate = pass_count / len(scores)
    avg_length = sum(lengths) / len(lengths)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question", "expected", "response", "subject", "score", "length",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"  Questions:       {len(rows)}")
    print(f"  Avg keyword overlap: {avg_score:.3f}")
    print(f"  Pass rate (≥{pass_threshold}):  {pass_rate:.1%}")
    print(f"  Avg response length: {avg_length:.1f} words")
    print(f"  Results saved to: {results_path}")


if __name__ == "__main__":
    run_evaluation()
