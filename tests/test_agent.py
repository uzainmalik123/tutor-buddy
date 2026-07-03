import sys
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


def test_agent():
    retriever = CurriculumRetriever()
    agent = TutorAgent(retriever)

    print("=== Answer: 'What is the water cycle?' ===\n")
    answer = agent.answer_question("What is the water cycle?")
    print(answer)
    print()

    print("=== Quiz: 'fractions' (2 questions) ===\n")
    quiz = agent.generate_quiz("fractions", num_questions=2)
    print(quiz)


if __name__ == "__main__":
    test_agent()
