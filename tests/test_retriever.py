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

def test_retriever():
    r = CurriculumRetriever()
    result = r.retrieve("what is photosynthesis", top_k=3)
    print("=== Retrieve: 'what is photosynthesis' ===\n")
    print(result)


if __name__ == "__main__":
    test_retriever()
