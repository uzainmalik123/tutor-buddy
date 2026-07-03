import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

venv_site = root / "venv" / "lib"
if venv_site.exists():
    site_packages = sorted(venv_site.rglob("site-packages"))
    if site_packages:
        sys.path.insert(0, str(site_packages[0]))

from src.ocr.ocr_pipeline import extract_text_from_image


def test_ocr():
    image_path = root / "assets" / "test_worksheet.png"
    print(f"=== OCR test: {image_path} ===\n")
    result = extract_text_from_image(str(image_path))
    print(result)


if __name__ == "__main__":
    test_ocr()
