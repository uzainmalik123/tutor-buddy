from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image


def extract_text_from_image(image_path: str | Path) -> str:
    """Extract and clean text from an image using OCR."""
    try:
        img = Image.open(image_path).convert("RGB")
        img = np.array(img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(thresh)
        cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        return cleaned
    except Exception as e:
        print(e)
        return "Could not read image. Please try a clearer photo."
