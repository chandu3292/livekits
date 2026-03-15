"""
Core OCR extraction using pytesseract.
Handles image-to-text conversion with preprocessing.
"""

import logging
from PIL import Image, ImageFilter, ImageEnhance
import pytesseract

logger = logging.getLogger("ocr")


class OCRExtractor:
    """Pytesseract-based OCR extractor with image preprocessing."""

    def __init__(self, lang: str = "eng", tesseract_cmd: str = None):
        """
        Args:
            lang: Tesseract language code (e.g., 'eng', 'tam', 'tel', 'eng+tam')
            tesseract_cmd: Path to tesseract executable (auto-detected if None)
        """
        self.lang = lang
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR accuracy."""
        # Convert to grayscale
        img = image.convert("L")
        # Increase contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        # Sharpen
        img = img.filter(ImageFilter.SHARPEN)
        # Binarize (threshold)
        img = img.point(lambda x: 0 if x < 140 else 255, "1")
        return img

    def extract_text(self, image: Image.Image, preprocess: bool = True) -> str:
        """
        Extract text from a PIL Image.

        Args:
            image: PIL Image object
            preprocess: Whether to preprocess the image first

        Returns:
            Extracted text string
        """
        if preprocess:
            image = self.preprocess_image(image)

        try:
            text = pytesseract.image_to_string(image, lang=self.lang)
            return text.strip()
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""

    def extract_with_details(self, image: Image.Image) -> dict:
        """
        Extract text with bounding box and confidence data.

        Returns:
            Dict with 'text', 'data' (TSV parsed), and 'hocr' keys
        """
        if image.mode != "L":
            image = self.preprocess_image(image)

        try:
            text = pytesseract.image_to_string(image, lang=self.lang)
            data = pytesseract.image_to_data(image, lang=self.lang, output_type=pytesseract.Output.DICT)
            return {
                "text": text.strip(),
                "data": data,
            }
        except Exception as e:
            logger.error(f"OCR detailed extraction failed: {e}")
            return {"text": "", "data": {}}
