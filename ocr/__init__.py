"""
OCR Module - Pytesseract-based OCR with structured extraction
Supports: PDF, DOCX, MD, Images (PNG, JPG, JPEG, TIFF, BMP)
"""

from .extractor import OCRExtractor
from .table_extractor import TableExtractor
from .file_handlers import process_file

__all__ = [
    'OCRExtractor',
    'TableExtractor',
    'process_file',
]
