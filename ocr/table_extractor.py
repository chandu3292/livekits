"""
Structured table and data extraction from OCR results.
Detects tables, key-value pairs, and structured data in images/documents.
"""

import re
import logging
from typing import List, Dict, Any
from PIL import Image
import pytesseract

logger = logging.getLogger("ocr.table")


class TableExtractor:
    """Extracts structured data (tables, key-value pairs) from images using pytesseract."""

    def __init__(self, lang: str = "eng"):
        self.lang = lang

    def extract_tables(self, image: Image.Image) -> List[List[str]]:
        """
        Extract tabular data from an image using pytesseract's TSV output.

        Args:
            image: PIL Image

        Returns:
            List of rows, where each row is a list of cell strings
        """
        try:
            data = pytesseract.image_to_data(
                image, lang=self.lang, output_type=pytesseract.Output.DICT
            )
        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
            return []

        # Group words by block and line
        blocks: Dict[int, Dict[int, List[dict]]] = {}
        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip()
            if not text:
                continue
            block_num = data["block_num"][i]
            line_num = data["line_num"][i]
            if block_num not in blocks:
                blocks[block_num] = {}
            if line_num not in blocks[block_num]:
                blocks[block_num][line_num] = []
            blocks[block_num][line_num].append({
                "text": text,
                "left": data["left"][i],
                "top": data["top"][i],
                "width": data["width"][i],
                "conf": int(data["conf"][i]) if data["conf"][i] != "-1" else 0,
            })

        # Find blocks that look like tables (multiple lines with aligned columns)
        tables = []
        for block_num, lines in sorted(blocks.items()):
            if len(lines) < 2:
                continue

            rows = []
            for line_num, words in sorted(lines.items()):
                # Sort words by horizontal position
                words.sort(key=lambda w: w["left"])
                row = self._cluster_words_into_columns(words)
                rows.append(row)

            # Only include if it looks tabular (consistent column count)
            if self._is_table_like(rows):
                tables.extend(rows)

        return tables

    def _cluster_words_into_columns(self, words: List[dict], gap_threshold: int = 30) -> List[str]:
        """Group words into columns based on horizontal gaps."""
        if not words:
            return []

        columns = []
        current_cell = [words[0]["text"]]
        prev_right = words[0]["left"] + words[0]["width"]

        for w in words[1:]:
            gap = w["left"] - prev_right
            if gap > gap_threshold:
                columns.append(" ".join(current_cell))
                current_cell = [w["text"]]
            else:
                current_cell.append(w["text"])
            prev_right = w["left"] + w["width"]

        columns.append(" ".join(current_cell))
        return columns

    def _is_table_like(self, rows: List[List[str]]) -> bool:
        """Check if rows have consistent column counts (table-like)."""
        if len(rows) < 2:
            return False
        col_counts = [len(r) for r in rows]
        most_common = max(set(col_counts), key=col_counts.count)
        matching = sum(1 for c in col_counts if c == most_common)
        return matching >= len(rows) * 0.6 and most_common >= 2

    def extract_key_value_pairs(self, text: str) -> Dict[str, str]:
        """
        Extract key-value pairs from OCR text.
        Handles patterns like "Key: Value", "Key = Value", "Key - Value"
        """
        pairs = {}
        patterns = [
            r'^(.+?)\s*:\s+(.+)$',
            r'^(.+?)\s*=\s+(.+)$',
            r'^(.+?)\s*-\s+(.+)$',
        ]
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    key = match.group(1).strip()
                    value = match.group(2).strip()
                    if len(key) < 50 and len(key) > 1:
                        pairs[key] = value
                    break
        return pairs

    def format_table_as_text(self, rows: List[List[str]]) -> str:
        """Format extracted table rows as a readable text table."""
        if not rows:
            return ""

        # Calculate column widths
        max_cols = max(len(r) for r in rows)
        col_widths = [0] * max_cols
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))

        # Format rows
        lines = []
        for row_idx, row in enumerate(rows):
            padded = []
            for i in range(max_cols):
                cell = row[i] if i < len(row) else ""
                padded.append(cell.ljust(col_widths[i]))
            lines.append(" | ".join(padded))
            # Add separator after header row
            if row_idx == 0:
                lines.append("-+-".join("-" * w for w in col_widths))

        return "\n".join(lines)
