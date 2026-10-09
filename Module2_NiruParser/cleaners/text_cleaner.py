"""
Text Cleaner - Clean and normalize extracted text
"""

import re
import unicodedata
from loguru import logger

try:
    import ftfy

    FTFY_AVAILABLE = True
except ImportError:
    FTFY_AVAILABLE = False


class TextCleaner:
    """Clean and normalize text"""

    def __init__(self):
        # Patterns to clean
        self.patterns = {
            # Multiple whitespace
            "whitespace": re.compile(r"\s+"),
            # Multiple newlines
            "newlines": re.compile(r"\n{3,}"),
            # HTML entities
            "html_entities": re.compile(r"&[a-zA-Z]+;"),
        }

    def clean(self, text: str, aggressive: bool = False) -> str:
        """
        Clean and normalize text

        Args:
            text: Raw text
            aggressive: If True, apply more aggressive cleaning

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        try:
            # Normalize Unicode
            text = unicodedata.normalize("NFKC", text)

            # Remove HTML entities
            text = self.patterns["html_entities"].sub(" ", text)

            # Remove excessive newlines (keep paragraph breaks)
            text = self.patterns["newlines"].sub("\n\n", text)

            # Normalize whitespace
            text = self.patterns["whitespace"].sub(" ", text)

            # Remove leading/trailing whitespace from each line
            lines = [line.strip() for line in text.split("\n")]
            text = "\n".join(line for line in lines if line)

            if aggressive:
                # Remove very short lines (likely artifacts)
                lines = [line for line in text.split("\n") if len(line) > 20]
                text = "\n".join(lines)

                # Remove lines that are mostly numbers/special chars
                lines = []
                for line in text.split("\n"):
                    alpha_ratio = sum(c.isalpha() for c in line) / max(len(line), 1)
                    if alpha_ratio > 0.5:  # At least 50% alphabetic
                        lines.append(line)
                text = "\n".join(lines)

            # Final cleanup
            text = text.strip()

            return text

        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text

    def remove_boilerplate(self, text: str) -> str:
        """Remove common boilerplate text"""
        # Common phrases to remove
        boilerplate_phrases = [
            r"Cookie Policy",
            r"Privacy Policy",
            r"Terms of Service",
            r"Subscribe to our newsletter",
            r"Share this article",
            r"Follow us on",
            r"Advertisement",
            r"All rights reserved",
            r"Copyright \d{4}",
            r"Click here",
            r"Read more",
            r"Related Articles?",
            r"You might also like",
            r"This article was originally published",
            r"Send us your stories",
            r"For more information",
            r"Contact us",
        ]

        for phrase in boilerplate_phrases:
            text = re.sub(phrase, "", text, flags=re.IGNORECASE)

        return text

    def fix_encoding(self, text: str) -> str:
        """Fix common encoding issues using ftfy (fallback to manual map)"""
        if FTFY_AVAILABLE:
            try:
                return ftfy.fix_text(text)
            except Exception as e:
                logger.warning(f"ftfy failed, falling back to manual fix: {e}")

        # Manual mojibake replacements (fallback when ftfy unavailable)
        replacements = {
            "â€™": "'",
            "â€œ": '"',
            "â€": '"',
            'â€"': "—",
            'â€"': "–",
            "Ã©": "é",
            "Ã¨": "è",
            "Ã¡": "á",
            "Ã¢": "â",
            "Ã¼": "ü",
            "Ã¶": "ö",
            "Ã¤": "ä",
            "Ã ": "à",
        }

        for wrong, right in replacements.items():
            text = text.replace(wrong, right)

        return text
