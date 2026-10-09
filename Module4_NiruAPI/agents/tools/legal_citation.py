"""Legal Citation Tool - Format legal references in proper Kenya/international format"""

import re
from typing import Dict, Any, Optional, List
from loguru import logger


class LegalCitationTool:
    """Format legal citations in proper Kenyan and international legal formats"""

    name = "legal_citation"
    description = "Format case citations, statute references, constitutional articles, and bibliography entries in proper legal format"

    def execute(self, query: str) -> Dict[str, Any]:
        try:
            query = query.strip()
            result = self._format_citation(query)
            return result
        except Exception as e:
            logger.error(f"Citation formatting failed: {e}")
            return {"success": False, "error": str(e), "formatted": query}

    def _format_citation(self, text: str) -> Dict[str, Any]:
        # Kenyan case citation: e.g., "Njoya v AG 2004 eKLR" → "[2004] eKLR (Njoya v Attorney General)"
        case_match = re.match(
            r"^([A-Za-z\s]+)\s+v(?:s|\.)?\s+([A-Za-z\s]+)\s+(\d{4})\s*(?:eKLR|KLR|EA)?\s*$",
            text.strip(),
        )
        if case_match:
            p1 = case_match.group(1).strip()
            p2 = case_match.group(2).strip()
            year = case_match.group(3)
            # Expand common abbreviations
            p1_full = {
                "AG": "Attorney General",
                "AG": "Attorney General",
                "IEBC": "Independent Electoral and Boundaries Commission",
                "KEBS": "Kenya Bureau of Standards",
                "KRA": "Kenya Revenue Authority",
                "NLC": "National Land Commission",
            }.get(p1.upper(), p1)
            p2_full = {
                "AG": "Attorney General",
                "AG": "Attorney General",
                "IEBC": "Independent Electoral and Boundaries Commission",
                "KEBS": "Kenya Bureau of Standards",
                "KRA": "Kenya Revenue Authority",
            }.get(p2.upper(), p2)
            formatted = f"{p1_full} v {p2_full} [{year}] eKLR"
            return {
                "success": True,
                "formatted": formatted,
                "format": "Kenyan Case Citation",
                "components": {
                    "plaintiff": p1_full,
                    "defendant": p2_full,
                    "year": year,
                    "reporter": "eKLR",
                },
            }

        # Statute: "Finance Act 2024" → "Finance Act, No. X of 2024"
        statute_match = re.match(
            r"^([A-Za-z\s]+)\s+(?:Act|Bill)\s+(\d{4})\s*$", text.strip()
        )
        if statute_match:
            name = statute_match.group(1).strip()
            year = statute_match.group(2)
            formatted = f"{name} Act, No. X of {year}"
            return {
                "success": True,
                "formatted": formatted,
                "format": "Kenyan Statute Reference",
                "components": {"name": name, "type": "Act", "year": year},
            }

        # Article: e.g., "Article 27 COK" or "Article 27(4) COK"
        article_match = re.match(
            r"^Article\s+(\d+(?:\([a-z0-9]+\))?)\s*(?:of\s+)?(?:the\s+)?(?:Constitution|COK)?\s*$",
            text.strip(),
            re.IGNORECASE,
        )
        if article_match:
            article_no = article_match.group(1)
            formatted = f"Article {article_no} of the Constitution of Kenya, 2010"
            return {
                "success": True,
                "formatted": formatted,
                "format": "Constitutional Article Reference",
                "components": {
                    "article": article_no,
                    "document": "Constitution of Kenya, 2010",
                },
            }

        # Section: e.g., "Section 23(1) Finance Act"
        section_match = re.match(
            r"^Section\s+(\d+(?:\([a-z0-9]+\))?)\s*(?:of\s+)?(.+?)(?:\s+Act)?\s*$",
            text.strip(),
            re.IGNORECASE,
        )
        if section_match:
            section_no = section_match.group(1)
            act_name = section_match.group(2).strip()
            formatted = f"Section {section_no} of the {act_name} Act"
            return {
                "success": True,
                "formatted": formatted,
                "format": "Statutory Section Reference",
                "components": {"section": section_no, "act": act_name},
            }

        # Default: return as plain text with formatting suggestion
        return {
            "success": True,
            "formatted": text,
            "format": "Plain text",
            "note": "Could not detect citation format. Try formats like: 'Njoya v AG 2004 eKLR', 'Article 27 COK', 'Finance Act 2024'",
        }
