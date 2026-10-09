"""
Legal Metadata Enricher - Extract granular legal document metadata
Specialized for Constitution articles, Bills, and Acts
"""

from typing import Dict, List, Optional
from loguru import logger
import re


class LegalMetadataEnricher:
    """Extract structured metadata from legal documents"""

    def __init__(self):
        # Patterns for Constitution
        self.article_pattern = re.compile(r"Article\s+(\d+[A-Z]?)", re.IGNORECASE)
        self.clause_pattern = re.compile(r"\((\d+[a-z]?)\)")
        self.section_pattern = re.compile(r"Section\s+(\d+)", re.IGNORECASE)

        # Patterns for Bills/Acts
        self.bill_clause_pattern = re.compile(r"Clause\s+(\d+)", re.IGNORECASE)
        self.part_pattern = re.compile(r"PART\s+([IVX]+|[0-9]+)", re.IGNORECASE)

    def enrich_constitution(self, chunk: Dict) -> Dict:
        """
        Enrich Constitution chunks with granular metadata

        Extracts:
        - article_number (e.g., "43", "43A")
        - article_title (e.g., "Economic and social rights")
        - clause (e.g., "1(b)")
        - section (if applicable)
        """
        text = chunk.get("text", "")
        title = chunk.get("title", "")

        # Extract article number
        article_match = self.article_pattern.search(text)
        if article_match:
            chunk["article_number"] = article_match.group(1)
        else:
            # Try from title or metadata
            title_match = self.article_pattern.search(title)
            if title_match:
                chunk["article_number"] = title_match.group(1)

        # Extract article title (usually after "Article X - ")
        article_title_pattern = re.compile(
            r"Article\s+\d+[A-Z]?\s*[-–—]\s*([^\n\r]+)", re.IGNORECASE
        )
        title_match = article_title_pattern.search(text)
        if title_match:
            chunk["article_title"] = title_match.group(1).strip()
        elif "article_title" not in chunk:
            # Try to infer from context
            chunk["article_title"] = self._extract_section_heading(text)

        # Extract clause numbers
        clause_matches = self.clause_pattern.findall(text)
        if clause_matches:
            # Store the first/main clause reference
            chunk["clause"] = clause_matches[0]
            # Store all clause references for searching
            chunk["all_clauses"] = clause_matches

        # Extract section if present
        section_match = self.section_pattern.search(text)
        if section_match:
            chunk["section"] = section_match.group(1)

        # Categorize by subject matter (for better retrieval)
        chunk["legal_subjects"] = self._identify_constitutional_subjects(text, title)

        # Mark as Constitution
        chunk["category"] = "Constitution"
        chunk["document_type"] = "Constitution"

        logger.debug(
            f"Enriched Constitution chunk: Article {chunk.get('article_number', 'N/A')}"
        )
        return chunk

    def enrich_bill(self, chunk: Dict) -> Dict:
        """
        Enrich Bill/Act chunks with granular metadata

        Extracts:
        - clause_number (e.g., "16")
        - subject (e.g., "Housing Levy")
        - part (e.g., "IV")
        - bill_title
        """
        text = chunk.get("text", "")
        title = chunk.get("title", "")

        # Extract clause number
        clause_match = self.bill_clause_pattern.search(text)
        if clause_match:
            chunk["clause_number"] = clause_match.group(1)

        # Extract Part number
        part_match = self.part_pattern.search(text)
        if part_match:
            chunk["part"] = part_match.group(1)

        # Extract subject/topic from heading
        subject = self._extract_section_heading(text)
        if subject:
            chunk["subject"] = subject

        # Store bill title
        if "bill" in title.lower() or "act" in title.lower():
            chunk["bill_title"] = title

        # Identify if it's a Bill or Act
        if "bill" in title.lower():
            chunk["category"] = "Bill"
            chunk["document_type"] = "Bill"
        elif "act" in title.lower():
            chunk["category"] = "Act"
            chunk["document_type"] = "Act"
        else:
            chunk["category"] = "Legislation"
            chunk["document_type"] = "Legislation"

        # Extract year if present
        year_match = re.search(r"(20\d{2})", title)
        if year_match:
            chunk["year"] = year_match.group(1)

        # Identify legal subjects
        chunk["legal_subjects"] = self._identify_bill_subjects(text, title)

        logger.debug(f"Enriched Bill chunk: Clause {chunk.get('clause_number', 'N/A')}")
        return chunk

    def enrich_parliament(self, chunk: Dict) -> Dict:
        """Enrich Parliamentary proceedings"""
        text = chunk.get("text", "")

        # Extract speaker information
        speaker_pattern = re.compile(
            r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*:", re.MULTILINE
        )
        speakers = speaker_pattern.findall(text)
        if speakers:
            chunk["speakers"] = list(set(speakers))
            chunk["primary_speaker"] = speakers[0] if speakers else None

        # Extract debate topics
        chunk["subjects"] = self._extract_debate_topics(text)

        chunk["category"] = "Parliament"
        chunk["document_type"] = "Parliamentary Proceeding"

        return chunk

    def auto_enrich(self, chunk: Dict) -> Dict:
        """
        Automatically detect document type and apply appropriate enrichment
        """
        title = chunk.get("title", "").lower()
        text = chunk.get("text", "")
        source_url = chunk.get("source_url", "").lower()

        # Detect Constitution
        if "constitution" in title or "constitution" in source_url:
            return self.enrich_constitution(chunk)

        # Detect Bills/Acts
        elif "bill" in title or "act" in title:
            return self.enrich_bill(chunk)

        # Detect Parliament
        elif "parliament" in source_url or "hansard" in title.lower():
            return self.enrich_parliament(chunk)

        # Default: basic enrichment
        else:
            chunk.setdefault("category", "General")
            return chunk

    def _extract_section_heading(self, text: str) -> Optional[str]:
        """
        Extract section/clause heading (usually bold or uppercase)
        """
        lines = text.split("\n")
        for line in lines[:5]:  # Check first few lines
            line = line.strip()
            # Look for short, capitalized lines (likely headings)
            if line.isupper() and 5 < len(line) < 100:
                return line.title()
            # Look for lines ending with specific patterns
            if re.match(r"^[A-Z][^.!?]*$", line) and 5 < len(line) < 100:
                return line
        return None

    def _identify_constitutional_subjects(self, text: str, title: str) -> List[str]:
        """
        Identify constitutional subject areas for better retrieval
        """
        subjects = []
        text_lower = (text + " " + title).lower()

        # Define constitutional subject areas
        subject_keywords = {
            "rights": ["right", "rights", "freedom", "liberty"],
            "economic_rights": ["property", "housing", "economic", "social"],
            "governance": ["government", "parliament", "executive", "judiciary"],
            "taxation": ["tax", "levy", "revenue", "finance"],
            "land": ["land", "property", "ownership"],
            "citizenship": ["citizen", "citizenship", "nationality"],
            "devolution": ["county", "devolution", "local government"],
            "bill_of_rights": ["bill of rights", "fundamental rights"],
            "elections": ["election", "electoral", "vote", "voting"],
            "public_finance": ["public finance", "budget", "appropriation"],
        }

        for subject, keywords in subject_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                subjects.append(subject)

        return subjects

    def _identify_bill_subjects(self, text: str, title: str) -> List[str]:
        """
        Identify bill subject areas
        """
        subjects = []
        text_lower = (text + " " + title).lower()

        # Common bill subjects
        subject_keywords = {
            "taxation": ["tax", "levy", "duty", "revenue"],
            "finance": ["finance", "budget", "appropriation", "expenditure"],
            "housing": ["housing", "shelter", "residential"],
            "health": ["health", "medical", "healthcare"],
            "education": ["education", "school", "university"],
            "agriculture": ["agriculture", "farming", "crop"],
            "trade": ["trade", "commerce", "business"],
            "security": ["security", "police", "defense"],
            "energy": ["energy", "power", "electricity"],
            "environment": ["environment", "conservation", "climate"],
        }

        for subject, keywords in subject_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                subjects.append(subject)

        return subjects

    def _extract_debate_topics(self, text: str) -> List[str]:
        """Extract topics from parliamentary debate"""
        # Simple keyword extraction
        topics = []
        text_lower = text.lower()

        common_topics = [
            "budget",
            "finance",
            "health",
            "education",
            "security",
            "agriculture",
            "infrastructure",
            "corruption",
            "revenue",
            "county",
            "devolution",
            "parliament",
            "bill",
            "motion",
        ]

        for topic in common_topics:
            if topic in text_lower:
                topics.append(topic)

        return topics

    def validate_legal_metadata(self, chunk: Dict) -> bool:
        """
        Validate that legal document has required metadata
        """
        doc_type = chunk.get("document_type", "")

        if doc_type == "Constitution":
            # Constitution should have article_number
            return "article_number" in chunk or "article_title" in chunk

        elif doc_type in ["Bill", "Act"]:
            # Bills should have clause_number or subject
            return "clause_number" in chunk or "subject" in chunk

        return True  # Other types always valid
