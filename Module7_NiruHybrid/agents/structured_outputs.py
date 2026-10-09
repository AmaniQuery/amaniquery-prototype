"""
Structured Outputs with Pydantic Validation
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from loguru import logger


class Dossier(BaseModel):
    """Structured dossier output"""

    title: str = Field(..., description="Title of the dossier")
    summary: str = Field(..., description="Executive summary")
    key_findings: List[str] = Field(..., description="Key findings")
    sources: List[Dict[str, str]] = Field(..., description="Source citations")
    recommendations: List[str] = Field(..., description="Recommendations")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class EmailDraft(BaseModel):
    """Structured email draft output"""

    to: str = Field(..., description="Recipient email")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body")
    cc: Optional[List[str]] = Field(None, description="CC recipients")
    bcc: Optional[List[str]] = Field(None, description="BCC recipients")


class ResearchReport(BaseModel):
    """Structured research report output"""

    query: str = Field(..., description="Research query")
    answer: str = Field(..., description="Research answer")
    sources: List[Dict[str, Any]] = Field(..., description="Sources used")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    methodology: str = Field(..., description="Research methodology")
    limitations: List[str] = Field(..., description="Limitations")


class StructuredOutputs:
    """
    Manages structured outputs with Pydantic validation

    Provides validation and creation methods for structured data types
    including dossiers, email drafts, and research reports.
    """

    def __init__(self):
        """Initialize structured outputs manager"""
        self.validation_count = 0
        self.error_count = 0

    def validate_dossier(self, data: Dict[str, Any]) -> Dossier:
        """Validate and create dossier"""
        self.validation_count += 1
        try:
            return Dossier(**data)
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error validating dossier: {e}")
            raise ValueError(f"Invalid dossier format: {e}")

    def validate_email(self, data: Dict[str, Any]) -> EmailDraft:
        """Validate and create email draft"""
        self.validation_count += 1
        try:
            return EmailDraft(**data)
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error validating email: {e}")
            raise ValueError(f"Invalid email format: {e}")

    def validate_research_report(self, data: Dict[str, Any]) -> ResearchReport:
        """Validate and create research report"""
        self.validation_count += 1
        try:
            return ResearchReport(**data)
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error validating research report: {e}")
            raise ValueError(f"Invalid research report format: {e}")

    def create_dossier(
        self,
        title: str,
        summary: str,
        key_findings: List[str],
        sources: List[Dict[str, str]],
        recommendations: List[str],
    ) -> Dossier:
        """Create a validated dossier"""
        data = {
            "title": title,
            "summary": summary,
            "key_findings": key_findings or [],
            "sources": sources or [],
            "recommendations": recommendations or [],
        }
        return self.validate_dossier(data)

    def create_email(
        self,
        to: str,
        subject: str,
        body: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
    ) -> EmailDraft:
        """Create a validated email draft"""
        data = {"to": to, "subject": subject, "body": body, "cc": cc, "bcc": bcc}
        return self.validate_email(data)

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return {
            "validation_count": self.validation_count,
            "error_count": self.error_count,
            "success_rate": (
                (self.validation_count - self.error_count) / self.validation_count
                if self.validation_count > 0
                else 0.0
            ),
        }
