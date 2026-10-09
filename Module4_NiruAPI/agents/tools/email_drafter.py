"""
Email Drafter Tool - Professional Email Drafting with Templates

Features:
- Multiple email templates (formal, informal, legal, business)
- Email validation
- Markdown to HTML conversion
- Kenyan legal communication format
- LLM-ready tool schema
"""

import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class EmailFormat(Enum):
    """Supported email formats."""

    PLAIN = "plain"
    HTML = "html"
    MARKDOWN = "markdown"


class EmailTemplate(Enum):
    """Pre-defined email templates."""

    FORMAL = "formal"
    INFORMAL = "informal"
    LEGAL = "legal"
    BUSINESS = "business"
    FOLLOW_UP = "follow_up"
    REQUEST = "request"
    ACKNOWLEDGEMENT = "acknowledgement"


@dataclass
class EmailRecipient:
    """Validated email recipient."""

    email: str
    name: Optional[str] = None

    @property
    def formatted(self) -> str:
        if self.name:
            return f"{self.name} <{self.email}>"
        return self.email


class EmailDrafterTool:
    """
    Professional email drafter with templates and validation.

    Features:
    - Multiple email templates
    - Email validation
    - Kenyan legal formatting
    - CC/BCC support
    - Markdown to HTML conversion
    """

    name = "email_draft"
    description = (
        "Draft professional emails with templates. "
        "Supports: formal, informal, legal, business formats. "
        "Best for: generating well-formatted email content."
    )

    # Email regex pattern
    EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

    # Templates
    TEMPLATES = {
        EmailTemplate.FORMAL: {
            "greeting": "Dear {recipient_name},",
            "closing": "Yours sincerely,",
            "style": "formal",
        },
        EmailTemplate.INFORMAL: {
            "greeting": "Hi {recipient_name},",
            "closing": "Best regards,",
            "style": "casual",
        },
        EmailTemplate.LEGAL: {
            "greeting": "Dear {recipient_name},",
            "closing": "Yours faithfully,",
            "style": "legal",
            "disclaimer": (
                "\n\n---\nDISCLAIMER: This email and any attachments are confidential and "
                "intended solely for the addressee. If you have received this email in error, "
                "please notify the sender immediately and delete it. Any unauthorized use, "
                "disclosure, or distribution is prohibited."
            ),
        },
        EmailTemplate.BUSINESS: {
            "greeting": "Dear {recipient_name},",
            "closing": "Kind regards,",
            "style": "business",
        },
        EmailTemplate.FOLLOW_UP: {
            "greeting": "Hi {recipient_name},",
            "opening": "I hope this email finds you well. I'm following up on our previous conversation.",
            "closing": "Looking forward to hearing from you.",
            "style": "follow_up",
        },
        EmailTemplate.REQUEST: {
            "greeting": "Dear {recipient_name},",
            "opening": "I am writing to request the following:",
            "closing": "Thank you for your consideration.",
            "style": "request",
        },
        EmailTemplate.ACKNOWLEDGEMENT: {
            "greeting": "Dear {recipient_name},",
            "opening": "Thank you for your communication. This is to acknowledge receipt of:",
            "closing": "We will respond in due course.",
            "style": "acknowledgement",
        },
    }

    def __init__(
        self,
        sender_name: Optional[str] = None,
        sender_email: Optional[str] = None,
        default_signature: Optional[str] = None,
    ):
        """
        Initialize email drafter.

        Args:
            sender_name: Default sender name
            sender_email: Default sender email
            default_signature: Default email signature
        """
        self.sender_name = sender_name
        self.sender_email = sender_email
        self.default_signature = default_signature or self._create_default_signature()

        logger.info("EmailDrafterTool initialized")

    def _create_default_signature(self) -> str:
        """Create default email signature."""
        if self.sender_name:
            return f"\n\n{self.sender_name}"
        return ""

    def execute(
        self,
        to: str,
        subject: str,
        body: str,
        cc: Optional[str] = None,
        bcc: Optional[str] = None,
        format: str = "plain",
        template: Optional[str] = None,
        recipient_name: Optional[str] = None,
        sender_name: Optional[str] = None,
        include_signature: bool = True,
        include_timestamp: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Draft a professional email.

        Args:
            to: Recipient email address(es), comma-separated
            subject: Email subject
            body: Email body content
            cc: CC recipients (comma-separated)
            bcc: BCC recipients (comma-separated)
            format: Email format ('plain', 'html', 'markdown')
            template: Template to use ('formal', 'legal', 'business', etc.)
            recipient_name: Name of recipient for greeting
            sender_name: Sender name for signature
            include_signature: Whether to include signature
            include_timestamp: Whether to include timestamp

        Returns:
            Drafted email with formatted content
        """
        try:
            # Validate recipients
            to_list = self._parse_recipients(to)
            if not to_list:
                return self._error_response("Invalid recipient email address")

            cc_list = self._parse_recipients(cc) if cc else []
            bcc_list = self._parse_recipients(bcc) if bcc else []

            # Determine template
            email_template = None
            if template:
                try:
                    email_template = EmailTemplate(template.lower())
                except ValueError:
                    logger.warning(f"Unknown template: {template}")

            # Build email content
            formatted_body = self._format_body(
                body=body,
                template=email_template,
                recipient_name=recipient_name or to_list[0].email.split("@")[0],
                sender_name=sender_name or self.sender_name,
                include_signature=include_signature,
            )

            # Add timestamp if requested
            if include_timestamp:
                formatted_body += f"\n\n---\nDrafted: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"

            # Convert format if needed
            if format.lower() == "html":
                formatted_body = self._to_html(formatted_body)

            # Build email object
            email_content = {
                "to": [r.formatted for r in to_list],
                "subject": subject,
                "body": formatted_body,
                "format": format,
                "template": template,
                "drafted_at": datetime.utcnow().isoformat(),
            }

            if cc_list:
                email_content["cc"] = [r.formatted for r in cc_list]

            if bcc_list:
                email_content["bcc"] = [r.formatted for r in bcc_list]

            # Create plain text version for display
            email_text = self._format_email_text(email_content)
            email_content["formatted"] = email_text

            # Create mailto link
            email_content["mailto_link"] = self._create_mailto_link(
                to_list, subject, formatted_body, cc_list
            )

            return {
                "email": email_content,
                "success": True,
                "word_count": len(formatted_body.split()),
                "character_count": len(formatted_body),
            }

        except Exception as e:
            logger.error(f"Error drafting email: {e}")
            return self._error_response(str(e))

    def _parse_recipients(self, recipients: str) -> List[EmailRecipient]:
        """Parse and validate recipient string."""
        if not recipients:
            return []

        result = []
        for recipient in recipients.split(","):
            recipient = recipient.strip()

            # Check for "Name <email>" format
            match = re.match(r"^(.+?)\s*<(.+?)>$", recipient)
            if match:
                name, email = match.groups()
                if self._is_valid_email(email):
                    result.append(EmailRecipient(email=email, name=name.strip()))
            elif self._is_valid_email(recipient):
                result.append(EmailRecipient(email=recipient))

        return result

    def _is_valid_email(self, email: str) -> bool:
        """Validate email address."""
        return bool(self.EMAIL_PATTERN.match(email))

    def _format_body(
        self,
        body: str,
        template: Optional[EmailTemplate],
        recipient_name: str,
        sender_name: Optional[str],
        include_signature: bool,
    ) -> str:
        """Format email body with template."""
        parts = []

        if template and template in self.TEMPLATES:
            tmpl = self.TEMPLATES[template]

            # Greeting
            greeting = tmpl.get("greeting", "").format(recipient_name=recipient_name)
            if greeting:
                parts.append(greeting)

            # Opening (if template has one)
            opening = tmpl.get("opening", "")
            if opening:
                parts.append(f"\n{opening}")

        # Main body
        parts.append(f"\n{body}")

        if template and template in self.TEMPLATES:
            tmpl = self.TEMPLATES[template]

            # Closing
            closing = tmpl.get("closing", "")
            if closing:
                parts.append(f"\n\n{closing}")

            # Disclaimer (for legal emails)
            disclaimer = tmpl.get("disclaimer", "")
            if disclaimer:
                parts.append(disclaimer)

        # Signature
        if include_signature:
            sig = sender_name or self.sender_name
            if sig:
                parts.append(f"\n{sig}")
            elif self.default_signature:
                parts.append(self.default_signature)

        return "\n".join(parts)

    def _to_html(self, text: str) -> str:
        """Convert plain text to basic HTML."""
        # Escape HTML
        html = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # Convert line breaks
        html = html.replace("\n\n", "</p><p>").replace("\n", "<br>")

        # Wrap in paragraph
        return f"<html><body><p>{html}</p></body></html>"

    def _format_email_text(self, email: Dict[str, Any]) -> str:
        """Format email for display."""
        lines = [
            f"To: {', '.join(email['to'])}",
        ]

        if email.get("cc"):
            lines.append(f"CC: {', '.join(email['cc'])}")

        if email.get("bcc"):
            lines.append(f"BCC: {', '.join(email['bcc'])}")

        lines.extend(
            [
                f"Subject: {email['subject']}",
                "",
                email["body"],
            ]
        )

        return "\n".join(lines)

    def _create_mailto_link(
        self,
        to_list: List[EmailRecipient],
        subject: str,
        body: str,
        cc_list: List[EmailRecipient],
    ) -> str:
        """Create mailto link for opening in email client."""
        import urllib.parse

        to_emails = ",".join([r.email for r in to_list])
        params = {"subject": subject, "body": body}

        if cc_list:
            params["cc"] = ",".join([r.email for r in cc_list])

        query = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
        return f"mailto:{to_emails}?{query}"

    def _error_response(self, error: str) -> Dict[str, Any]:
        """Create error response."""
        return {
            "email": None,
            "success": False,
            "error": error,
        }

    def get_tool_schema(self) -> Dict[str, Any]:
        """Get tool schema for LLM function calling."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email(s), comma-separated",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line",
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body content",
                    },
                    "template": {
                        "type": "string",
                        "enum": [
                            "formal",
                            "informal",
                            "legal",
                            "business",
                            "follow_up",
                            "request",
                        ],
                        "description": "Email template to use",
                    },
                    "cc": {
                        "type": "string",
                        "description": "CC recipients, comma-separated",
                    },
                    "recipient_name": {
                        "type": "string",
                        "description": "Recipient name for greeting",
                    },
                },
                "required": ["to", "subject", "body"],
            },
        }

    def list_templates(self) -> List[str]:
        """List available email templates."""
        return [t.value for t in EmailTemplate]
