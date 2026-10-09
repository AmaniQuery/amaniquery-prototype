"""Bill Status Tool - Look up Kenyan parliamentary bill status and votes"""

from typing import Dict, Any, Optional
from loguru import logger


class BillStatusTool:
    """Look up Kenyan parliamentary bill status and voting information"""

    name = "bill_status"
    description = "Get current status, voting results, and metadata for a Kenyan parliamentary bill"

    def __init__(self, metadata_manager=None):
        self._metadata_manager = metadata_manager

    def execute(self, bill_name: str) -> Dict[str, Any]:
        try:
            logger.info(f"[Tool] bill_status: '{bill_name}'")
            if not self._metadata_manager:
                return {
                    "success": False,
                    "error": "Metadata manager not initialized",
                    "suggestion": "Try kb_search for bill information instead",
                }

            bill_docs = self._metadata_manager.filter_by_category("bills", limit=50)
            bill_name_lower = bill_name.lower()
            matching = [
                doc
                for doc in bill_docs
                if bill_name_lower in doc.get("title", "").lower()
                or bill_name_lower in doc.get("source_name", "").lower()
            ]

            if matching:
                bill = matching[0]
                meta = bill.get("metadata", {})
                return {
                    "success": True,
                    "bill_name": bill.get("title", bill_name),
                    "status": meta.get("status", "Unknown"),
                    "vote_count": meta.get("vote_count", "Not available"),
                    "stage": meta.get("stage", "Not specified"),
                    "date": meta.get("date", bill.get("date")),
                    "session": meta.get("session", "Unknown"),
                    "sponsor": meta.get("sponsor", "Unknown"),
                    "summary": bill.get("summary", "")[:300],
                }

            return {
                "success": False,
                "error": f"Bill '{bill_name}' not found",
                "suggestion": "Try kb_search with a more specific name or year",
            }
        except Exception as e:
            logger.error(f"Bill lookup failed: {e}")
            return {"success": False, "error": str(e)}
