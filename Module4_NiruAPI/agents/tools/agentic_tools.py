"""
Agentic RAG Tool Registry
Extended tools specifically for ReAct agent
"""

from typing import Dict, Any, Optional, List
from loguru import logger

from Module4_NiruAPI.agents.tools.kb_search import KnowledgeBaseSearchTool

# =============================================================================
# AGENTIC RAG TOOLS
# =============================================================================


class BillStatusTool:
    """Look up Kenyan bill status and voting information"""

    def __init__(self, metadata_manager=None):
        self.metadata_manager = metadata_manager
        self.name = "lookup_bill_status"
        self.description = "Get current status, voting results, and metadata for a Kenyan parliamentary bill"

    def execute(self, bill_name: str) -> Dict[str, Any]:
        """Execute bill status lookup"""
        try:
            logger.info(f"[Tool] lookup_bill_status: '{bill_name}'")

            if not self.metadata_manager:
                return {"success": False, "error": "Metadata manager not initialized"}

            # Search for bill in metadata using category filter
            try:
                # Search bills category
                bill_docs = self.metadata_manager.filter_by_category("bills", limit=50)

                # Find matching bill by name (case-insensitive)
                bill_name_lower = bill_name.lower()
                matching_bills = [
                    doc
                    for doc in bill_docs
                    if bill_name_lower in doc.get("title", "").lower()
                    or bill_name_lower in doc.get("source_name", "").lower()
                ]

                if matching_bills:
                    # Use first match
                    bill = matching_bills[0]
                    metadata = bill.get("metadata", {})

                    return {
                        "success": True,
                        "bill_name": bill.get("title", bill_name),
                        "status": metadata.get("status", "Unknown"),
                        "vote_count": metadata.get("vote_count", "Not available"),
                        "date": metadata.get("date", bill.get("date")),
                        "session": metadata.get("session", "Unknown"),
                        "summary": bill.get("summary", "")[:200],
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Bill '{bill_name}' not found in knowledge base",
                        "suggestion": "Try searching with a more specific name or year",
                    }
            except Exception as e:
                logger.error(f"Error searching bill metadata: {e}")
                return {"success": False, "error": str(e)}

        except Exception as e:
            logger.error(f"Bill lookup failed: {e}")
            return {"success": False, "error": str(e)}


class HansardTool:
    """Fetch Hansard parliamentary debates"""

    def __init__(self, vector_store=None):
        self.vector_store = vector_store
        self.name = "fetch_hansard"
        self.description = "Retrieve parliamentary debate transcripts (Hansard) by date and optional speaker"

    def execute(self, date: str, speaker: str = "any") -> Dict[str, Any]:
        """Execute Hansard fetch"""
        try:
            logger.info(f"[Tool] fetch_hansard: date={date}, speaker={speaker}")

            if not self.vector_store:
                return {"success": False, "error": "Vector store not initialized"}

            # Build search query
            query_filters = {"category": "hansard", "date": date}
            if speaker and speaker != "any":
                query_filters["speaker"] = speaker

            # Search hansard
            results = self.vector_store.search(
                query=f"parliamentary debate {date}", filters=query_filters, top_k=10
            )

            return {
                "success": True,
                "date": date,
                "speaker": speaker,
                "transcripts": results,
                "count": len(results),
            }
        except Exception as e:
            logger.error(f"Hansard fetch failed: {e}")
            return {"success": False, "error": str(e)}


class AgenticToolRegistry:
    """Registry specifically for Agentic RAG tools"""

    def __init__(self, vector_store=None, rag_pipeline=None, metadata_manager=None):
        self.tools: Dict[str, Any] = {}

        # Register knowledge base search (reuse existing)
        if vector_store and rag_pipeline:
            kb_tool = KnowledgeBaseSearchTool()
            kb_tool.vector_store = vector_store
            kb_tool.rag_pipeline = rag_pipeline
            self.tools["search_knowledge_base"] = kb_tool

        # Register bill status tool
        if metadata_manager:
            self.tools["lookup_bill_status"] = BillStatusTool(metadata_manager)

        # Register Hansard tool
        if vector_store:
            self.tools["fetch_hansard"] = HansardTool(vector_store)

        logger.info(f"Agentic tool registry initialized with {len(self.tools)} tools")

    def execute(self, tool_name: str, **params) -> Dict[str, Any]:
        """Execute a tool by name"""
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}",
                "available_tools": list(self.tools.keys()),
            }

        try:
            tool = self.tools[tool_name]
            result = tool.execute(**params)
            logger.info(f"Tool {tool_name} executed")
            return result
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {"success": False, "error": str(e)}

    def get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions for ReAct prompt"""
        descriptions = []
        for name, tool in self.tools.items():
            descriptions.append(f"- {name}: {tool.description}")
        return "\n".join(descriptions)


# =============================================================================
# GLOBAL REGISTRY
# =============================================================================

_agentic_registry: Optional[AgenticToolRegistry] = None


def get_agentic_tools() -> Optional[AgenticToolRegistry]:
    """Get the global agentic tool registry"""
    return _agentic_registry


def initialize_agentic_tools(
    vector_store, rag_pipeline, metadata_manager
) -> AgenticToolRegistry:
    """Initialize the global agentic tool registry"""
    global _agentic_registry
    _agentic_registry = AgenticToolRegistry(
        vector_store, rag_pipeline, metadata_manager
    )
    return _agentic_registry
