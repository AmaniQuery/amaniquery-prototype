"""
Tool Registry - Manages all available tools and tool-chaining
"""

from typing import Dict, Any, Optional, List
from loguru import logger

from .web_search import WebSearchTool
from .twitter_scraper import TwitterScraperTool
from .news_search import NewsSearchTool
from .youtube_search import YouTubeSearchTool
from .url_fetcher import URLFetcherTool
from .calculator import CalculatorTool
from .file_writer import FileWriterTool
from .email_drafter import EmailDrafterTool
from .kb_search import KnowledgeBaseSearchTool
from .bill_status import BillStatusTool
from .hansard_tool import HansardTool
from .fees_calculator import FeesCalculatorTool
from .legal_citation import LegalCitationTool


class ToolRegistry:
    """
    Registry for all available tools
    Supports tool registration, execution, and chaining
    """

    def __init__(self):
        """Initialize tool registry with all available tools"""
        self.tools: Dict[str, Any] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register all default tools"""
        try:
            self.register_tool("web_search", WebSearchTool())
        except Exception as e:
            logger.warning(f"Failed to register web_search tool: {e}")

        try:
            self.register_tool("twitter_search", TwitterScraperTool())
        except Exception as e:
            logger.warning(f"Failed to register twitter_search tool: {e}")

        try:
            self.register_tool("news_search", NewsSearchTool())
        except Exception as e:
            logger.warning(f"Failed to register news_search tool: {e}")

        try:
            self.register_tool("youtube_search", YouTubeSearchTool())
        except Exception as e:
            logger.warning(f"Failed to register youtube_search tool: {e}")

        try:
            self.register_tool("url_fetch", URLFetcherTool())
        except Exception as e:
            logger.warning(f"Failed to register url_fetch tool: {e}")

        try:
            self.register_tool("calculator", CalculatorTool())
        except Exception as e:
            logger.warning(f"Failed to register calculator tool: {e}")

        try:
            self.register_tool("file_write", FileWriterTool())
        except Exception as e:
            logger.warning(f"Failed to register file_write tool: {e}")

        try:
            self.register_tool("email_draft", EmailDrafterTool())
        except Exception as e:
            logger.warning(f"Failed to register email_draft tool: {e}")

        try:
            self.register_tool("kb_search", KnowledgeBaseSearchTool())
        except Exception as e:
            logger.warning(f"Failed to register kb_search tool: {e}")

        # New tools
        try:
            self.register_tool("file_write", FileWriterTool())
        except Exception as e:
            logger.warning(f"Failed to register file_write tool: {e}")

        try:
            self.register_tool("email_draft", EmailDrafterTool())
        except Exception as e:
            logger.warning(f"Failed to register email_draft tool: {e}")

        try:
            self.register_tool("bill_status", BillStatusTool())
        except Exception as e:
            logger.warning(f"Failed to register bill_status tool: {e}")

        try:
            self.register_tool("hansard", HansardTool())
        except Exception as e:
            logger.warning(f"Failed to register hansard tool: {e}")

        try:
            self.register_tool("fees_calculator", FeesCalculatorTool())
        except Exception as e:
            logger.warning(f"Failed to register fees_calculator tool: {e}")

        try:
            self.register_tool("legal_citation", LegalCitationTool())
        except Exception as e:
            logger.warning(f"Failed to register legal_citation tool: {e}")

        logger.info(f"Registered {len(self.tools)} tools")

    def register_tool(self, name: str, tool: Any):
        """Register a tool"""
        self.tools[name] = tool
        logger.debug(f"Registered tool: {name}")

    def get_tool(self, name: str) -> Optional[Any]:
        """Get a tool by name"""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """List all available tool names"""
        return list(self.tools.keys())

    def execute_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """
        Execute a tool

        Args:
            name: Tool name
            args: Tool arguments

        Returns:
            Tool result
        """
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(
                f"Tool '{name}' not found. Available tools: {self.list_tools()}"
            )

        try:
            # Check for LangChain BaseTool
            if hasattr(tool, "invoke"):
                return tool.invoke(args)
            elif hasattr(tool, "execute"):
                return tool.execute(**args)
            elif callable(tool):
                return tool(**args)
            else:
                raise ValueError(
                    f"Tool '{name}' is not callable and has no invoke/execute method"
                )
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            raise

    def chain_tools(self, tool_chain: List[Dict[str, Any]]) -> List[Any]:
        """
        Execute a chain of tools where output of one feeds into the next

        Args:
            tool_chain: List of tool calls, each with 'tool' and 'args'

        Returns:
            List of results from each tool
        """
        results = []
        previous_result = None

        for tool_call in tool_chain:
            tool_name = tool_call.get("tool")
            tool_args = tool_call.get("args", {})

            # If previous result exists, inject it into args
            if previous_result is not None:
                tool_args["previous_result"] = previous_result

            result = self.execute_tool(tool_name, tool_args)
            results.append(result)
            previous_result = result

        return results
