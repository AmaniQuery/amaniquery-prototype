"""
Agent Router - Routes tasks to appropriate agents
"""

from typing import Dict, Any, List, Optional
from loguru import logger


class AgentRouter:
    """
    Routes tasks to appropriate specialist agents
    """

    def __init__(self):
        """Initialize router"""
        self.routing_rules = {
            "research": ["researcher", "validator"],
            "citation": ["citer"],
            "editing": ["editor"],
            "validation": ["validator"],
            "comprehensive": ["researcher", "validator", "editor", "citer"],
        }

    def route(self, task_type: str, query: str) -> List[str]:
        """
        Route a task to appropriate agents

        Args:
            task_type: Type of task
            query: Query to analyze for routing

        Returns:
            List of agent types to use
        """
        # Check explicit routing rules
        if task_type in self.routing_rules:
            return self.routing_rules[task_type]

        # Intelligent routing based on query
        query_lower = query.lower()

        if "cite" in query_lower or "source" in query_lower:
            return ["researcher", "citer"]
        elif "edit" in query_lower or "format" in query_lower:
            return ["editor"]
        elif "validate" in query_lower or "check" in query_lower:
            return ["validator"]
        elif "research" in query_lower:
            return ["researcher", "validator"]
        else:
            # Default: comprehensive
            return ["researcher", "validator", "editor", "citer"]

    def add_route(self, task_type: str, agent_chain: List[str]):
        """
        Add a custom routing rule

        Args:
            task_type: Task type
            agent_chain: List of agent types
        """
        self.routing_rules[task_type] = agent_chain
        logger.info(f"Added routing rule: {task_type} -> {agent_chain}")
