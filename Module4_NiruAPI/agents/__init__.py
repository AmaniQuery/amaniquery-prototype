"""
Agentic AI System - 7-Layer Architecture
"""

from typing import Optional, Any

# Direct import of AgenticResearchSystem
from .state_machine import AgenticResearchSystem

__all__ = [
    "AgenticResearchSystem",
    "get_agentic_research_system",
]

# Lazy instance for singleton pattern
_agentic_system: Optional[AgenticResearchSystem] = None


def get_agentic_research_system() -> AgenticResearchSystem:
    """Get or create the agentic research system instance"""
    global _agentic_system
    if _agentic_system is None:
        _agentic_system = AgenticResearchSystem()
    return _agentic_system
