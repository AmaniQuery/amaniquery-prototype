"""
Agent Composition Layer - Multi-agent coordination
"""

from .agent_orchestrator import AgentOrchestrator
from .sub_agent_factory import SubAgentFactory
from .router import AgentRouter

__all__ = ["AgentOrchestrator", "SubAgentFactory", "AgentRouter"]
