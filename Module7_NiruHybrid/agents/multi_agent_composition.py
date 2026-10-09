"""
Multi-Agent Composition for Hybrid Module
Specialist agents: Citer, Editor, Validator
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Module4_NiruAPI.agents.composition.sub_agent_factory import (
    SubAgentFactory,
    SubAgent,
)
from Module4_NiruAPI.agents.composition.agent_orchestrator import AgentOrchestrator


class MultiAgentComposition:
    """
    Multi-agent composition for hybrid module

    Coordinates specialist agents (Citer, Editor, Validator) via agent orchestration.
    Implements error handling and graceful degradation.
    """

    def __init__(self):
        """Initialize multi-agent composition"""
        try:
            self.orchestrator = AgentOrchestrator()
            self.citer_agent = SubAgentFactory.create_agent("citer")
            self.editor_agent = SubAgentFactory.create_agent("editor")
            self.validator_agent = SubAgentFactory.create_agent("validator")

            self.processing_count = 0
            self.error_count = 0
            logger.info("Multi-agent composition initialized")
        except Exception as e:
            logger.error(f"Failed to initialize multi-agent composition: {e}")
            self.orchestrator = None
            self.citer_agent = None
            self.editor_agent = None
            self.validator_agent = None
            self.processing_count = 0
            self.error_count = 0

    def process_with_agents(
        self,
        content: str,
        sources: List[Dict[str, Any]],
        use_citer: bool = True,
        use_editor: bool = True,
        use_validator: bool = True,
    ) -> Dict[str, Any]:
        """
        Process content through specialist agents

        Args:
            content: Content to process
            sources: Sources for citation
            use_citer: Whether to use citer agent
            use_editor: Whether to use editor agent
            use_validator: Whether to use validator agent

        Returns:
            Processed content with agent outputs
        """
        if not self.orchestrator:
            logger.warning("Orchestrator not available, returning original content")
            return {
                "final_result": {"content": content, "sources": sources},
                "agent_chain": [],
                "error": "Orchestrator not available",
            }

        if not content:
            return {
                "final_result": {"content": "", "sources": sources},
                "agent_chain": [],
                "error": "Empty content",
            }

        self.processing_count += 1

        try:
            agent_chain = []

            if use_validator and self.validator_agent:
                agent_chain.append("validator")
            if use_editor and self.editor_agent:
                agent_chain.append("editor")
            if use_citer and self.citer_agent:
                agent_chain.append("citer")

            if not agent_chain:
                logger.warning("No agents available for processing")
                return {
                    "final_result": {"content": content, "sources": sources},
                    "agent_chain": [],
                    "error": "No agents available",
                }

            input_data = {"content": content, "sources": sources or []}

            result = self.orchestrator.orchestrate(
                task="hybrid_processing", input_data=input_data, agent_chain=agent_chain
            )

            result["processing_metadata"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "agents_used": agent_chain,
            }

            return result
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error in multi-agent processing: {e}")
            return {
                "final_result": {"content": content, "sources": sources},
                "agent_chain": [],
                "error": str(e),
            }

    def cite_sources(self, content: str, sources: List[Dict[str, Any]]) -> str:
        """Add citations to content using Citer agent"""
        if not self.citer_agent:
            logger.warning("Citer agent not available")
            return content

        try:
            result = self.citer_agent.process(
                {"content": content, "sources": sources or []}
            )
            return result.get("formatted_content", content)
        except Exception as e:
            logger.error(f"Error in citation: {e}")
            return content

    def edit_content(self, content: str) -> str:
        """Edit content using Editor agent"""
        if not self.editor_agent:
            logger.warning("Editor agent not available")
            return content

        try:
            result = self.editor_agent.process({"content": content})
            return result.get("edited_content", content)
        except Exception as e:
            logger.error(f"Error in editing: {e}")
            return content

    def validate_content(
        self, content: str, sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate content using Validator agent"""
        if not self.validator_agent:
            return {
                "validated": True,
                "validation_score": 1.0,
                "note": "Validator not available",
            }

        try:
            result = self.validator_agent.process(
                {"content": content, "sources": sources or []}
            )
            return result
        except Exception as e:
            logger.error(f"Error in validation: {e}")
            return {"validated": False, "validation_score": 0.0, "error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "processing_count": self.processing_count,
            "error_count": self.error_count,
            "citer_available": self.citer_agent is not None,
            "editor_available": self.editor_agent is not None,
            "validator_available": self.validator_agent is not None,
            "orchestrator_available": self.orchestrator is not None,
        }
