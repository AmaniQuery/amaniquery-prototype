"""
Agent Orchestrator - Coordinates multiple specialist agents
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from .sub_agent_factory import SubAgentFactory, SubAgent


class AgentOrchestrator:
    """
    Orchestrates multiple specialist agents to work together
    """

    def __init__(self):
        """Initialize agent orchestrator"""
        self.agents: Dict[str, SubAgent] = {}
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize all available agents"""
        agent_types = SubAgentFactory.list_agent_types()
        for agent_type in agent_types:
            agent = SubAgentFactory.create_agent(agent_type)
            if agent:
                self.agents[agent_type] = agent
                logger.debug(f"Initialized agent: {agent_type}")

    def orchestrate(
        self,
        task: str,
        input_data: Dict[str, Any],
        agent_chain: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Orchestrate multiple agents to complete a task

        Args:
            task: Task description
            input_data: Input data for agents
            agent_chain: Optional list of agent types to use (default: all)

        Returns:
            Orchestrated result
        """
        if agent_chain is None:
            # Default chain: researcher -> validator -> editor -> citer
            agent_chain = ["researcher", "validator", "editor", "citer"]

        current_data = input_data.copy()
        agent_results = []

        for agent_type in agent_chain:
            agent = self.agents.get(agent_type)
            if not agent:
                logger.warning(f"Agent {agent_type} not available, skipping")
                continue

            try:
                result = agent.process(current_data)
                agent_results.append(result)

                # Update current data with result for next agent
                if "content" in result:
                    current_data["content"] = result.get("content") or result.get(
                        "edited_content", ""
                    )
                if "sources" in result:
                    current_data["sources"] = result.get(
                        "sources", current_data.get("sources", [])
                    )

                logger.debug(
                    f"Agent {agent_type} completed: {result.get('task', 'unknown')}"
                )
            except Exception as e:
                logger.error(f"Error in agent {agent_type}: {e}")
                agent_results.append({"agent": agent_type, "error": str(e)})

        return {
            "task": task,
            "agent_chain": agent_chain,
            "agent_results": agent_results,
            "final_result": current_data,
        }
