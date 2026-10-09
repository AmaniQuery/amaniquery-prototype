"""
Sub-Agent Factory - Creates specialist agents (Researcher, Citer, Editor, Validator)
"""

from typing import Dict, Any, Optional
from loguru import logger
import os
import json

try:
    import litellm
except ImportError:
    litellm = None


def _llm_call(system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
    """Call an LLM via litellm with the given prompts."""
    if litellm is None:
        raise ImportError("litellm is required for agent processing")
    model = os.getenv("AGENT_MODEL", "gpt-3.5-turbo")
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


class SubAgent:
    """Base class for sub-agents"""

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return output"""
        raise NotImplementedError


class ResearcherAgent(SubAgent):
    """Specialist agent for research tasks"""

    def __init__(self):
        super().__init__("Researcher", "Conducts research and gathers information")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get("query", "")
        context = input_data.get("context", "")
        sources = input_data.get("sources", [])
        try:
            system_prompt = (
                "You are a legal research specialist for Kenyan law. "
                "Synthesize information from the provided context and sources "
                "to answer the query thoroughly. Cite sources inline as [1], [2], etc."
            )
            user_prompt = f"Query: {query}\n\nContext: {context}\n\nSources: {json.dumps(sources, indent=2)}"
            result = _llm_call(system_prompt, user_prompt, max_tokens=2048)
        except Exception as e:
            logger.warning(f"ResearcherAgent LLM failed, using fallback: {e}")
            source_summary = "\n".join(
                f"- {s.get('title', 'Source')}: {s.get('content', '')[:200]}"
                for s in sources[:5]
            )
            result = (
                f"Research findings for '{query}':\n\n{source_summary}"
                if source_summary
                else f"Research conducted on: {query}"
            )
        return {
            "agent": self.name,
            "task": "research",
            "query": query,
            "result": result,
        }


class CiterAgent(SubAgent):
    """Specialist agent for citation and source management"""

    def __init__(self):
        super().__init__("Citer", "Manages citations and sources")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        content = input_data.get("content", "")
        sources = input_data.get("sources", [])
        citations = []
        for i, source in enumerate(sources, 1):
            title = source.get("title", "Source")
            url = source.get("url", "")
            date_str = source.get("date", "")
            parts = [f"[{i}] {title}"]
            if url:
                parts.append(url)
            if date_str:
                parts.append(f"({date_str})")
            citations.append(" - ".join(parts))
        formatted = f"{content}\n\n---\n**Sources:**\n" + "\n".join(citations)
        return {
            "agent": self.name,
            "task": "citation",
            "citations": citations,
            "citation_count": len(citations),
            "formatted_content": formatted,
        }


class EditorAgent(SubAgent):
    """Specialist agent for editing and formatting"""

    def __init__(self):
        super().__init__("Editor", "Edits and formats content")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        content = input_data.get("content", "")
        target_audience = input_data.get("target_audience", "general")
        try:
            system_prompt = (
                f"You are a legal content editor. Polish the following text for a {target_audience} audience. "
                "Improve clarity, fix grammar, ensure professional tone, and preserve all citations and factual content."
            )
            edited = _llm_call(system_prompt, content, max_tokens=2048)
        except Exception as e:
            logger.warning(f"EditorAgent LLM failed, using fallback: {e}")
            edited = content.strip()
        return {
            "agent": self.name,
            "task": "editing",
            "original_length": len(content),
            "edited_length": len(edited),
            "edited_content": edited,
        }


class ValidatorAgent(SubAgent):
    """Specialist agent for validation and fact-checking"""

    def __init__(self):
        super().__init__("Validator", "Validates and fact-checks content")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        content = input_data.get("content", "")
        sources = input_data.get("sources", [])
        query = input_data.get("query", "")
        try:
            system_prompt = (
                "You are a legal fact-checker. Validate the following content against its sources. "
                "Return a JSON object with: validated (bool), score (0-1), issues (list of strings), "
                "and suggestions (list of strings)."
            )
            user_prompt = f"Query: {query}\n\nContent: {content}\n\nSources: {json.dumps(sources, indent=2)}"
            raw = _llm_call(system_prompt, user_prompt, max_tokens=1024)
            result = json.loads(raw)
        except Exception as e:
            logger.warning(
                f"ValidatorAgent LLM failed, using heuristic validation: {e}"
            )
            has_sources = len(sources) > 0
            has_content = len(content) > 0
            is_well_formed = content.count(".") > 1
            score = sum([has_sources, has_content, is_well_formed]) / 3.0
            result = {
                "validated": score >= 0.7,
                "score": round(score, 2),
                "issues": [] if score >= 0.7 else ["Content may be incomplete"],
                "suggestions": [],
            }
        return {
            "agent": self.name,
            "task": "validation",
            "validation_score": result.get("score", 0),
            "validated": result.get("validated", False),
            "issues": result.get("issues", []),
            "suggestions": result.get("suggestions", []),
            "checks": {
                "has_sources": len(sources) > 0,
                "has_content": len(content) > 0,
                "is_well_formed": content.count(".") > 1,
            },
        }


class SubAgentFactory:
    """Factory for creating specialist sub-agents"""

    _agents = {
        "researcher": ResearcherAgent,
        "citer": CiterAgent,
        "editor": EditorAgent,
        "validator": ValidatorAgent,
    }

    @classmethod
    def create_agent(cls, agent_type: str) -> Optional[SubAgent]:
        """
        Create a sub-agent

        Args:
            agent_type: Type of agent (researcher, citer, editor, validator)

        Returns:
            Sub-agent instance or None
        """
        agent_class = cls._agents.get(agent_type.lower())
        if agent_class:
            return agent_class()
        else:
            logger.warning(f"Unknown agent type: {agent_type}")
            return None

    @classmethod
    def list_agent_types(cls) -> list:
        """List available agent types"""
        return list(cls._agents.keys())
