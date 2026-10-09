"""
Refactored Research Module using Agentic AI Architecture
"""

import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from loguru import logger

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from Module4_NiruAPI.agents.state_machine import AgenticResearchSystem
from Module4_NiruAPI.rag_pipeline import RAGPipeline
from Module4_NiruAPI.agents.swarm.swarm_orchestrator import SwarmOrchestrator
from Module4_NiruAPI.agents.tools.tool_registry import ToolRegistry
from Module4_NiruAPI.agents.memory.memory_manager import AgentMemoryManager
from Module4_NiruAPI.config_manager import ConfigManager


class AgenticResearchModule:
    """
    Refactored research module using 7-layer agentic AI architecture
    """

    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        """
        Initialize agentic research module

        Args:
            config_manager: Optional config manager
            tool_registry: Optional shared tool registry (recommended for performance)
        """
        self.config_manager = config_manager or ConfigManager()

        # Initialize components
        try:
            # Initialize RAG pipeline
            self.rag_pipeline = RAGPipeline(config_manager=config_manager)

            # Initialize swarm orchestrator
            self.swarm_orchestrator = SwarmOrchestrator(rag_pipeline=self.rag_pipeline)

            # Use shared tool registry if provided (avoids slow initialization)
            if tool_registry:
                self.tool_registry = tool_registry
                logger.info("Using shared tool registry (fast path)")
            else:
                self.tool_registry = ToolRegistry()
                logger.warning("Creating new tool registry (slow path)")

            # Initialize memory manager
            self.memory_manager = AgentMemoryManager()

            # Initialize agentic research system
            self.agentic_system = AgenticResearchSystem(
                rag_pipeline=self.rag_pipeline,
                swarm_orchestrator=self.swarm_orchestrator,
                tool_registry=self.tool_registry,
                memory_manager=self.memory_manager,
                max_iterations=5,
            )

            logger.info("Agentic research module initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing agentic research module: {e}")
            raise

    async def analyze_legal_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a legal query using agentic AI system

        Args:
            query: The legal question or query to analyze
            context: Additional context about the query (optional)

        Returns:
            Dictionary containing detailed analysis results
        """
        try:
            # Use agentic system to research
            result = await self.agentic_system.research(query, context)

            # Extract detailed information from result
            final_answer = result.get("final_answer") or result.get("answer", "")
            sources = result.get("sources", [])
            tools_used = result.get("tools_used", [])
            reflection = result.get("reflection", "")
            plan = result.get("metadata", {}).get("plan", [])
            actions_taken = result.get("metadata", {}).get("actions_count", 0)

            # Format response with detailed, readable structure
            analysis_result = {
                "original_query": query,
                "analysis": {
                    "query_interpretation": self._format_query_interpretation(
                        query, final_answer, reflection
                    ),
                    "applicable_laws": self._extract_laws(result),
                    "legal_analysis": self._format_legal_analysis(
                        final_answer, sources
                    ),
                    "practical_guidance": {
                        "steps": self._extract_guidance_steps(result),
                        "summary": self._extract_guidance_summary(final_answer),
                    },
                    "additional_considerations": self._extract_considerations(result),
                    "research_process": {
                        "steps_completed": len(plan) if isinstance(plan, list) else 0,
                        "actions_taken": actions_taken,
                        "tools_used": [
                            tool.get("tool", tool) if isinstance(tool, dict) else tool
                            for tool in tools_used[:10]
                        ],
                        "sources_consulted": len(sources),
                    },
                },
                "model_used": "agentic-ai-system",
                "research_timestamp": result.get("metadata", {}).get(
                    "timestamp", datetime.utcnow().isoformat()
                ),
                "report_confidence": result.get("confidence", 0.0),
                "sources": sources,
                "tools_used": tools_used,
                "reflection": reflection,
            }

            return analysis_result
        except Exception as e:
            logger.error(f"Error in agentic legal query analysis: {e}")
            return {
                "original_query": query,
                "analysis": {
                    "query_interpretation": "Unable to complete research analysis due to technical error.",
                    "applicable_laws": ["Error occurred during analysis"],
                    "legal_analysis": f"Technical error prevented analysis completion: {str(e)}",
                    "practical_guidance": {"steps": ["Please try again later"]},
                    "additional_considerations": ["Contact support if issue persists"],
                },
                "model_used": "agentic-ai-system",
                "research_timestamp": datetime.utcnow().isoformat(),
                "report_confidence": 0.0,
                "error": f"Research analysis failed: {str(e)}",
            }

    def _extract_laws(self, result: Dict[str, Any]) -> List[str]:
        """Extract applicable laws from result"""
        sources = result.get("sources", [])
        laws = []

        for source in sources:
            if (
                source.get("type") == "legal"
                or "law" in source.get("title", "").lower()
            ):
                laws.append(source.get("title", "Unknown Law"))

        if not laws:
            laws = ["Researching relevant Kenyan laws..."]

        return laws

    def _extract_guidance_steps(self, result: Dict[str, Any]) -> List[str]:
        """Extract practical guidance steps from result"""
        answer = result.get("answer", "")

        # Look for numbered steps or bullet points
        steps = []
        lines = answer.split("\n")

        for line in lines:
            line = line.strip()
            # Check for numbered steps (1., 2., etc.) or bullets (-, •)
            if line and (
                line[0].isdigit() or line.startswith("-") or line.startswith("•")
            ):
                step = line.lstrip("0123456789.-• ").strip()
                if step:
                    steps.append(step)

        if not steps:
            steps = [
                "Review the analysis above",
                "Consult relevant legal resources",
                "Seek professional advice if needed",
            ]

        return steps[:10]  # Limit to 10 steps

    def _extract_considerations(self, result: Dict[str, Any]) -> List[str]:
        """Extract additional considerations from result"""
        answer = result.get("answer", "")
        reflection = result.get("reflection", "")

        considerations = []

        # Extract from reflection
        if reflection:
            lines = reflection.split("\n")
            for line in lines:
                if "consider" in line.lower() or "note" in line.lower():
                    considerations.append(line.strip())

        # Extract from answer
        if "consider" in answer.lower() or "important" in answer.lower():
            # Try to find consideration sentences
            sentences = answer.split(".")
            for sentence in sentences:
                if "consider" in sentence.lower() or "important" in sentence.lower():
                    considerations.append(sentence.strip())

        if not considerations:
            considerations = [
                "Review all applicable laws",
                "Consider recent legal developments",
                "Consult with legal professionals",
            ]

        return considerations[:5]  # Limit to 5 considerations

    def _format_query_interpretation(
        self, query: str, answer: str, reflection: str
    ) -> str:
        """Format query interpretation in a readable way"""
        if not answer:
            return "Analyzing your legal question to understand the key legal issues and requirements..."

        # Extract first 2-3 sentences as interpretation
        sentences = answer.split(".")
        interpretation = ". ".join(sentences[:3]).strip()
        if interpretation and not interpretation.endswith("."):
            interpretation += "."

        if reflection:
            reflection_snippet = (
                reflection[:200] + "..." if len(reflection) > 200 else reflection
            )
            interpretation += f"\n\n**Research Note:** {reflection_snippet}"

        return (
            interpretation
            or "Understanding the legal context and requirements of your question..."
        )

    def _format_legal_analysis(self, answer: str, sources: List[Dict[str, Any]]) -> str:
        """Format legal analysis in a detailed, readable way"""
        if not answer:
            return "Conducting detailed legal analysis based on Kenyan laws and regulations..."

        # Add source citations if available
        formatted = answer

        if sources:
            source_refs = []
            for i, source in enumerate(sources[:5], 1):
                title = source.get("title", "Source")
                url = source.get("url", "")
                if url:
                    source_refs.append(f"[{i}] {title}")
                else:
                    source_refs.append(f"[{i}] {title}")

            if source_refs:
                formatted += f"\n\n**Sources Consulted:**\n" + "\n".join(
                    f"- {ref}" for ref in source_refs
                )

        return formatted

    def _extract_guidance_summary(self, answer: str) -> str:
        """Extract a summary of practical guidance from the answer"""
        if not answer:
            return "Practical guidance will be provided based on the legal analysis."

        # Look for guidance-related sections
        guidance_keywords = [
            "should",
            "must",
            "recommend",
            "advise",
            "guidance",
            "steps",
            "action",
        ]
        sentences = answer.split(".")

        guidance_sentences = []
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in guidance_keywords):
                guidance_sentences.append(sentence.strip())

        if guidance_sentences:
            return ". ".join(guidance_sentences[:3]) + "."

        # Fallback: extract first actionable sentence
        for sentence in sentences:
            if len(sentence) > 20 and any(
                word in sentence.lower()
                for word in ["you", "your", "business", "company", "legal"]
            ):
                return sentence.strip() + "."

        return (
            "Review the legal analysis above for specific guidance on your situation."
        )

    def generate_legal_report(
        self, analysis_results: Dict[str, Any], report_focus: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive legal report based on query analysis

        Args:
            analysis_results: Results from analyze_legal_query
            report_focus: Type of report focus (comprehensive, constitutional, criminal, civil, administrative)

        Returns:
            Dictionary containing the legal report
        """
        analysis = analysis_results.get("analysis", {})

        # Use swarm orchestrator to generate report
        query = analysis_results.get("original_query", "Legal Research Query")
        context = {
            "analysis": analysis,
            "report_focus": report_focus,
            "sources": analysis_results.get("sources", []),
        }

        try:
            # Generate report using swarm
            report_query = f"""
            Based on the following legal analysis, generate a comprehensive {report_focus} legal report:
            
            {json.dumps(analysis, indent=2)}
            
            Focus on: {report_focus}
            """

            # Use async method
            import asyncio

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            report_content = loop.run_until_complete(
                self.swarm_orchestrator.query_with_context(report_query, context)
            )

            return {
                "report": {
                    "title": f"Legal Report - {report_focus.title()} Analysis",
                    "content": report_content,
                    "focus_area": report_focus,
                    "generated_at": datetime.utcnow().isoformat(),
                    "model_used": "agentic-ai-system",
                },
                "metadata": {
                    "analysis_timestamp": analysis_results.get("research_timestamp"),
                    "report_focus": report_focus,
                    "report_length": len(report_content),
                    "sources_count": len(analysis_results.get("sources", [])),
                },
                "source_analysis": analysis_results,
            }
        except Exception as e:
            logger.error(f"Error generating legal report: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "report_focus": report_focus,
            }

    def conduct_legal_research(
        self, legal_topics: List[str], research_questions: List[str]
    ) -> Dict[str, Any]:
        """
        Conduct legal research on specific topics

        Args:
            legal_topics: List of legal topics to research
            research_questions: List of specific research questions

        Returns:
            Dictionary containing legal research findings
        """
        try:
            # Combine topics and questions into a comprehensive query
            query = f"""
            Research the following legal topics: {', '.join(legal_topics)}
            
            Address these research questions:
            {chr(10).join(f'- {q}' for q in research_questions)}
            """

            # Use agentic system
            import asyncio

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(self.agentic_system.research(query))

            return {
                "legal_research": {
                    "findings": result.get("answer", ""),
                    "topics_researched": legal_topics,
                    "questions_addressed": research_questions,
                    "generated_at": datetime.utcnow().isoformat(),
                    "model_used": "agentic-ai-system",
                    "sources": result.get("sources", []),
                },
                "metadata": {
                    "topic_count": len(legal_topics),
                    "question_count": len(research_questions),
                    "findings_length": len(result.get("answer", "")),
                    "sources_count": len(result.get("sources", [])),
                },
            }
        except Exception as e:
            logger.error(f"Error conducting legal research: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "topics": legal_topics,
                "questions": research_questions,
            }
