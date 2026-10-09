"""
ReAct Agent Node for Agentic RAG (Modern Implementation)
Implements standard ReAct pattern using native tool calling and LangGraph state.
"""

import json
from typing import Dict, Any, List, Literal, Annotated
from loguru import logger
from pydantic import BaseModel, Field

# LangChain/LangGraph imports
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool

# Import AmaniQ components
from Module4_NiruAPI.agents.tools.agentic_tools import get_agentic_tools

# =============================================================================
# TOOL SCHEMAS (Pydantic)
# =============================================================================


class BillStatusSchema(BaseModel):
    """Arguments for lookup_bill_status tool"""

    bill_name: str = Field(
        ..., description="Name of the bill to look up (e.g., 'Finance Bill 2024')"
    )


class HansardSchema(BaseModel):
    """Arguments for fetch_hansard tool"""

    date: str = Field(..., description="Date of the debate in YYYY-MM-DD format")
    speaker: str = Field(
        default="any", description="Name of the speaker to filter by (optional)"
    )


class KBSearchSchema(BaseModel):
    """Arguments for search_knowledge_base tool"""

    query: str = Field(..., description="Search query for the knowledge base")
    category: str = Field(
        default="all",
        description="Category filter: 'law', 'parliament', 'news', or 'all'",
    )


# =============================================================================
# REACT REASONING NODE
# =============================================================================

REACT_SYSTEM_PROMPT = """You are AmaniQuery, a senior legal researcher for Kenya.
Your goal is to answer the user's complex legal or parliamentary query by gathering facts step-by-step.

You have access to the following tools:
1. `lookup_bill_status(bill_name)`: Get status and voting records of a bill.
2. `fetch_hansard(date, speaker)`: Get parliamentary debates.
3. `search_knowledge_base(query, category)`: Search Kenyan laws, cases, and news.

**Rules:**
- Use tools to gather facts. Do NOT hallucinate legal provisions.
- You can call multiple tools in one turn if needed.
- If a tool fails, try a different search term or strategy.
- When you have enough information, output your final answer directly (without tool calls).
- Your final answer must be comprehensive, citing specific sections/dates found.

Current Date: 2025-01-15
"""


def react_reasoning_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ReAct reasoning node - Decides next action (tool call or final answer).
    Uses native tool calling if supported, or structured prompting.
    """
    logger.info("=== REACT REASONING NODE ===")

    # Get inputs
    messages = state.get("react_messages", [])
    if not messages:
        # First turn: Initialize with system prompt and user query
        query = state.get("current_query", "")
        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

    # Initialize client (import locally to avoid circular import)
    from Module4_NiruAPI.agents.amaniq_v2 import MoonshotClient, AmaniQConfig

    config = AmaniQConfig()
    client = MoonshotClient.get_client(config)

    # Define tools for the model
    tools_schema = [
        {
            "type": "function",
            "function": {
                "name": "lookup_bill_status",
                "description": "Get current status, voting results, and metadata for a Kenyan parliamentary bill",
                "parameters": BillStatusSchema.model_json_schema(),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_hansard",
                "description": "Retrieve parliamentary debate transcripts (Hansard)",
                "parameters": HansardSchema.model_json_schema(),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": "Search the cloud knowledge base for Kenyan legal content",
                "parameters": KBSearchSchema.model_json_schema(),
            },
        },
    ]

    try:
        # Call LLM
        response = client.chat.completions.create(
            model="moonshot-v1-32k",
            messages=messages,
            tools=tools_schema,
            tool_choice="auto",  # Let model decide
            temperature=0.3,
            max_tokens=1000,
        )

        message = response.choices[0].message

        # Update history
        new_messages = list(messages)
        new_messages.append(message.model_dump())

        # Check for tool calls
        if message.tool_calls:
            logger.info(f"[ReAct] Generated {len(message.tool_calls)} tool calls")
            return {
                "react_messages": new_messages,
                "react_last_message": message,
                "react_status": "continue",
            }
        else:
            # Final answer
            logger.info("[ReAct] Final answer generated")
            return {
                "react_messages": new_messages,
                "react_final_answer": message.content,
                "react_success": True,
                "react_status": "done",
            }

    except Exception as e:
        logger.error(f"[ReAct] Reasoning error: {e}")
        return {"react_failed": True, "error": str(e), "react_status": "error"}


# =============================================================================
# TOOL EXECUTION NODE
# =============================================================================


def react_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ReAct tool execution node - Executes scheduled tools.
    """
    logger.info("=== REACT TOOL NODE ===")

    last_message = state.get("react_last_message")
    if not last_message or not last_message.tool_calls:
        logger.warning("[ReAct] No tool calls to execute")
        return {"react_status": "continue"}

    # Get tool registry
    tool_registry = get_agentic_tools()
    if not tool_registry:
        logger.error("Agentic tools not initialized")
        return {"error": "Tools unavailable", "react_status": "error"}

    messages = state.get("react_messages", [])
    new_messages = list(messages)

    for tool_call in last_message.tool_calls:
        function_name = tool_call.function.name
        arguments_str = tool_call.function.arguments
        call_id = tool_call.id

        logger.info(f"[ReAct] Executing {function_name}...")

        try:
            # Parse arguments
            args = json.loads(arguments_str)

            # Execute
            result = tool_registry.execute(function_name, **args)

            # Format output
            output_str = json.dumps(result, indent=2, default=str)

        except Exception as e:
            logger.error(f"[ReAct] Tool execution error: {e}")
            output_str = f"Error executing tool: {str(e)}"

        # Append tool output message (OpenAI format)
        new_messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": function_name,
                "content": output_str,
            }
        )

    return {"react_messages": new_messages, "react_status": "continue"}
