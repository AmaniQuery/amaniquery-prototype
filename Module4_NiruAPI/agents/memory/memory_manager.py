"""
Agent Memory Manager - Manages short-term, long-term, episodic, and semantic memory
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from Module3_NiruDB.chat_manager import ChatDatabaseManager
from Module3_NiruDB.vector_store import VectorStore


class AgentMemoryManager:
    """
    Manages agent memory across multiple layers:
    - Short-term: Current session/working memory
    - Long-term: PostgreSQL + Vector stores
    - Episodic: Who, what, when, how
    - Semantic: Facts and rules
    """

    def __init__(
        self,
        chat_manager: Optional[ChatDatabaseManager] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        """
        Initialize memory manager

        Args:
            chat_manager: Chat manager for conversation history
            vector_store: Vector store for semantic memory
        """
        self.chat_manager = chat_manager or ChatDatabaseManager()
        self.vector_store = vector_store or VectorStore()

        # Short-term memory (in-memory)
        self.working_memory: Dict[str, Any] = {}
        self.episodic_memory: List[Dict[str, Any]] = []

    def store_episode(
        self,
        query: str,
        actions: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        reflection: Optional[str] = None,
    ):
        """
        Store an episodic memory (who, what, when, how)

        Args:
            query: Query that was processed
            actions: Actions taken
            tools: Tools used
            reflection: Reflection on the process
        """
        episode = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "actions": actions,
            "tools": tools,
            "reflection": reflection,
            "type": "episodic",
        }

        self.episodic_memory.append(episode)

        # Keep only last 100 episodes in short-term
        if len(self.episodic_memory) > 100:
            # Move older episodes to long-term storage
            old_episodes = self.episodic_memory[:-100]
            self._store_episodes_long_term(old_episodes)
            self.episodic_memory = self.episodic_memory[-100:]

        logger.debug(f"Stored episodic memory: {query[:50]}...")

    def _store_episodes_long_term(self, episodes: List[Dict[str, Any]]):
        """Store episodes in long-term storage"""
        try:
            # Store in vector store for semantic search
            texts = []
            metadatas = []

            for episode in episodes:
                text = f"Query: {episode['query']}\nReflection: {episode.get('reflection', '')}"
                metadata = {
                    "type": "episodic",
                    "timestamp": episode["timestamp"],
                    "actions_count": len(episode.get("actions", [])),
                    "tools_count": len(episode.get("tools", [])),
                }

                texts.append(text)
                metadatas.append(metadata)

            if texts:
                self.vector_store.add_documents(texts=texts, metadatas=metadatas)
        except Exception as e:
            logger.error(f"Error storing episodes to long-term memory: {e}")

    def store_semantic(self, facts: List[Dict[str, Any]]):
        """
        Store semantic memory (facts and rules)

        Args:
            facts: List of facts to store, each with 'fact' and optional 'metadata'
        """
        try:
            texts = []
            metadatas = []

            for fact_data in facts:
                fact = fact_data.get("fact", "")
                metadata = fact_data.get("metadata", {})
                metadata["type"] = "semantic"
                metadata["timestamp"] = datetime.utcnow().isoformat()

                texts.append(fact)
                metadatas.append(metadata)

            if texts:
                self.vector_store.add_documents(texts=texts, metadatas=metadatas)
                logger.debug(f"Stored {len(texts)} semantic facts")
        except Exception as e:
            logger.error(f"Error storing semantic memory: {e}")

    def recall_episodic(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Recall episodic memories relevant to a query

        Args:
            query: Query to search for
            top_k: Number of memories to return

        Returns:
            List of relevant episodic memories
        """
        # Search in short-term memory first
        relevant = []
        query_lower = query.lower()

        for episode in self.episodic_memory:
            if query_lower in episode["query"].lower():
                relevant.append(episode)

        # If not enough in short-term, search long-term
        if len(relevant) < top_k:
            try:
                long_term_results = self.vector_store.search(
                    query, top_k=top_k - len(relevant), filter={"type": "episodic"}
                )

                for result in long_term_results:
                    relevant.append(
                        {
                            "query": result.get("metadata", {}).get("query", ""),
                            "timestamp": result.get("metadata", {}).get(
                                "timestamp", ""
                            ),
                            "type": "episodic",
                            "source": "long_term",
                        }
                    )
            except Exception as e:
                logger.error(f"Error recalling long-term episodic memory: {e}")

        return relevant[:top_k]

    def recall_semantic(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Recall semantic memories (facts) relevant to a query

        Args:
            query: Query to search for
            top_k: Number of facts to return

        Returns:
            List of relevant semantic memories
        """
        try:
            results = self.vector_store.search(
                query, top_k=top_k, filter={"type": "semantic"}
            )

            facts = []
            for result in results:
                facts.append(
                    {
                        "fact": result.get("content", ""),
                        "metadata": result.get("metadata", {}),
                        "score": result.get("score", 0.0),
                    }
                )

            return facts
        except Exception as e:
            logger.error(f"Error recalling semantic memory: {e}")
            return []

    def get_working_memory(self, key: str) -> Optional[Any]:
        """Get value from working memory"""
        return self.working_memory.get(key)

    def set_working_memory(self, key: str, value: Any):
        """Set value in working memory"""
        self.working_memory[key] = value

    def clear_working_memory(self):
        """Clear working memory"""
        self.working_memory.clear()

    def get_conversation_history(
        self, session_id: str, max_turns: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history from long-term storage

        Args:
            session_id: Session ID
            max_turns: Maximum number of turns to return

        Returns:
            List of conversation turns
        """
        try:
            # Use chat manager to get messages
            messages = self.chat_manager.get_messages(session_id, limit=max_turns * 2)
            # Format as turns
            history = []
            for msg in messages:
                history.append(
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": (
                            msg.created_at.isoformat()
                            if hasattr(msg.created_at, "isoformat")
                            else str(msg.created_at)
                        ),
                    }
                )
            return history
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
