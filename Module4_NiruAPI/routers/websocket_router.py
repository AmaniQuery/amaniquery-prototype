"""
WebSocket Router for Real-time News Updates
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List
from loguru import logger
import json
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f"WebSocket connected. Total connections: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(
            f"WebSocket disconnected. Total connections: {len(self.active_connections)}"
        )

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return

        message_json = json.dumps(message)
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.error(f"Error sending message to WebSocket: {e}")
                disconnected.append(connection)

        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)


# Global connection manager
connection_manager = ConnectionManager()


@router.websocket("/ws/news/stream")
async def news_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time news updates"""
    await connection_manager.connect(websocket)

    try:
        # Send welcome message
        await websocket.send_json(
            {"type": "connected", "message": "Connected to news stream"}
        )

        # Keep connection alive and listen for messages
        while True:
            try:
                # Wait for client message (ping/pong or subscription)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)

                try:
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif message.get("type") == "subscribe":
                        # Handle subscription to specific sources/categories
                        await websocket.send_json(
                            {
                                "type": "subscribed",
                                "sources": message.get("sources", []),
                                "categories": message.get("categories", []),
                            }
                        )
                except json.JSONDecodeError:
                    pass

            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({"type": "keepalive"})

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(websocket)


def broadcast_new_article(article: dict):
    """Broadcast a new article to all connected WebSocket clients"""
    message = {"type": "new_article", "article": article}
    # Run in event loop if available
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(connection_manager.broadcast(message))
        else:
            loop.run_until_complete(connection_manager.broadcast(message))
    except:
        # Fallback: try to get or create event loop
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(connection_manager.broadcast(message))
        except:
            logger.warning("Could not broadcast article to WebSocket clients")
