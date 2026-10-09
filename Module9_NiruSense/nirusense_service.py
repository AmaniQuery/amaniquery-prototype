"""
NiruSense orchestrator startup service.
Integrated with AmaniQuery backend.
"""

import asyncio
import threading
from loguru import logger
from Module9_NiruSense.processing.orchestrator import main as orchestrator_main


def start_nirusense_orchestrator():
    """Start the NiruSense processing orchestrator in a separate thread"""
    import asyncio

    loop = None
    try:
        logger.info("[AI] Initializing NiruSense Processing Pipeline...")

        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        logger.info("[AI] Starting NiruSense orchestrator...")
        logger.info("   Processing documents from Redis stream")

        # Run the orchestrator in the isolated event loop
        loop.run_until_complete(orchestrator_main())

    except KeyboardInterrupt:
        logger.info("NiruSense orchestrator interrupted")
    except Exception as e:
        logger.error(f"[ERROR] Failed to start NiruSense orchestrator: {e}")
        import traceback

        logger.error(traceback.format_exc())
    finally:
        # Properly close the event loop
        if loop is not None and not loop.is_closed():
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception:
                pass
            finally:
                loop.close()
                logger.debug("NiruSense event loop closed")


def start_nirusense_thread():
    """Start NiruSense orchestrator in background thread"""
    try:
        nirusense_thread = threading.Thread(
            target=start_nirusense_orchestrator, daemon=True, name="NiruSense"
        )
        nirusense_thread.start()
        logger.info("[OK] NiruSense thread started")

        # Give the thread a moment to initialize
        import time

        time.sleep(1.0)

        if nirusense_thread.is_alive():
            logger.info("[OK] NiruSense orchestrator is running")

            # Start scheduler if enabled
            try:
                from Module9_NiruSense.scheduler import get_nirusense_scheduler

                scheduler = get_nirusense_scheduler()
                if scheduler.start():
                    logger.info("[OK] NiruSense scheduler started")
            except Exception as e:
                logger.warning(f"NiruSense scheduler not started: {e}")

            return True
        else:
            logger.error("[ERROR] NiruSense thread died immediately")
            return False

    except Exception as e:
        logger.error(f"[ERROR] Failed to start NiruSense thread: {e}")
        return False
