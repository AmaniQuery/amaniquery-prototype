"""
File Writer Tool - Writes content to files
"""

from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from loguru import logger


class FileWriterTool:
    """Tool for writing content to files"""

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize file writer

        Args:
            base_path: Base directory for file operations (defaults to ./output)
        """
        self.base_path = Path(base_path or "./output")
        self.base_path.mkdir(parents=True, exist_ok=True)

    def execute(
        self, filename: str, content: str, mode: str = "w", encoding: str = "utf-8"
    ) -> Dict[str, Any]:
        """
        Write content to a file

        Args:
            filename: Name of the file (relative to base_path)
            content: Content to write
            mode: Write mode ('w' for overwrite, 'a' for append)
            encoding: File encoding

        Returns:
            Write operation result
        """
        try:
            # Ensure filename is safe (no path traversal)
            safe_filename = Path(filename).name
            file_path = self.base_path / safe_filename

            # Write file
            with open(file_path, mode, encoding=encoding) as f:
                f.write(content)

            return {
                "filename": str(file_path),
                "bytes_written": len(content.encode(encoding)),
                "success": True,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error writing file {filename}: {e}")
            return {"filename": filename, "error": str(e), "success": False}
