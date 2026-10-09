"""
Video Processor - Extract frames and audio from video files for RAG integration

Provides utilities for:
- Key frame extraction (fixed interval or scene-based)
- Audio track extraction for speech-to-text
- Video metadata extraction
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass, field
from loguru import logger

try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("OpenCV not available. Install with: pip install opencv-python")


@dataclass
class VideoMetadata:
    """Video file metadata"""

    duration: float  # Duration in seconds
    fps: float  # Frames per second
    width: int
    height: int
    total_frames: int
    codec: str
    has_audio: bool
    file_size: int  # Size in bytes

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "duration": self.duration,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames,
            "codec": self.codec,
            "has_audio": self.has_audio,
            "file_size": self.file_size,
        }


@dataclass
class ExtractedFrame:
    """Extracted video frame"""

    frame_path: str  # Path to saved frame image
    timestamp: float  # Timestamp in seconds
    frame_number: int
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "frame_path": self.frame_path,
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "metadata": self.metadata,
        }


@dataclass
class ExtractionResult:
    """Result of video extraction"""

    frames: List[ExtractedFrame]
    audio_path: Optional[str]  # Path to extracted audio file
    video_metadata: VideoMetadata
    output_dir: str

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "frames": [f.to_dict() for f in self.frames],
            "audio_path": self.audio_path,
            "video_metadata": self.video_metadata.to_dict(),
            "output_dir": self.output_dir,
        }


class VideoProcessor:
    """
    Video processor for extracting frames and audio from video files

    Supports multiple extraction strategies:
    - Fixed interval: Extract frames at regular intervals
    - Scene detection: Extract frames when scene changes (future enhancement)
    - Key frames: Extract I-frames only (future enhancement)
    """

    def __init__(
        self,
        temp_dir: Optional[str] = None,
        ffmpeg_path: Optional[str] = None,
        cleanup_on_error: bool = True,
    ):
        """
        Initialize video processor

        Args:
            temp_dir: Directory for temporary files (uses system temp if None)
            ffmpeg_path: Path to ffmpeg binary (auto-detect if None)
            cleanup_on_error: Whether to cleanup on errors
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.ffmpeg_path = ffmpeg_path or self._find_ffmpeg()
        self.cleanup_on_error = cleanup_on_error

        if not OPENCV_AVAILABLE:
            logger.warning("OpenCV not available - some features may be limited")

        logger.info(
            f"Video processor initialized (ffmpeg: {self.ffmpeg_path or 'not found'})"
        )

    def _find_ffmpeg(self) -> Optional[str]:
        """Find ffmpeg binary in system PATH"""
        ffmpeg_names = ["ffmpeg", "ffmpeg.exe"]

        for name in ffmpeg_names:
            path = shutil.which(name)
            if path:
                return path

        # Check common locations
        common_paths = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
        ]

        for path in common_paths:
            if os.path.exists(path):
                return path

        return None

    def _ensure_ffmpeg(self) -> str:
        """Ensure ffmpeg is available, raise error if not"""
        if not self.ffmpeg_path:
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg and ensure it's in PATH. "
                "See: https://ffmpeg.org/download.html"
            )
        return self.ffmpeg_path

    def get_video_metadata(self, video_path: Union[str, Path]) -> VideoMetadata:
        """
        Get metadata for a video file

        Args:
            video_path: Path to video file

        Returns:
            VideoMetadata object with video information
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        file_size = video_path.stat().st_size

        # Try OpenCV first (faster, no external dependency)
        if OPENCV_AVAILABLE:
            try:
                return self._get_metadata_opencv(video_path, file_size)
            except Exception as e:
                logger.debug(f"OpenCV metadata extraction failed: {e}")

        # Fall back to ffprobe
        return self._get_metadata_ffprobe(video_path, file_size)

    def _get_metadata_opencv(self, video_path: Path, file_size: int) -> VideoMetadata:
        """Get metadata using OpenCV"""
        cap = cv2.VideoCapture(str(video_path))

        try:
            if not cap.isOpened():
                raise RuntimeError("Failed to open video with OpenCV")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((codec_int >> 8 * i) & 0xFF) for i in range(4)])

            duration = total_frames / fps if fps > 0 else 0

            return VideoMetadata(
                duration=duration,
                fps=fps,
                width=width,
                height=height,
                total_frames=total_frames,
                codec=codec.strip(),
                has_audio=True,  # OpenCV can't detect audio, assume true
                file_size=file_size,
            )
        finally:
            cap.release()

    def _get_metadata_ffprobe(self, video_path: Path, file_size: int) -> VideoMetadata:
        """Get metadata using ffprobe"""
        ffmpeg = self._ensure_ffmpeg()
        ffprobe = ffmpeg.replace("ffmpeg", "ffprobe")

        if not os.path.exists(ffprobe):
            # Try without path modification
            ffprobe = shutil.which("ffprobe") or "ffprobe"

        cmd = [
            ffprobe,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(video_path),
        ]

        try:
            import json

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            data = json.loads(result.stdout)

            # Parse video stream
            video_stream = None
            has_audio = False

            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video" and not video_stream:
                    video_stream = stream
                elif stream.get("codec_type") == "audio":
                    has_audio = True

            if not video_stream:
                raise RuntimeError("No video stream found")

            format_info = data.get("format", {})

            fps_str = video_stream.get("r_frame_rate", "0/1")
            fps_num, fps_den = map(int, fps_str.split("/"))
            fps = fps_num / fps_den if fps_den > 0 else 0

            return VideoMetadata(
                duration=float(format_info.get("duration", 0)),
                fps=fps,
                width=int(video_stream.get("width", 0)),
                height=int(video_stream.get("height", 0)),
                total_frames=int(video_stream.get("nb_frames", 0))
                or int(float(format_info.get("duration", 0)) * fps),
                codec=video_stream.get("codec_name", "unknown"),
                has_audio=has_audio,
                file_size=file_size,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("ffprobe timed out")
        except json.JSONDecodeError:
            raise RuntimeError("Failed to parse ffprobe output")

    def extract_frames(
        self,
        video_path: Union[str, Path],
        output_dir: Optional[str] = None,
        num_frames: int = 10,
        strategy: str = "fixed_interval",
        image_format: str = "jpg",
        quality: int = 85,
    ) -> List[ExtractedFrame]:
        """
        Extract frames from video

        Args:
            video_path: Path to video file
            output_dir: Directory to save frames (creates temp if None)
            num_frames: Number of frames to extract
            strategy: Extraction strategy ('fixed_interval', 'first_last', 'keyframes')
            image_format: Output image format ('jpg', 'png')
            quality: JPEG quality (1-100)

        Returns:
            List of ExtractedFrame objects
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Create output directory
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="video_frames_", dir=self.temp_dir)
        else:
            os.makedirs(output_dir, exist_ok=True)

        # Get video metadata
        metadata = self.get_video_metadata(video_path)

        logger.info(
            f"Extracting {num_frames} frames from {video_path.name} ({metadata.duration:.1f}s)"
        )

        # Calculate frame timestamps based on strategy
        if strategy == "fixed_interval":
            timestamps = self._calculate_fixed_interval_timestamps(
                duration=metadata.duration,
                num_frames=num_frames,
            )
        elif strategy == "first_last":
            timestamps = [0.0, metadata.duration - 0.1]
        elif strategy == "keyframes":
            # Future: Use scene detection
            timestamps = self._calculate_fixed_interval_timestamps(
                duration=metadata.duration,
                num_frames=num_frames,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Extract frames using OpenCV or ffmpeg
        if OPENCV_AVAILABLE:
            return self._extract_frames_opencv(
                video_path, output_dir, timestamps, metadata, image_format, quality
            )
        else:
            return self._extract_frames_ffmpeg(
                video_path, output_dir, timestamps, metadata, image_format, quality
            )

    def _calculate_fixed_interval_timestamps(
        self,
        duration: float,
        num_frames: int,
    ) -> List[float]:
        """Calculate evenly spaced timestamps"""
        if num_frames <= 1:
            return [duration / 2]

        # Avoid first/last 5% of video (often intros/credits)
        start = duration * 0.05
        end = duration * 0.95

        interval = (end - start) / (num_frames - 1)
        return [start + i * interval for i in range(num_frames)]

    def _extract_frames_opencv(
        self,
        video_path: Path,
        output_dir: str,
        timestamps: List[float],
        metadata: VideoMetadata,
        image_format: str,
        quality: int,
    ) -> List[ExtractedFrame]:
        """Extract frames using OpenCV"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        try:
            if not cap.isOpened():
                raise RuntimeError("Failed to open video with OpenCV")

            for i, timestamp in enumerate(timestamps):
                # Seek to timestamp
                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame at {timestamp:.2f}s")
                    continue

                # Save frame
                frame_filename = f"frame_{i:04d}_{timestamp:.2f}s.{image_format}"
                frame_path = os.path.join(output_dir, frame_filename)

                if image_format == "jpg":
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                else:
                    cv2.imwrite(frame_path, frame)

                frame_number = int(timestamp * metadata.fps)

                frames.append(
                    ExtractedFrame(
                        frame_path=frame_path,
                        timestamp=timestamp,
                        frame_number=frame_number,
                        metadata={
                            "source_video": str(video_path),
                            "resolution": f"{metadata.width}x{metadata.height}",
                        },
                    )
                )

                logger.debug(
                    f"Extracted frame {i+1}/{len(timestamps)} at {timestamp:.2f}s"
                )

            logger.info(f"Successfully extracted {len(frames)} frames")
            return frames

        finally:
            cap.release()

    def _extract_frames_ffmpeg(
        self,
        video_path: Path,
        output_dir: str,
        timestamps: List[float],
        metadata: VideoMetadata,
        image_format: str,
        quality: int,
    ) -> List[ExtractedFrame]:
        """Extract frames using ffmpeg"""
        ffmpeg = self._ensure_ffmpeg()
        frames = []

        for i, timestamp in enumerate(timestamps):
            frame_filename = f"frame_{i:04d}_{timestamp:.2f}s.{image_format}"
            frame_path = os.path.join(output_dir, frame_filename)

            cmd = [
                ffmpeg,
                "-ss",
                str(timestamp),
                "-i",
                str(video_path),
                "-vframes",
                "1",
                "-q:v",
                str(
                    int((100 - quality) / 100 * 31 + 1)
                ),  # Convert quality to ffmpeg scale
                "-y",  # Overwrite
                frame_path,
            ]

            try:
                subprocess.run(cmd, capture_output=True, timeout=30, check=True)

                if os.path.exists(frame_path):
                    frame_number = int(timestamp * metadata.fps)

                    frames.append(
                        ExtractedFrame(
                            frame_path=frame_path,
                            timestamp=timestamp,
                            frame_number=frame_number,
                            metadata={
                                "source_video": str(video_path),
                                "resolution": f"{metadata.width}x{metadata.height}",
                            },
                        )
                    )

                    logger.debug(
                        f"Extracted frame {i+1}/{len(timestamps)} at {timestamp:.2f}s"
                    )
                else:
                    logger.warning(f"Failed to extract frame at {timestamp:.2f}s")

            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout extracting frame at {timestamp:.2f}s")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Error extracting frame at {timestamp:.2f}s: {e}")

        logger.info(f"Successfully extracted {len(frames)} frames")
        return frames

    def extract_audio(
        self,
        video_path: Union[str, Path],
        output_path: Optional[str] = None,
        audio_format: str = "wav",
        sample_rate: int = 16000,
        mono: bool = True,
    ) -> Optional[str]:
        """
        Extract audio track from video

        Args:
            video_path: Path to video file
            output_path: Path for output audio file (auto-generate if None)
            audio_format: Output audio format ('wav', 'mp3', 'flac')
            sample_rate: Audio sample rate in Hz
            mono: Convert to mono if True

        Returns:
            Path to extracted audio file, or None if no audio track
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        ffmpeg = self._ensure_ffmpeg()

        # Generate output path if not provided
        if output_path is None:
            output_path = os.path.join(
                self.temp_dir, f"{video_path.stem}_audio.{audio_format}"
            )

        # Build ffmpeg command
        cmd = [
            ffmpeg,
            "-i",
            str(video_path),
            "-vn",  # No video
            "-acodec",
            (
                "pcm_s16le"
                if audio_format == "wav"
                else "libmp3lame" if audio_format == "mp3" else "flac"
            ),
            "-ar",
            str(sample_rate),
        ]

        if mono:
            cmd.extend(["-ac", "1"])

        cmd.extend(["-y", output_path])  # Overwrite

        logger.info(f"Extracting audio from {video_path.name}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 min timeout
            )

            if result.returncode != 0:
                if (
                    "Stream map" in result.stderr
                    and "matches no streams" in result.stderr
                ):
                    logger.warning(f"No audio track found in {video_path.name}")
                    return None
                logger.error(f"ffmpeg error: {result.stderr[:500]}")
                return None

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Audio extracted: {output_path}")
                return output_path
            else:
                logger.warning("Audio extraction produced empty file")
                return None

        except subprocess.TimeoutExpired:
            logger.error("Audio extraction timed out")
            return None

    def process_video(
        self,
        video_path: Union[str, Path],
        output_dir: Optional[str] = None,
        num_frames: int = 10,
        extract_audio: bool = True,
        frame_strategy: str = "fixed_interval",
        audio_format: str = "wav",
    ) -> ExtractionResult:
        """
        Process video: extract frames and audio

        Args:
            video_path: Path to video file
            output_dir: Directory for output files
            num_frames: Number of frames to extract
            extract_audio: Whether to extract audio track
            frame_strategy: Frame extraction strategy
            audio_format: Audio output format

        Returns:
            ExtractionResult with frames, audio path, and metadata
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Create output directory
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="video_extract_", dir=self.temp_dir)
        else:
            os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Processing video: {video_path.name}")

        # Get metadata
        metadata = self.get_video_metadata(video_path)

        # Extract frames
        frames = self.extract_frames(
            video_path=video_path,
            output_dir=output_dir,
            num_frames=num_frames,
            strategy=frame_strategy,
        )

        # Extract audio
        audio_path = None
        if extract_audio and metadata.has_audio:
            audio_output = os.path.join(output_dir, f"audio.{audio_format}")
            audio_path = self.extract_audio(
                video_path=video_path,
                output_path=audio_output,
                audio_format=audio_format,
            )

        result = ExtractionResult(
            frames=frames,
            audio_path=audio_path,
            video_metadata=metadata,
            output_dir=output_dir,
        )

        logger.info(
            f"Video processing complete: {len(frames)} frames, "
            f"audio={'extracted' if audio_path else 'none'}"
        )

        return result

    def cleanup_extraction(self, result: ExtractionResult) -> None:
        """
        Clean up extracted files

        Args:
            result: ExtractionResult to clean up
        """
        if os.path.exists(result.output_dir):
            shutil.rmtree(result.output_dir, ignore_errors=True)
            logger.debug(f"Cleaned up extraction directory: {result.output_dir}")
