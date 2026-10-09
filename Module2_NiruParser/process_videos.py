"""
Process YouTube Videos - Extract transcripts and create embeddings
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from Module2_NiruParser.pipeline import ProcessingPipeline
from Module2_NiruParser.extractors.transcript_extractor import TranscriptExtractor
from Module3_NiruDB.vector_store import VectorStore
import json


def process_video_file(video_file_path: str):
    """
    Process videos from JSON file

    Args:
        video_file_path: Path to JSON file with video metadata
    """
    logger.info(f"Processing videos from {video_file_path}")

    # Load video data
    with open(video_file_path, "r", encoding="utf-8") as f:
        videos = json.load(f)

    if not videos:
        logger.warning("No videos found in file")
        return

    logger.info(f"Found {len(videos)} videos to process")

    # Initialize components
    transcript_extractor = TranscriptExtractor()
    vector_store = VectorStore()

    total_chunks = 0
    successful = 0
    failed = 0

    for i, video in enumerate(videos, 1):
        try:
            logger.info(
                f"[{i}/{len(videos)}] Processing video: {video.get('title', video.get('video_id'))}"
            )

            # Extract video ID
            video_id = video.get("video_id")
            if not video_id:
                logger.warning("No video ID found")
                failed += 1
                continue

            # Extract transcript
            transcript_data = transcript_extractor.extract_transcript(video_id)

            if not transcript_data:
                logger.warning(f"No transcript for video {video_id}")
                failed += 1
                continue

            # Chunk transcript by time
            time_chunks = transcript_extractor.chunk_transcript(
                transcript_data, chunk_duration=60, overlap_duration=10
            )

            if not time_chunks:
                logger.warning(f"No chunks created for video {video_id}")
                failed += 1
                continue

            # Create document chunks with metadata
            chunks = []
            for j, time_chunk in enumerate(time_chunks):
                chunk = {
                    "id": f"{video_id}_{j}",
                    "text": time_chunk["text"],
                    "metadata": {
                        "video_id": video_id,
                        "video_url": video.get(
                            "video_url", f"https://www.youtube.com/watch?v={video_id}"
                        ),
                        "title": video.get("title", "Unknown"),
                        "category": video.get("category", "Parliamentary Record"),
                        "source_type": "YouTube Video",
                        "source_name": video.get("source_name", "Parliament YouTube"),
                        "start_time_seconds": time_chunk["start_time"],
                        "end_time_seconds": time_chunk["end_time"],
                        "duration_seconds": time_chunk["duration"],
                        "timestamp_url": time_chunk["timestamp_url"],
                        "timestamp_formatted": time_chunk["timestamp_formatted"],
                        "transcript_language": transcript_data.get("language", "en"),
                        "is_generated": transcript_data.get("is_generated", True),
                        "upload_date": video.get("upload_date"),
                        "scraped_at": video.get("scraped_at"),
                    },
                }
                chunks.append(chunk)

            # Add to vector store
            vector_store.add_documents(chunks)

            total_chunks += len(chunks)
            successful += 1
            logger.info(f"✓ Added {len(chunks)} timestamped chunks to vector store")

        except Exception as e:
            logger.error(f"Error processing video {video.get('video_id')}: {e}")
            failed += 1
            continue

    logger.info("=" * 70)
    logger.info("VIDEO PROCESSING COMPLETE")
    logger.info(f"Successful: {successful}/{len(videos)}")
    logger.info(f"Failed: {failed}/{len(videos)}")
    logger.info(f"Total chunks created: {total_chunks}")
    logger.info("=" * 70)


def process_single_video(video_url: str, title: str = None):
    """
    Process a single YouTube video

    Args:
        video_url: YouTube video URL
        title: Optional video title
    """
    logger.info(f"Processing single video: {video_url}")

    # Extract video ID
    if "v=" in video_url:
        video_id = video_url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in video_url:
        video_id = video_url.split("youtu.be/")[1].split("?")[0]
    else:
        logger.error("Invalid YouTube URL")
        return

    # Create video document
    video = {
        "video_id": video_id,
        "video_url": video_url,
        "title": title or f"YouTube Video {video_id}",
        "category": "Parliamentary Record",
        "source_type": "YouTube Video",
        "source_name": "Parliament YouTube",
    }

    # Initialize components
    transcript_extractor = TranscriptExtractor()
    vector_store = VectorStore()

    try:
        # Extract transcript
        transcript_data = transcript_extractor.extract_transcript(video_id)

        if not transcript_data:
            logger.warning("No transcript available")
            return

        # Chunk transcript
        time_chunks = transcript_extractor.chunk_transcript(
            transcript_data, chunk_duration=60, overlap_duration=10
        )

        if not time_chunks:
            logger.warning("No chunks created")
            return

        # Create document chunks
        chunks = []
        for i, time_chunk in enumerate(time_chunks):
            chunk = {
                "id": f"{video_id}_{i}",
                "text": time_chunk["text"],
                "metadata": {
                    "video_id": video_id,
                    "video_url": video_url,
                    "title": title or f"YouTube Video {video_id}",
                    "category": "Parliamentary Record",
                    "source_type": "YouTube Video",
                    "source_name": "Parliament YouTube",
                    "start_time_seconds": time_chunk["start_time"],
                    "end_time_seconds": time_chunk["end_time"],
                    "duration_seconds": time_chunk["duration"],
                    "timestamp_url": time_chunk["timestamp_url"],
                    "timestamp_formatted": time_chunk["timestamp_formatted"],
                    "transcript_language": transcript_data.get("language", "en"),
                    "is_generated": transcript_data.get("is_generated", True),
                },
            }
            chunks.append(chunk)

        # Add to vector store
        vector_store.add_documents(chunks)

        logger.info(f"✓ Successfully processed video: {len(chunks)} chunks added")

    except Exception as e:
        logger.error(f"Error processing video: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process YouTube videos")
    parser.add_argument("--file", help="JSON file with video metadata")
    parser.add_argument("--url", help="Single video URL to process")
    parser.add_argument("--title", help="Title for single video")

    args = parser.parse_args()

    if args.file:
        process_video_file(args.file)
    elif args.url:
        process_single_video(args.url, args.title)
    else:
        print("Usage:")
        print("  Process from file: python process_videos.py --file videos.json")
        print(
            "  Process single: python process_videos.py --url 'https://youtube.com/watch?v=...' --title 'Video Title'"
        )
