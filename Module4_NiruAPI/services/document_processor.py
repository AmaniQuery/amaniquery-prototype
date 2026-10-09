"""
Document Processing Service for Chat Attachments
Handles PDF, image, and text file processing for chat attachments
"""

import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import tempfile
from loguru import logger

try:
    import PyPDF2

    PDF_AVAILABLE = True
except ImportError:
    try:
        import pdfplumber

        PDF_AVAILABLE = True
        PDF_LIB = "pdfplumber"
    except ImportError:
        PDF_AVAILABLE = False
        PDF_LIB = None

try:
    from PIL import Image
    import pytesseract

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

from Module2_NiruParser.chunkers import TextChunker
from Module2_NiruParser.embedders import TextEmbedder, VisionEmbedder
from Module2_NiruParser.config import Config
from Module4_NiruAPI.services.pdf_page_extractor import PDFPageExtractor

try:
    from Module4_NiruAPI.services.cloudinary_service import CloudinaryService

    CLOUDINARY_AVAILABLE = True
except (ImportError, ValueError) as e:
    CLOUDINARY_AVAILABLE = False
    CloudinaryService = None
    logger.warning(f"Cloudinary service not available: {e}")


class DocumentProcessor:
    """Process uploaded documents for chat attachments"""

    def __init__(self, config: Optional[Config] = None, enable_vision: bool = True):
        self.config = config or Config()
        self.chunker = TextChunker(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
        )
        self.embedder = TextEmbedder(
            model_name=self.config.EMBEDDING_MODEL,
            batch_size=self.config.EMBEDDING_BATCH_SIZE,
        )
        # Use a persistent directory for uploads
        self.upload_dir = Path("amaniquery_uploads")
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Cloudinary service if available
        self.cloudinary_service = None
        if CLOUDINARY_AVAILABLE:
            try:
                self.cloudinary_service = CloudinaryService()
                logger.info("Cloudinary service initialized")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize Cloudinary service: {e}. Files will not be uploaded to Cloudinary."
                )
                self.cloudinary_service = None

        # Initialize vision components if enabled
        self.enable_vision = enable_vision
        self.vision_embedder = None
        self.pdf_extractor = None

        if self.enable_vision:
            try:
                self.vision_embedder = VisionEmbedder()
                self.pdf_extractor = PDFPageExtractor()
                logger.info("Vision RAG components initialized")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize vision components: {e}. Vision RAG will be disabled."
                )
                self.enable_vision = False

    def process_file(self, file_content: bytes, filename: str, session_id: str) -> Dict:
        """
        Process an uploaded file and return chunks with embeddings

        Args:
            file_content: File content as bytes
            filename: Original filename
            session_id: Chat session ID

        Returns:
            Dictionary with attachment metadata and processed chunks
        """
        try:
            # Determine file type
            file_type = self._get_file_type(filename)

            # Save file temporarily
            file_id = str(uuid.uuid4())
            file_path = self.upload_dir / f"{session_id}_{file_id}_{filename}"
            file_path.write_bytes(file_content)

            # Extract text based on file type
            # For images with Vision RAG enabled, OCR is optional
            text = ""
            chunks = []

            if file_type == "pdf":
                text = self._extract_pdf_text(file_path)
            elif file_type == "image":
                # For images, try OCR if available, but don't fail if Vision RAG is enabled
                if OCR_AVAILABLE:
                    try:
                        text = self._extract_image_text(file_path)
                    except Exception as e:
                        logger.warning(f"OCR extraction failed for image: {e}")
                        text = ""
                elif self.enable_vision:
                    # Vision RAG doesn't need OCR - skip text extraction
                    logger.info(
                        f"Skipping OCR for image {filename} - using Vision RAG instead"
                    )
                    text = ""
                else:
                    raise ValueError(
                        "OCR not available and Vision RAG is disabled. Install pytesseract and PIL for image text extraction."
                    )
            elif file_type == "text":
                text = self._extract_text_file(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            # Process text chunks only if text was extracted
            if text and len(text.strip()) >= 10:
                # Create chunks
                chunks = self.chunker.chunk_text(text)

                # Generate embeddings
                embeddings = self.embedder.embed_batch(
                    [chunk["text"] for chunk in chunks]
                )

                # Add embeddings to chunks
                for i, chunk in enumerate(chunks):
                    chunk["embedding"] = embeddings[i]
                    chunk["session_id"] = session_id
                    chunk["attachment_id"] = file_id
                    chunk["source"] = f"attachment:{filename}"
            else:
                # No text extracted - this is OK for Vision RAG
                if not self.enable_vision and file_type == "image":
                    raise ValueError(
                        "No text extracted from image and Vision RAG is disabled"
                    )
                logger.info(
                    f"No text extracted from {filename}, processing with Vision RAG only"
                )

            # Create attachment metadata
            attachment_metadata = {
                "id": file_id,
                "filename": filename,
                "file_type": file_type,
                "file_size": len(file_content),
                "uploaded_at": datetime.utcnow().isoformat(),
                "processed": True,
                "chunk_count": len(chunks),
                "text_length": len(text) if text else 0,
            }

            result = {
                "attachment": attachment_metadata,
                "chunks": chunks,
            }

            # Generate vision embeddings for images and PDFs
            if self.enable_vision and file_type in ["image", "pdf"]:
                try:
                    # Don't delete file yet - vision processing needs it
                    vision_data = self._process_vision_embeddings(
                        file_path=file_path,
                        file_type=file_type,
                        file_id=file_id,
                        filename=filename,
                        session_id=session_id,
                    )
                    result["vision_data"] = vision_data
                except Exception as e:
                    # Don't fail the entire file processing if vision RAG fails
                    logger.warning(
                        f"Failed to process vision embeddings: {e}. File will still be processed for text."
                    )
                    result["vision_data"] = None
                    # If no text was extracted and vision failed, that's OK - at least we tried

            # Upload to Cloudinary if service is available
            cloudinary_url = None
            if self.cloudinary_service:
                try:
                    cloudinary_result = self.cloudinary_service.upload_file(
                        file_path=file_path, filename=filename, session_id=session_id
                    )
                    cloudinary_url = cloudinary_result.get(
                        "secure_url"
                    ) or cloudinary_result.get("url")
                    logger.info(f"Uploaded {filename} to Cloudinary: {cloudinary_url}")
                except Exception as e:
                    # Don't fail the entire file processing if Cloudinary upload fails
                    logger.warning(
                        f"Failed to upload {filename} to Cloudinary: {e}. File processing will continue."
                    )
                    cloudinary_url = None

            # Add Cloudinary URL to attachment metadata
            if cloudinary_url:
                attachment_metadata["cloudinary_url"] = cloudinary_url

            # Update result with modified attachment metadata
            result["attachment"] = attachment_metadata

            # Keep file for download
            # try:
            #     file_path.unlink()
            # except Exception as e:
            #     logger.warning(f"Failed to delete temp file: {e}")

            return result

        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            raise

    def _get_file_type(self, filename: str) -> str:
        """Determine file type from filename"""
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            return "pdf"
        elif ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]:
            return "image"
        elif ext in [".txt", ".md", ".text"]:
            return "text"
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        if not PDF_AVAILABLE:
            raise ImportError(
                "PDF processing libraries not available. Install PyPDF2 or pdfplumber."
            )

        text_parts = []

        try:
            if PDF_LIB == "pdfplumber":
                import pdfplumber

                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)
            else:
                # Use PyPDF2
                with open(file_path, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)

            return "\n\n".join(text_parts)
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise

    def _extract_image_text(self, file_path: Path) -> str:
        """Extract text from image using OCR"""
        if not OCR_AVAILABLE:
            raise ImportError("OCR not available. Install pytesseract and PIL.")

        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            logger.error(f"Error extracting image text: {e}")
            raise

    def _extract_text_file(self, file_path: Path) -> str:
        """Extract text from text file"""
        try:
            # Try UTF-8 first
            try:
                return file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # Fallback to latin-1
                return file_path.read_text(encoding="latin-1")
        except Exception as e:
            logger.error(f"Error reading text file: {e}")
            raise

    def store_chunks_in_vector_store(
        self, chunks: List[Dict], vector_store, collection_name: str
    ):
        """
        Store processed chunks in vector store with session-specific collection

        Args:
            chunks: List of chunk dictionaries with embeddings
            vector_store: VectorStore instance
            collection_name: Collection name (typically session_id)
        """
        try:
            # Create or get collection
            if hasattr(vector_store, "get_or_create_collection"):
                collection = vector_store.get_or_create_collection(collection_name)
            else:
                # Fallback: use default collection with metadata filter
                collection = None

            # Add chunks to vector store
            texts = [chunk["text"] for chunk in chunks]
            embeddings = [chunk["embedding"] for chunk in chunks]
            metadatas = [
                {
                    "session_id": chunk.get("session_id"),
                    "attachment_id": chunk.get("attachment_id"),
                    "source": chunk.get("source"),
                    "chunk_index": chunk.get("chunk_index", 0),
                }
                for chunk in chunks
            ]
            ids = [
                f"{chunk.get('attachment_id')}_{i}" for i, chunk in enumerate(chunks)
            ]

            if collection:
                collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids,
                )
            else:
                # Use vector_store.add method
                vector_store.add(
                    texts=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids,
                )

            logger.info(f"Stored {len(chunks)} chunks in collection {collection_name}")

        except Exception as e:
            logger.error(f"Error storing chunks in vector store: {e}")
            raise

    def cleanup_session_chunks(self, vector_store, session_id: str):
        """
        Remove all chunks for a session from vector store

        Args:
            vector_store: VectorStore instance
            session_id: Chat session ID
        """
        try:
            collection_name = f"chat_session_{session_id}"
            logger.info(
                f"Cleaning up chunks for session {session_id} in collection {collection_name}"
            )

            # Get the backend type
            backend = getattr(vector_store, "backend", None)

            if backend == "chromadb":
                # ChromaDB: Delete collection or delete by metadata filter
                try:
                    # Try to get the collection
                    if hasattr(vector_store, "client"):
                        try:
                            collection = vector_store.client.get_collection(
                                collection_name
                            )
                            # Delete all items in the collection
                            all_ids = collection.get()["ids"]
                            if all_ids:
                                collection.delete(ids=all_ids)
                                logger.info(
                                    f"Deleted {len(all_ids)} chunks from ChromaDB collection {collection_name}"
                                )
                            # Optionally delete the entire collection
                            vector_store.client.delete_collection(collection_name)
                            logger.info(
                                f"Deleted ChromaDB collection {collection_name}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Collection {collection_name} may not exist: {e}"
                            )
                except Exception as e:
                    logger.error(f"Error deleting ChromaDB collection: {e}")

            elif backend == "qdrant":
                # QDrant: Delete points by filter
                try:
                    from qdrant_client.http import models

                    # Delete all points in the collection that match session_id
                    vector_store.client.delete(
                        collection_name=collection_name,
                        points_selector=models.FilterSelector(
                            filter=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="session_id",
                                        match=models.MatchValue(value=session_id),
                                    )
                                ]
                            )
                        ),
                    )
                    logger.info(
                        f"Deleted chunks from QDrant collection {collection_name}"
                    )
                except Exception as e:
                    # If collection doesn't exist, that's okay
                    if (
                        "not found" in str(e).lower()
                        or "does not exist" in str(e).lower()
                    ):
                        logger.info(
                            f"Collection {collection_name} does not exist, nothing to clean"
                        )
                    else:
                        logger.error(f"Error deleting QDrant collection: {e}")

            elif backend == "upstash":
                # Upstash: Delete by IDs (we'd need to query first to get IDs)
                try:
                    # Query to get all IDs for this session
                    # Note: Upstash doesn't have a direct delete by filter, so we'd need to track IDs
                    logger.warning(
                        "Upstash cleanup requires tracking chunk IDs - not fully implemented"
                    )
                    # For now, log that cleanup is needed
                    logger.info(
                        f"Upstash cleanup requested for session {session_id} - manual cleanup may be needed"
                    )
                except Exception as e:
                    logger.error(f"Error cleaning Upstash collection: {e}")
            else:
                # Try to use get_collection method if available
                try:
                    if hasattr(vector_store, "get_collection"):
                        collection = vector_store.get_collection(collection_name)
                        if hasattr(collection, "delete"):
                            # Try to get all IDs and delete them
                            if hasattr(collection, "get"):
                                all_data = collection.get()
                                if all_data and "ids" in all_data:
                                    collection.delete(ids=all_data["ids"])
                                    logger.info(
                                        f"Deleted {len(all_data['ids'])} chunks from collection {collection_name}"
                                    )
                        elif hasattr(collection, "delete_collection"):
                            collection.delete_collection()
                            logger.info(f"Deleted collection {collection_name}")
                except Exception as e:
                    logger.warning(f"Could not clean collection {collection_name}: {e}")

            logger.info(f"Completed cleanup for session {session_id}")

        except Exception as e:
            logger.error(f"Error cleaning up session chunks: {e}")
            # Don't raise - cleanup is best effort

    def _process_vision_embeddings(
        self,
        file_path: Path,
        file_type: str,
        file_id: str,
        filename: str,
        session_id: str,
    ) -> Dict:
        """
        Process vision embeddings for images and PDF pages

        Args:
            file_path: Path to the file
            file_type: Type of file ("image" or "pdf")
            file_id: Unique file ID
            filename: Original filename
            session_id: Chat session ID

        Returns:
            Dictionary with vision data (images, embeddings, metadata)
        """
        if not self.enable_vision or not self.vision_embedder:
            return None

        vision_images = []

        try:
            if file_type == "image":
                # Process single image
                embedding = self.vision_embedder.embed_image(file_path)

                # Save image to persistent location for Vision RAG
                vision_dir = self.upload_dir / "vision" / session_id
                vision_dir.mkdir(parents=True, exist_ok=True)
                vision_image_path = vision_dir / f"{file_id}_{filename}"

                # Copy image to vision directory
                import shutil

                shutil.copy2(file_path, vision_image_path)

                vision_images.append(
                    {
                        "id": f"{file_id}_image",
                        "file_path": str(vision_image_path),
                        "embedding": embedding.tolist(),  # Convert to list for JSON
                        "metadata": {
                            "filename": filename,
                            "file_id": file_id,
                            "session_id": session_id,
                            "type": "image",
                            "page_number": None,
                            "source_file": filename,
                        },
                    }
                )

            elif file_type == "pdf":
                # Extract PDF pages as images
                page_images = self.pdf_extractor.extract_pages_from_file(file_path)

                # Save pages to persistent location
                vision_dir = self.upload_dir / "vision" / session_id
                vision_dir.mkdir(parents=True, exist_ok=True)

                # Generate embeddings for each page
                for page_num, page_image in enumerate(page_images, start=1):
                    # Save page image
                    page_filename = f"{file_id}_page_{page_num:04d}.png"
                    page_path = vision_dir / page_filename
                    page_image.save(page_path, format="PNG")

                    # Generate embedding
                    embedding = self.vision_embedder.embed_image(page_image)

                    vision_images.append(
                        {
                            "id": f"{file_id}_page_{page_num}",
                            "file_path": str(page_path),
                            "embedding": embedding.tolist(),
                            "metadata": {
                                "filename": page_filename,
                                "file_id": file_id,
                                "session_id": session_id,
                                "type": "pdf_page",
                                "page_number": page_num,
                                "source_file": filename,
                            },
                        }
                    )

            logger.info(f"Processed {len(vision_images)} vision item(s) for {filename}")
            return {
                "images": vision_images,
                "count": len(vision_images),
            }

        except Exception as e:
            logger.error(f"Error processing vision embeddings: {e}")
            return None
