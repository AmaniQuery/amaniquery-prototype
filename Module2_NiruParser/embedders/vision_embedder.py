"""
Vision Embedder using Cohere Embed-4 for multimodal embeddings
"""

import os
import time
import functools
from typing import List, Union, Optional
import numpy as np
from pathlib import Path
from loguru import logger
from PIL import Image
import base64
import io

try:
    import cohere

    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    logger.warning("Cohere package not available. Install with: pip install cohere")


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """Simple retry decorator with exponential backoff"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < max_attempts:
                        wait = delay * (backoff ** (attempt - 1))
                        logger.warning(
                            f"Retry {attempt}/{max_attempts} for {func.__name__}: {e}. Waiting {wait:.1f}s"
                        )
                        time.sleep(wait)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )
                        raise
            raise last_exc

        return wrapper

    return decorator


class VisionEmbedder:
    """Generate multimodal embeddings using Cohere Embed-4 API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "embed-english-v3.0",
    ):
        if not COHERE_AVAILABLE:
            raise ImportError(
                "Cohere package not available. Install with: pip install cohere"
            )

        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY not set in environment or provided")

        self.model = model
        self.client = cohere.Client(api_key=self.api_key)
        self.dimension = 1024

        logger.info(f"Vision embedder initialized with model: {model}")

    def _image_to_base64(self, image: Union[str, Path, Image.Image]) -> str:
        """
        Convert image to base64 string for API

        Args:
            image: Image path (str/Path) or PIL Image object

        Returns:
            Base64 encoded image string
        """
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            img = Image.open(image_path)
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return img_base64

    @retry(max_attempts=3, delay=1.0)
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text query

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        try:
            response = self.client.embed(
                texts=[text],
                model=self.model,
                input_type="search_query",
            )

            embedding = np.array(response.embeddings[0])
            return embedding
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            return np.zeros(self.dimension)

    @retry(max_attempts=3, delay=1.0)
    def embed_image(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            with open(image_path, "rb") as f:
                image_bytes = f.read()
        elif isinstance(image, Image.Image):
            buffered = io.BytesIO()
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(buffered, format="JPEG", quality=85)
            image_bytes = buffered.getvalue()
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        mime_type = "image/jpeg"
        if isinstance(image, (str, Path)):
            ext = Path(image).suffix.lower()
            if ext == ".png":
                mime_type = "image/png"
        elif isinstance(image, Image.Image):
            if image.format == "PNG":
                mime_type = "image/png"

        img_base64 = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{img_base64}"

        try:
            response = self.client.embed(
                images=[data_uri],
                model=self.model,
                input_type="image",
            )
            return np.array(response.embeddings[0])
        except Exception as api_error:
            error_str = str(api_error)
            if "not found" in error_str.lower() or "404" in error_str:
                logger.warning(
                    f"Model {self.model} not found, trying embed-multilingual-v3.0"
                )
                try:
                    response = self.client.embed(
                        images=[data_uri],
                        model="embed-multilingual-v3.0",
                        input_type="image",
                    )
                    embedding = np.array(response.embeddings[0])
                    self.model = "embed-multilingual-v3.0"
                    return embedding
                except Exception as fallback_error:
                    logger.error(f"Fallback model also failed: {fallback_error}")
                    raise ValueError(
                        f"Cohere embedding models not available: {api_error}"
                    )
            else:
                logger.error(f"Cohere embed API error: {api_error}")
                raise ValueError(f"Failed to generate image embedding: {api_error}")

    @retry(max_attempts=3, delay=1.0)
    def embed_images_batch(
        self, images: List[Union[str, Path, Image.Image]]
    ) -> np.ndarray:
        if not images:
            return np.array([])

        try:
            data_uri_list = []
            for img in images:
                if isinstance(img, (str, Path)):
                    image_path = Path(img)
                    with open(image_path, "rb") as f:
                        image_bytes = f.read()
                    ext = image_path.suffix.lower()
                    mime_type = "image/png" if ext == ".png" else "image/jpeg"
                elif isinstance(img, Image.Image):
                    buffered = io.BytesIO()
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img.save(buffered, format="JPEG", quality=85)
                    image_bytes = buffered.getvalue()
                    mime_type = "image/png" if img.format == "PNG" else "image/jpeg"
                else:
                    raise ValueError(f"Unsupported image type: {type(img)}")

                img_base64 = base64.b64encode(image_bytes).decode("utf-8")
                data_uri = f"data:{mime_type};base64,{img_base64}"
                data_uri_list.append(data_uri)

            response = self.client.embed(
                images=data_uri_list,
                model=self.model,
                input_type="image",
            )

            return np.array(response.embeddings)
        except Exception as e:
            logger.error(f"Error generating batch image embeddings: {e}")
            return np.zeros((len(images), self.dimension))

    @retry(max_attempts=3, delay=1.0)
    def embed_text_batch(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])

        try:
            response = self.client.embed(
                texts=texts,
                model=self.model,
                input_type="search_query",
            )
            return np.array(response.embeddings)
        except Exception as e:
            logger.error(f"Error generating batch text embeddings: {e}")
            return np.zeros((len(texts), self.dimension))

    def get_model_info(self) -> dict:
        """Get information about the embedding model"""
        return {
            "model_name": self.model,
            "dimension": self.dimension,
            "provider": "cohere",
            "multimodal": True,
        }
