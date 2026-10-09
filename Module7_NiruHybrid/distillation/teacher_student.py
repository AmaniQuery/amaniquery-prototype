import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Union, List, Tuple
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class ModelWrapper(ABC):
    """
    Abstract base class for models in the distillation setup.
    Wraps different model types (HF, Custom, etc.) to provide a uniform interface.
    """

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Run inference on input data."""
        pass

    @abstractmethod
    def get_embeddings(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Get vector embeddings for text."""
        pass

    @abstractmethod
    def get_score(self, query: str, doc: str) -> float:
        """Get relevance score for a query-document pair."""
        pass

    @abstractmethod
    def train_mode(self):
        """Set model to training mode."""
        pass

    @abstractmethod
    def eval_mode(self):
        """Set model to evaluation mode."""
        pass


class BiEncoderWrapper(ModelWrapper):
    """
    Wrapper for Bi-Encoders (like HybridEncoder or SentenceTransformers).
    Produces embeddings for query and doc separately.
    """

    def __init__(self, model: nn.Module, device: str = "cpu", tokenizer: Any = None):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.model.to(device)

    def predict(self, input_data: Any) -> Any:
        return self.get_embeddings(input_data)

    def get_embeddings(self, text: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(text, str):
            text = [text]

        # Check if model has 'encode' method (SentenceTransformer/HybridEncoder style)
        if hasattr(self.model, "encode"):
            # Handle HybridEncoder specific signature if needed
            try:
                embeddings = self.model.encode(text)
            except TypeError:
                # Fallback for models that might expect different args
                embeddings = self.model.encode(text, return_pooled=True)

            if isinstance(embeddings, list):
                embeddings = torch.tensor(embeddings)
            return embeddings.to(self.device)

        # Fallback to forward pass if no encode method (Raw PyTorch)
        if self.tokenizer:
            inputs = self.tokenizer(
                text, padding=True, truncation=True, return_tensors="pt", max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.set_grad_enabled(self.model.training):
                outputs = self.model(**inputs)

            # Mean Pooling
            if hasattr(outputs, "last_hidden_state"):
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = (
                    inputs["attention_mask"]
                    .unsqueeze(-1)
                    .expand(token_embeddings.size())
                    .float()
                )
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                return sum_embeddings / sum_mask
            elif hasattr(outputs, "pooler_output"):
                return outputs.pooler_output
            else:
                # Assume output is already embeddings
                return outputs

        raise NotImplementedError(
            "Model must implement 'encode' method or provide a tokenizer"
        )

    def get_score(self, query: str, doc: str) -> float:
        """
        Compute cosine similarity between query and doc embeddings.
        """
        q_emb = self.get_embeddings(query)
        d_emb = self.get_embeddings(doc)
        return F.cosine_similarity(q_emb, d_emb).item()

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()


class CrossEncoderWrapper(ModelWrapper):
    """
    Wrapper for Cross-Encoders (e.g., BERT for sequence classification).
    Scores (query, doc) pairs directly.
    """

    def __init__(self, model: Any, device: str = "cpu", tokenizer: Any = None):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        if isinstance(model, nn.Module):
            self.model.to(device)

    def predict(self, input_data: Any) -> Any:
        # input_data expected to be list of (query, doc) pairs
        # Expects model to have a predict method taking list of pairs
        # e.g. sentence_transformers.CrossEncoder
        if hasattr(self.model, "predict"):
            return self.model.predict(input_data)

        # Fallback for raw HF model
        if self.tokenizer and isinstance(self.model, nn.Module):
            queries, docs = zip(*input_data)
            inputs = self.tokenizer(
                list(queries),
                list(docs),
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                    # If binary classification, take positive class score
                    if logits.shape[1] == 2:
                        return F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    # If regression, take scalar
                    return logits.view(-1).cpu().numpy()

        raise NotImplementedError(
            "Model must implement 'predict' method for pairs or provide a tokenizer"
        )

    def get_embeddings(self, text: Union[str, List[str]]) -> torch.Tensor:
        raise NotImplementedError(
            "Cross-Encoders do not produce independent embeddings."
        )

    def get_score(self, query: str, doc: str) -> float:
        return float(self.predict([(query, doc)])[0])

    def train_mode(self):
        if isinstance(self.model, nn.Module):
            self.model.train()

    def eval_mode(self):
        if isinstance(self.model, nn.Module):
            self.model.eval()


class TeacherStudentPair:
    """
    Manages a pair of models: a large 'Teacher' and a small 'Student'.
    Supports knowledge distillation training steps.
    """

    def __init__(
        self,
        teacher_model: ModelWrapper,
        student_model: ModelWrapper,
        temperature: float = 2.0,
        alpha: float = 0.5,
        distillation_loss: str = "mse",
        device: str = "cpu",
    ):
        """
        Initialize the pair.

        Args:
            teacher_model: The large, accurate model (Teacher).
            student_model: The small, fast model (Student).
            temperature: Softmax temperature for distillation loss.
            alpha: Weight for the distillation loss (vs student's original loss).
            distillation_loss: 'mse' or 'kldiv'.
            device: 'cpu' or 'cuda'.
        """
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.distillation_loss_type = distillation_loss
        self.device = device

        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")
        self.mse_loss = nn.MSELoss()

    def distill_step(
        self,
        batch_queries: List[str],
        batch_docs: List[str],
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Perform a single training step of knowledge distillation.

        Scenario: Student (Bi-Encoder) learns from Teacher (Cross-Encoder or Bi-Encoder).

        Args:
            batch_queries: List of query texts.
            batch_docs: List of document texts.
            labels: Ground truth relevance scores (if available).
            optimizer: Optimizer for the student model.

        Returns:
            Dictionary of loss values.
        """
        self.student.train_mode()
        self.teacher.eval_mode()

        optimizer.zero_grad()

        # 1. Get Teacher Logits/Scores (Target)
        with torch.no_grad():
            if isinstance(self.teacher, CrossEncoderWrapper):
                # Teacher is Cross-Encoder: scores pairs directly
                pairs = list(zip(batch_queries, batch_docs))
                teacher_scores = torch.tensor(self.teacher.predict(pairs)).to(
                    self.device
                )
            else:
                # Teacher is Bi-Encoder: cosine similarity
                t_q_emb = self.teacher.get_embeddings(batch_queries)
                t_d_emb = self.teacher.get_embeddings(batch_docs)
                teacher_scores = F.cosine_similarity(t_q_emb, t_d_emb)

        # 2. Get Student Logits/Scores
        # Student is typically a Bi-Encoder in retrieval scenarios
        s_q_emb = self.student.get_embeddings(batch_queries)
        s_d_emb = self.student.get_embeddings(batch_docs)
        student_scores = F.cosine_similarity(s_q_emb, s_d_emb)

        # 3. Calculate Losses

        if self.distillation_loss_type == "kldiv":
            # KL Divergence: Treat batch as a distribution (List-wise distillation)
            # Softmax over the batch dimension
            # Note: This assumes the batch contains candidates for the SAME query
            # or we treat the whole batch as a ranking list.
            # For mixed batches, this is an approximation.

            # LogSoftmax for Student (Input)
            student_log_probs = F.log_softmax(student_scores / self.temperature, dim=0)

            # Softmax for Teacher (Target)
            teacher_probs = F.softmax(teacher_scores / self.temperature, dim=0)

            distillation_loss = self.kl_div_loss(student_log_probs, teacher_probs) * (
                self.temperature**2
            )

        else:
            # MSE: Point-wise regression
            # Ensure shapes match
            if teacher_scores.shape != student_scores.shape:
                teacher_scores = teacher_scores.view_as(student_scores)

            distillation_loss = self.mse_loss(student_scores, teacher_scores)

        # Student Loss (Ground Truth)
        # If we have hard labels (e.g., 0 or 1), use MSE or Margin loss
        labels = labels.to(self.device)
        if labels.shape != student_scores.shape:
            labels = labels.view_as(student_scores)

        student_loss = self.mse_loss(student_scores, labels)

        # Combined Loss
        loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss

        loss.backward()
        optimizer.step()

        return {
            "loss": loss.item(),
            "distillation_loss": distillation_loss.item(),
            "student_loss": student_loss.item(),
        }

    def cascade_predict(self, input_data: Any, threshold: float = 0.8) -> Any:
        """
        Simple cascade prediction:
        Try Student first. If confidence < threshold, use Teacher.
        """
        # This is a high-level abstraction.
        # For retrieval, use DistillationCascade.retrieve_and_rerank instead.
        pass
