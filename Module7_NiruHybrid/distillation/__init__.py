"""
Model Distillation Cascade for AmaniQuery
=========================================

This sub-module implements a teacher-student distillation cascade to optimize
retrieval and generation performance. It allows a large, accurate "Teacher" model
to train or guide a smaller, faster "Student" model, or to use a cascade where
the Student filters results for the Teacher.

Components:
- TeacherStudentPair: Manages the teacher and student models.
- DistillationCascade: Implements the cascade logic (Student -> Teacher).
- ModelWrapper, BiEncoderWrapper, CrossEncoderWrapper: Wrappers for different model types.
"""

from .teacher_student import (
    TeacherStudentPair,
    ModelWrapper,
    BiEncoderWrapper,
    CrossEncoderWrapper,
)
from .distillation_cascade import DistillationCascade

__all__ = [
    "TeacherStudentPair",
    "ModelWrapper",
    "BiEncoderWrapper",
    "CrossEncoderWrapper",
    "DistillationCascade",
]
