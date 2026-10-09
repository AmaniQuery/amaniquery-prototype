"""
Password Validator Service
Enforces password strength rules and prevents common/predictable passwords
"""

import re
from typing import List, Tuple
from loguru import logger

from ..config import config

COMMON_PASSWORDS = {
    "password",
    "12345678",
    "password1",
    "password123",
    "qwerty123",
    "admin123",
    "letmein",
    "welcome123",
    "kenya123",
    "nairobi123",
    "abc123",
    "123456789",
    "password1234",
    "changeme",
    "1234",
    "ilovekenya",
    "passw0rd",
    "Password1",
    "Qwerty123",
}


class PasswordValidationResult:
    def __init__(self, valid: bool, errors: List[str], score: int = 0):
        self.valid = valid
        self.errors = errors
        self.score = score

    def __bool__(self):
        return self.valid


class PasswordValidator:
    """Validates passwords against configured rules"""

    MIN_LENGTH = config.PASSWORD_MIN_LENGTH
    REQUIRE_UPPERCASE = config.PASSWORD_REQUIRE_UPPERCASE
    REQUIRE_LOWERCASE = config.PASSWORD_REQUIRE_LOWERCASE
    REQUIRE_DIGIT = config.PASSWORD_REQUIRE_DIGIT
    REQUIRE_SPECIAL = config.PASSWORD_REQUIRE_SPECIAL
    MAX_LENGTH = 128

    @classmethod
    def validate(cls, password: str) -> PasswordValidationResult:
        """Validate password strength against all configured rules"""
        errors: List[str] = []
        score = 0

        if not password:
            return PasswordValidationResult(False, ["Password is required"], 0)

        if len(password) < cls.MIN_LENGTH:
            errors.append(f"Password must be at least {cls.MIN_LENGTH} characters")

        if len(password) > cls.MAX_LENGTH:
            errors.append(f"Password must not exceed {cls.MAX_LENGTH} characters")

        if cls.REQUIRE_UPPERCASE and not re.search(r"[A-Z]", password):
            errors.append("Password must contain at least one uppercase letter")

        if cls.REQUIRE_LOWERCASE and not re.search(r"[a-z]", password):
            errors.append("Password must contain at least one lowercase letter")

        if cls.REQUIRE_DIGIT and not re.search(r"\d", password):
            errors.append("Password must contain at least one number")

        if cls.REQUIRE_SPECIAL and not re.search(
            r'[!@#$%^&*(),.?":{}|<>_\-]', password
        ):
            errors.append("Password must contain at least one special character")

        normalized = password.lower().strip()
        if normalized in COMMON_PASSWORDS:
            errors.append("This password is too common. Choose a more unique password")

        common_patterns = [
            r"password",
            r"1234",
            r"qwerty",
            r"asdf",
            r"nairobi",
            r"kenya",
            r"admin",
            r"welcome",
            r"letmein",
            r"changeme",
        ]
        for pattern in common_patterns:
            if re.search(pattern, normalized):
                errors.append("Password contains a common word pattern")
                break

        consecutive = cls._check_consecutive(password)
        if consecutive:
            errors.append(consecutive)

        if not errors:
            length = len(password)
            if length >= 16:
                score = 5
            elif length >= 12:
                score = 4
            elif length >= 10:
                score = 3
            else:
                score = 2

            score += sum(
                [
                    2 if cls.REQUIRE_UPPERCASE and re.search(r"[A-Z]", password) else 0,
                    2 if cls.REQUIRE_LOWERCASE and re.search(r"[a-z]", password) else 0,
                    2 if cls.REQUIRE_DIGIT and re.search(r"\d", password) else 0,
                    (
                        2
                        if cls.REQUIRE_SPECIAL
                        and re.search(r'[!@#$%^&*(),.?":{}|<>_\-]', password)
                        else 0
                    ),
                    1 if re.search(r"[^A-Za-z0-9]", password) else 0,
                ]
            )

        return PasswordValidationResult(len(errors) == 0, errors, score)

    @classmethod
    def _check_consecutive(cls, password: str) -> str:
        """
        Check for repeated characters (e.g., 'aaa', '111').
        Also detect keyboard sequences.
        """
        for i in range(len(password) - 2):
            if password[i] == password[i + 1] == password[i + 2]:
                return "Password contains repeated characters (e.g., 'aaa')"

        keyboard_rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
        lower = password.lower()
        for row in keyboard_rows:
            for i in range(len(row) - 2):
                seq = row[i : i + 3]
                if seq in lower:
                    return f"Password contains a keyboard sequence ({seq})"
                if seq[::-1] in lower:
                    return (
                        f"Password contains a reversed keyboard sequence ({seq[::-1]})"
                    )

        return ""
