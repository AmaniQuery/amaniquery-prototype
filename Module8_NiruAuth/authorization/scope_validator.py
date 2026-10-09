"""
Scope Validator
Validates OAuth scopes and API key scopes
"""

from typing import List, Optional


class ScopeValidator:
    """Validates scopes"""

    @staticmethod
    def validate_scopes(
        requested_scopes: List[str], available_scopes: List[str]
    ) -> List[str]:
        """Validate requested scopes against available scopes"""
        if "*" in available_scopes:
            return requested_scopes

        valid_scopes = []
        for scope in requested_scopes:
            if scope in available_scopes:
                valid_scopes.append(scope)
            # Check for wildcard scopes (e.g., "read" matches "query:read")
            elif any(
                available_scope.endswith(f":{scope}") or available_scope == scope
                for available_scope in available_scopes
            ):
                valid_scopes.append(scope)

        return valid_scopes

    @staticmethod
    def has_scope(scopes: Optional[List[str]], required_scope: str) -> bool:
        """Check if scopes include required scope"""
        if not scopes:
            return False

        if "*" in scopes:
            return True

        if required_scope in scopes:
            return True

        # Check for resource wildcard (e.g., "query:*" matches "query:read")
        if ":" in required_scope:
            resource = required_scope.split(":")[0]
            resource_wildcard = f"{resource}:*"
            if resource_wildcard in scopes:
                return True

        return False

    @staticmethod
    def require_scope(scopes: Optional[List[str]], required_scope: str) -> bool:
        """Require scope or raise exception"""
        if not ScopeValidator.has_scope(scopes, required_scope):
            raise PermissionError(f"Scope required: {required_scope}")
        return True
