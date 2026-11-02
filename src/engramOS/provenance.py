"""Shared validation for caller-provided conversational provenance labels."""

from __future__ import annotations


ALLOWED_ROLES = frozenset({"user", "assistant", "system", "tool"})
ALLOWED_SOURCE_TYPES = frozenset(
    {"user_turn", "assistant_turn", "system_policy", "tool", "external_doc"}
)


class ProvenanceError(ValueError):
    """Raised when a role/source pair is unknown or inconsistent."""


_REQUIRED_ROLE_BY_SOURCE_TYPE = {
    "user_turn": "user",
    "assistant_turn": "assistant",
    "system_policy": "system",
}


def validate_role_source_type(role: str, source_type: str) -> None:
    """Validate known role and source values and the asymmetric trust labels."""
    if not isinstance(role, str) or role not in ALLOWED_ROLES:
        raise ProvenanceError(f"role must be one of {sorted(ALLOWED_ROLES)}")
    if not isinstance(source_type, str) or source_type not in ALLOWED_SOURCE_TYPES:
        raise ProvenanceError(
            f"source_type must be one of {sorted(ALLOWED_SOURCE_TYPES)}"
        )

    required_role = _REQUIRED_ROLE_BY_SOURCE_TYPE.get(source_type)
    if required_role is not None and role != required_role:
        raise ProvenanceError(
            "role/source_type mismatch: "
            f"source_type={source_type!r} requires role={required_role!r}; "
            f"received role={role!r}"
        )
    if role == "tool" and source_type not in {"tool", "external_doc"}:
        raise ProvenanceError(
            "role/source_type mismatch: role='tool' requires source_type='tool' "
            f"or 'external_doc'; received role={role!r}, source_type={source_type!r}"
        )
