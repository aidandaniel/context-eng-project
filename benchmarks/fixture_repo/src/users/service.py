"""User service: creation, lookup, search, and lifecycle operations.

Backed by an in-memory store for the fixture. Real implementations would inject
a repository, but the call shape (create_user/get_user/search/deactivate) is
what matters for retrieval scenarios. The store, pagination helpers, and
validation make this a realistically sized service module.
"""

from __future__ import annotations

import re
import uuid

from src.users.models import Role, User, UserProfile
from src.utils.logging import get_logger

logger = get_logger(__name__)

_USERS: dict[str, User] = {}
_PROFILES: dict[str, UserProfile] = {}

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class UserExistsError(Exception):
    """Raised when creating a user whose email is already registered."""


class InvalidEmailError(Exception):
    """Raised when an email fails basic format validation."""


def _validate_email(email: str) -> None:
    if not _EMAIL_RE.match(email):
        raise InvalidEmailError(email)


def _find_by_email(email: str) -> User | None:
    for user in _USERS.values():
        if user.email == email:
            return user
    return None


def create_user(email: str, role: Role = Role.MEMBER) -> User:
    """Create and store a new user, returning the created record."""
    _validate_email(email)
    if _find_by_email(email):
        raise UserExistsError(email)
    user = User(id=uuid.uuid4().hex, email=email, role=role)
    _USERS[user.id] = user
    _PROFILES[user.id] = UserProfile(user=user)
    logger.info("created user %s", user.id)
    return user


def get_user(user_id: str) -> User | None:
    return _USERS.get(user_id)


def get_profile(user_id: str) -> UserProfile | None:
    return _PROFILES.get(user_id)


def update_role(user_id: str, role: Role) -> bool:
    user = _USERS.get(user_id)
    if user is None:
        return False
    user.role = role
    return True


def list_users(active_only: bool = False) -> list[User]:
    users = list(_USERS.values())
    if active_only:
        users = [u for u in users if u.active]
    return sorted(users, key=lambda u: u.email)


def paginate(items: list[User], page: int, size: int) -> list[User]:
    start = max(0, (page - 1) * size)
    return items[start : start + size]


def deactivate_user(user_id: str) -> bool:
    user = _USERS.get(user_id)
    if user is None:
        return False
    user.deactivate()
    return True


def count_users() -> int:
    return len(_USERS)
