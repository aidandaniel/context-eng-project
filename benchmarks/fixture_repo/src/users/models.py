"""User domain models.

Plain dataclasses standing in for an ORM. Kept free of persistence logic so the
service layer can be tested without a database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Role(str, Enum):
    ADMIN = "admin"
    MEMBER = "member"
    GUEST = "guest"


@dataclass
class User:
    id: str
    email: str
    role: Role = Role.MEMBER
    active: bool = True
    display_name: str = ""

    def can_manage(self) -> bool:
        return self.role == Role.ADMIN

    def deactivate(self) -> None:
        self.active = False


@dataclass
class UserProfile:
    user: User
    bio: str = ""
    preferences: dict = field(default_factory=dict)
