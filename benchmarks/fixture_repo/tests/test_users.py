"""Tests for the user service."""

import pytest

from src.users.models import Role
from src.users.service import UserExistsError, create_user, deactivate_user, get_user


def test_create_and_get_user():
    user = create_user("a@example.com", role=Role.ADMIN)
    assert get_user(user.id) is user
    assert user.can_manage()


def test_duplicate_email_raises():
    create_user("dup@example.com")
    with pytest.raises(UserExistsError):
        create_user("dup@example.com")


def test_deactivate_user():
    user = create_user("d@example.com")
    assert deactivate_user(user.id) is True
    assert get_user(user.id).active is False
