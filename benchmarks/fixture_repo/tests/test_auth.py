"""Tests for the auth refresh/logout lifecycle."""

import time

import pytest

from src.auth.refresh import RefreshError, Session, logout, refreshToken, rotate_refresh_token


def _session():
    s = Session(user_id="u1", refresh_token=None, issued_at=time.time())
    rotate_refresh_token(s)
    return s


def test_refresh_token_returns_access_token():
    session = _session()
    token = refreshToken(session)
    assert isinstance(token, str)
    assert token


def test_refresh_after_logout_raises():
    session = _session()
    logout(session)
    with pytest.raises(RefreshError):
        refreshToken(session)


def test_expired_refresh_token_raises():
    session = _session()
    session.issued_at = 0.0
    with pytest.raises(RefreshError):
        refreshToken(session)
