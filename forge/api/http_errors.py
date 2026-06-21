"""Map sandbox security errors to HTTP responses."""

from __future__ import annotations

from typing import NoReturn

from fastapi import HTTPException

from seiso.security import SecurityError


def raise_forbidden(exc: SecurityError) -> NoReturn:
    raise HTTPException(403, str(exc)) from exc
