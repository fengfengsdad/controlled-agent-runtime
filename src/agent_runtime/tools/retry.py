from __future__ import annotations

import time
from enum import Enum
from typing import Callable, TypeVar

T = TypeVar("T")


class ErrorClass(str, Enum):
    RETRYABLE = "retryable"
    PERMANENT = "permanent"


class ClassifiedError(Exception):
    def __init__(self, message: str, error_class: ErrorClass) -> None:
        super().__init__(message)
        self.error_class = error_class


def classify_exception(exc: Exception) -> ErrorClass:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    retryable_markers = ("timeout", "temporar", "unavailable", "connection", "429", "502", "503")
    if any(marker in name or marker in message for marker in retryable_markers):
        return ErrorClass.RETRYABLE
    return ErrorClass.PERMANENT


def with_backoff(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay_sec: float = 0.05,
) -> T:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if classify_exception(exc) == ErrorClass.PERMANENT or attempt == max_attempts:
                raise ClassifiedError(str(exc), classify_exception(exc)) from exc
            time.sleep(base_delay_sec * (2 ** (attempt - 1)))
    assert last_exc is not None
    raise last_exc
