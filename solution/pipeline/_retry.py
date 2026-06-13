from __future__ import annotations

import logging
import random
import time
from functools import wraps
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)


def retry(retries: int = 5, base_delay: float = 5.0) -> Callable[[F], F]:
    """Exponential-backoff retry for transient network errors at external API boundaries."""
    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    if attempt == retries:
                        raise
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        "%s intento %d/%d fallido: %r; reintentando en %.1fs"
                      , fn.__name__
                      , attempt + 1
                      , retries
                      , exc
                      , delay
                    )
                    time.sleep(delay)
        return wrapper  # type: ignore[return-value]
    return decorator
