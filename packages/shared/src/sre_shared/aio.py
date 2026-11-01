"""asyncio entry point that works on Windows.

psycopg's async driver cannot use the Proactor loop that Windows defaults to, so CLIs call
`run()` instead of `asyncio.run()`.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Coroutine
from typing import Any


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    if sys.platform == "win32":
        return asyncio.run(coro, loop_factory=asyncio.SelectorEventLoop)
    return asyncio.run(coro)
