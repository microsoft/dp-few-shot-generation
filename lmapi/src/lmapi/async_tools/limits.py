# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import collections
import dataclasses
import functools
import time
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Any, TypeVar, cast

_MICROS_RATIO = 1_000_000
_NANOS_RATIO = 1_000_000_000


@dataclass
class TokenBucket:
    """Used to prevent too much resource consumption during any time period.

    These tokens are not words (like in NLP) and the bucket is not a map.
    Instead the tokens are like coins and the bucket is like a physical bucket.

    See https://en.wikipedia.org/wiki/Token_bucket#Algorithm for more details.
    However, unlike typical implementations of token buckets, this one allows
    the refill rate to change at any time.
    """

    # Maximum number of tokens that can be in the bucket, multiplied by 1_000_000.
    capacity_micros: int
    # Amount of tokens added per second, multipled by 1_000_000.
    refill_rate_micros: int

    # The current amount of tokens in the bucket.
    level_micros: int = 0

    # Time when `level` was last updated, in nanoseconds.
    last_updated_ns: int = dataclasses.field(default_factory=time.monotonic_ns)
    # One entry per call to `claim` that's waiting for the bucket to refill.
    _waiting: collections.deque[tuple[asyncio.Event, int]] = dataclasses.field(
        default_factory=collections.deque
    )
    # Set by the next `claim` in line so that it can be notified if the refill rate changes.
    _next_token_refill: asyncio.Event | None = None

    @staticmethod
    def create(capacity: float, refill_rate: float, initial_level: float = 0):
        assert capacity >= 0
        assert refill_rate >= 0
        assert initial_level >= 0

        return TokenBucket(
            int(capacity * _MICROS_RATIO),
            int(refill_rate * _MICROS_RATIO),
            int(initial_level * _MICROS_RATIO),
        )

    async def claim(self, tokens: float) -> None:
        tokens_micros = int(tokens * _MICROS_RATIO)
        assert tokens_micros <= self.capacity_micros
        del tokens

        event: asyncio.Event | None = None
        try:
            # Check if we have sufficient tokens already
            self._update_level()
            # async with self._lock:
            if self.level_micros >= tokens_micros:
                self.level_micros -= tokens_micros
                return

            event = asyncio.Event()
            self._waiting.append((event, tokens_micros))
            while True:
                # Check whether we're first in line
                first_event, _ = self._waiting[0]
                if event is first_event:
                    # We might already have enough tokens. If so, don't wait for more tokens to fill.
                    if self.level_micros >= tokens_micros:
                        break
                    try:
                        # Wait for the bucket to fill up.
                        self._next_token_refill = event

                        # Compute number of microseconds to wait until bucket will be full,
                        # if the rate doesn't change in the meantime.
                        # Round up to the next microsecond.
                        timeout_micros, remainder = divmod(
                            (tokens_micros - self.level_micros) * _MICROS_RATIO,
                            self.refill_rate_micros,
                        )
                        if remainder:
                            timeout_micros += 1

                        # Sleep for the computed number of microseconds.
                        # We assume that 1) the floating point value for timeout
                        # has sufficient precision to represent the desired
                        # number of microseconds, and 2) wait_for's timer has
                        # sufficient resolution.
                        # If the rate changes before the timeout, the event will fire.
                        await asyncio.wait_for(
                            event.wait(), timeout=timeout_micros / _MICROS_RATIO
                        )

                        # If we reach here, before we refilled all the tokens,
                        # the refill rate changed. Recompute how long we should
                        # wait until the tokens are refilled, and try again.
                        # No matter how the rate changed, we shouldn't have
                        # enough tokens at this point.
                        event.clear()
                        self._next_token_refill = None
                        continue
                    except asyncio.TimeoutError:
                        # We should have collected enough tokens by now to take our turn
                        self._update_level()
                        break
                else:
                    # Wait until we get to the start of the line
                    await event.wait()
                    event.clear()

            # async with self._lock:
            assert self.level_micros >= tokens_micros, (
                self.level_micros,
                tokens_micros,
            )
            self.level_micros -= tokens_micros

            self._waiting.popleft()
            if self._waiting:
                # Tell the next in line that they're first now.
                event, _ = self._waiting[0]
                event.set()
            return

        except asyncio.CancelledError:
            pass

    def add_to_rate(
        self, delta: float, min_rate: float | None = None, max_rate: float | None = None
    ) -> None:
        """Add to the refill rate.

        Supply a negative number to subtract from the rate. However, the
        negative number cannot cause the rate to become negative."""
        new_rate = self.refill_rate_micros + round(delta * _MICROS_RATIO)
        assert new_rate >= 0, "Cannot have a negative refill rate"
        if min_rate is not None:
            new_rate = max(new_rate, round(min_rate * _MICROS_RATIO))
        if max_rate is not None:
            new_rate = min(new_rate, round(max_rate * _MICROS_RATIO))
        self.refill_rate_micros = new_rate
        self.reset_rate(new_rate)

    def multiply_rate(
        self, ratio: float, min_rate: float | None = None, max_rate: float | None = None
    ) -> None:
        """Apply a multiplicative factor to the refill rate."""

        assert ratio >= 0, "Cannot have a negative ratio"
        new_rate = round(self.refill_rate_micros * ratio)
        if min_rate is not None:
            new_rate = max(new_rate, round(min_rate * _MICROS_RATIO))
        if max_rate is not None:
            new_rate = min(new_rate, round(max_rate * _MICROS_RATIO))
        self.reset_rate(new_rate)

    def reset_rate(self, new_rate: float) -> None:
        """Reset the refill rate."""
        # Refill tokens with the previous rate, before updating the rate.
        self._update_level()

        self.refill_rate_micros = round(new_rate * _MICROS_RATIO)
        # Tell the task waiting for tokens to refill to wake up so that we can
        # recompute how much longer it should sleep, with the current rate.
        if self._next_token_refill:
            self._next_token_refill.set()

    def _update_level(self) -> None:
        now = time.monotonic_ns()
        elapsed_nanos = now - self.last_updated_ns
        self.level_micros = min(
            self.capacity_micros,
            self.level_micros
            + (elapsed_nanos * self.refill_rate_micros) // _NANOS_RATIO,
        )
        self.last_updated_ns = now


class RateLimitExceededError(Exception):
    pass


AsyncCallableT = TypeVar("AsyncCallableT", bound=Callable[..., Awaitable[Any]])
AsyncContextManagerProducerT = TypeVar(
    "AsyncContextManagerProducerT",
    bound=Callable[..., AbstractAsyncContextManager[Any]],
)


@dataclass
class AdaptiveLimiter:
    """Adaptively rate-limit the wrapped function.

    When a call to the function is successful, we increase the rate additively;
    if it complains that the rate was exceeded, we decrease the rate multiplicatively.
    """

    initial_qps: float = 10
    max_qps: float = 500
    min_qps: float = 1
    bucket: TokenBucket = dataclasses.field(init=False)

    def __post_init__(self):
        self.bucket = TokenBucket.create(self.max_qps, self.initial_qps)

    def wrap_async_callable(self, func: AsyncCallableT) -> AsyncCallableT:
        @functools.wraps(func)
        async def wrapped(*args, **kwargs):
            while True:
                # Wait our turn
                await self.bucket.claim(1)
                # Try calling the function
                try:
                    result = await func(*args, **kwargs)
                    self.bucket.add_to_rate(1, max_rate=self.max_qps)
                    return result
                except RateLimitExceededError:
                    self.bucket.multiply_rate(0.9, min_rate=self.min_qps)

        return cast(AsyncCallableT, wrapped)

    def wrap_async_context_manager_producer(
        self, func: AsyncContextManagerProducerT
    ) -> AsyncContextManagerProducerT:
        @functools.wraps(func)
        @asynccontextmanager
        async def wrapped(*args, **kwargs):
            while True:
                # Wait our turn
                await self.bucket.claim(1)
                # Try calling the function
                try:
                    async with func(*args, **kwargs) as cm_result:
                        yield cm_result
                    self.bucket.add_to_rate(1, max_rate=self.max_qps)
                    break
                except RateLimitExceededError:
                    self.bucket.multiply_rate(0.9, min_rate=self.min_qps)

        return cast(AsyncContextManagerProducerT, wrapped)
