# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Callable, Iterable
from typing import Any, TypeVar

T = TypeVar("T")
U = TypeVar("U")


class FinishedMarker:
    pass


async def bucket(
    num_buckets: int,
    it: AsyncGenerator[T, None],
    process: Callable[[T], Iterable[tuple[int, U]]],
) -> list[AsyncGenerator[U, None]]:
    """
    Wraps `it` and distributes its items into `num_buckets` buckets based on the
    result of `process`.

    Similar to https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.bucket
    but with the following differences:
    - The buckets are integers in [0, num_buckets).
    - The `process` callable returns both the bucket and a transformed version
      of the element from `it`.
    """
    queues: list[asyncio.Queue[U | FinishedMarker]] = [
        asyncio.Queue() for _ in range(num_buckets)
    ]
    # Keep track of how many consumers are still open, so if they're all closed,
    # `fill_queues` can finish and close `it` early.
    num_open_consumers = num_buckets

    async def fill_queues() -> None:
        try:
            async with contextlib.aclosing(it):
                async for item in it:
                    if num_open_consumers == 0:
                        break
                    for bucket_index, processed_item in process(item):
                        queues[bucket_index].put_nowait(processed_item)
        finally:
            for queue in queues:
                queue.put_nowait(FinishedMarker())

    fill_queues_task = asyncio.create_task(fill_queues())

    async def gen(queue: asyncio.Queue[U | FinishedMarker]) -> AsyncGenerator[U, None]:
        nonlocal num_open_consumers
        try:
            # Below, we will run the generator for one iteration for returning it,
            # to ensure that it starts executing.  We would like this so that when
            # the generator is closed, it runs the finally block below to decrement
            # `num_open_consumers`. If the generator is closed before it has run any
            # iterations, then the finalizer below will not run, and
            # `num_open_consumers` will not be decremented.
            # Therefore, we yield nothing once here.
            yield  # type: ignore
            while True:
                next_item = await queue.get()
                if isinstance(next_item, FinishedMarker):
                    break
                yield next_item

            # Check that there were no exceptions in the fill_queues_task
            # TODO(richard): If there was an exception, the printed traceback isn't very useful; improve that.
            await fill_queues_task
        finally:
            num_open_consumers -= 1

    # Run each generator for one iteration.
    result = [gen(queue) for queue in queues]
    for result_item in result:
        await anext(result_item)
    return result


async def consume(it: AsyncGenerator[Any, None]) -> None:
    """Fully executes the async generator `it` while discarding all output."""
    async with contextlib.aclosing(it):
        async for _ in it:
            pass
