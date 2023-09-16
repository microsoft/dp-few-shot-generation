# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import itertools
from collections.abc import AsyncGenerator

import pytest

from lmapi.async_tools.asyncitertools import bucket


@pytest.mark.asyncio
async def test_bucket_normal() -> None:
    async def gen() -> AsyncGenerator[int, None]:
        for i in range(4):
            if i % 2 == 0:
                await asyncio.sleep(0)
            yield i

    it_indices_permutations = set(itertools.permutations(i % 3 for i in range(4)))
    for it_indices in it_indices_permutations:
        bucketed_its = await bucket(3, gen(), lambda x: [(x % 3, x)])
        regular_its = [iter([0, 3]), iter([1]), iter([2])]
        for it_index in it_indices:
            assert await anext(bucketed_its[it_index]) == next(regular_its[it_index])

        for it in bucketed_its:
            with pytest.raises(StopAsyncIteration):
                await anext(it)

        for it in regular_its:
            with pytest.raises(StopIteration):
                next(it)


class ExceptionForTest(Exception):
    pass


@pytest.mark.asyncio
async def test_bucket_exception() -> None:
    async def gen() -> AsyncGenerator[int, None]:
        for i in range(6):
            if i >= 5:
                raise ExceptionForTest

            if i % 2 == 0:
                await asyncio.sleep(0)
            yield i

    it0, it1, it2 = await bucket(3, gen(), lambda x: [(x % 3, x)])

    assert await anext(it0) == 0
    assert await anext(it0) == 3
    with pytest.raises(ExceptionForTest):
        await anext(it0)

    assert await anext(it1) == 1
    assert await anext(it1) == 4
    with pytest.raises(ExceptionForTest):
        await anext(it1)

    assert await anext(it2) == 2
    with pytest.raises(ExceptionForTest):
        await anext(it2)


@pytest.mark.asyncio
async def test_bucket_closing() -> None:
    """Tests that `bucket` closes the generator when all consumers are closed."""

    lock = asyncio.Lock()
    max_i = None
    gen_was_closed = False

    async def gen() -> AsyncGenerator[int, None]:
        nonlocal max_i, gen_was_closed
        try:
            for i in range(9):
                async with lock:
                    max_i = i
                    yield i
                    await asyncio.sleep(0)
        except GeneratorExit:
            gen_was_closed = True

    async with lock:
        it0, it1, it2 = await bucket(3, gen(), lambda x: [(x % 3, x)])
        await it0.aclose()
        await it2.aclose()

    # Give up control so that `gen` executes one iteration, yielding 0
    await asyncio.sleep(0)
    # Give up control so that `gen` executes one iteration, yielding 1
    await asyncio.sleep(0)

    async with lock:
        assert await anext(it1) == 1
        await it1.aclose()

    # `fill_queues` will run more iteration of `gen` and then realize that
    # `num_open_consumers == 0`, thus closing `gen` as well
    await asyncio.sleep(0)
    assert max_i == 2
    assert gen_was_closed

    # There should be no changes due to this additional sleep
    await asyncio.sleep(0)
    assert max_i == 2
    assert gen_was_closed
