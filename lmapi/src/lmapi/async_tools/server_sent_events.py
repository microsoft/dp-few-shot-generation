# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass


@dataclass
class ServerSentEvent:
    event: str | None = None
    data: str | None = None
    id: str | None = None
    retry: int | None = None


async def parse_event_stream(
    lines: AsyncIterable[str],
) -> AsyncIterator[ServerSentEvent]:
    """Implements part of https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation
    This function doesn't implement the logic specified there for dispatching the event; it only performs parsing.
    """
    next_event = ServerSentEvent()

    async for line in lines:
        line = line.rstrip("\n")
        if len(line) == 0:
            yield next_event
            next_event = ServerSentEvent()
            continue

        if line[0] == ":":
            continue

        field, _, value = line.partition(":")
        if value[0] == " ":
            value = value[1:]
        if field == "event":
            next_event.event = value
        elif field == "data":
            if next_event.data is None:
                next_event.data = value + "\n"
            else:
                next_event.data += value + "\n"
        elif field == "id":
            next_event.id = value
        elif field == "retry":
            try:
                next_event.retry = int(value)
            except ValueError:
                pass

    # No need to handle `next_event` now, as the specification states:
    #   Once the end of the file is reached, any pending data must be discarded.
    #   (If the file ends in the middle of an event, before the final empty line,
    #   the incomplete event is not dispatched.)
