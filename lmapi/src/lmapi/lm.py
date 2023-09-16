# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import abstractmethod
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias

import tiktoken


class TokenWithLogprob(Protocol):
    @property
    def text(self) -> str:
        """The textual representation of the token.

        For tokens that don't constitute valid UTF-8, this will look like "bytes:\\x??\\x??".
        """
        ...

    @property
    def bytes(self) -> bytes:
        "The bytes that make up the token."
        ...

    @property
    def logprob(self) -> float:
        ...

    @property
    def token_id(self) -> int:
        ...


@dataclass
class SampledToken:
    token: TokenWithLogprob
    top_choices: tuple[TokenWithLogprob, ...]


# Types:
# - rules for how we can interrupt the completion: stop, etc.
# - whether we can get logprobs

Completion: TypeAlias = Sequence[SampledToken]


@dataclass
class CompletionsSettings:
    """The meanings are the same as in https://learn.microsoft.com/en-us/azure/cognitive-services/openai/reference#completions

    If any field is set to None, the defaults from that documentation page apply."""

    temperature: float | None = None
    max_tokens: int | None = None
    n: int | None = None
    stop: str | Sequence[str] | None = None
    logprobs: int | None = None
    echo: bool | None = None
    logit_bias: Mapping[int, float] | None = None
    top_p: float | None=None


class LM(Protocol):
    @abstractmethod
    async def completions(
        self,
        prompt: str | Sequence[int] | Sequence[str] | Sequence[Sequence[int]],
        settings: CompletionsSettings | None = None,
    ) -> Sequence[Completion]:
        """Returns completions given the prompt."""
        ...

    @abstractmethod
    async def streaming_completions(
        self, prompt: str | Sequence[int], settings: CompletionsSettings | None = None
    ) -> Sequence[AsyncGenerator[SampledToken, None]]:
        """Returns completions in a streaming fashion as async generators of SampledText.

        To ensure that the HTTP connection to GPT is closed quickly, use `contextlib.aclosing`:
        ```
        # streams is a sequence with length equal to `n` in the settings,
        # the number of completions to generate given the prompt.
        # In this example, we assume n = 1.
        streams = gpt.streaming_completions(prompt)
        [stream] = streams
        async with contextlib.aclosing(stream):
            async for sampled_text in stream:
                ...
                if some_condition:
                    break
        ```

        This pattern ensures that when the loop exits with the `break`, `it.aclose()` is called immediately,
        rather than only when `it` is garbage collected.
        """
        ...

    @property
    def encoding(self) -> tiktoken.Encoding:
        """Returns the encoding scheme the model uses."""
        ...
