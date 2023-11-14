# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import ast
import dataclasses
import json
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Iterator,
    Sequence,
)
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import aiohttp
import tiktoken

from lmapi.async_tools import asyncitertools, limits
from lmapi.async_tools.server_sent_events import ServerSentEvent, parse_event_stream
from lmapi.http_settings import HttpSettingsProvider
from lmapi.lm import (
    LM,
    Capabilities,
    Completion,
    CompletionsSettings,
    SampledToken,
    TokenWithLogprob,
)

client_session: ContextVar[aiohttp.ClientSession] = ContextVar("client_session")


@dataclass(frozen=True)
class OpenAIAPIError(Exception):
    """Indicates a general OpenAI call error."""

    status_code: int
    text: str

    @property
    def user_message(self) -> str:
        return "Model communication failure"

    @cached_property
    def debug_message(self) -> str:
        return f"Unexpected status code: {self.status_code}. {self.text}"


@dataclass(slots=True)
class OpenAITokenWithLogprob:
    if TYPE_CHECKING:

        def _check_protocol(self) -> TokenWithLogprob:
            return self

    text: str
    logprob: float
    _encoding: tiktoken.Encoding

    _bytes: bytes | None = dataclasses.field(init=False, default=None)

    @property
    def bytes(self) -> bytes:
        if self._bytes is None:
            self._bytes = openai_token_to_bytes(self.text)
        return self._bytes

    _token_id: int | None = dataclasses.field(init=False, default=None)

    @property
    def token_id(self) -> int:
        if self._token_id is None:
            self._token_id = self._encoding.encode_single_token(self.bytes)
        return self._token_id


class _NextLogprobsFunction(Protocol):
    async def __call__(self, prompt: str | Sequence[int]) -> dict[int, float]:
        ...


class _CompletionsFunction(Protocol):
    async def __call__(
        self,
        prompt: str | Sequence[int] | Sequence[str] | Sequence[Sequence[int]],
        settings: CompletionsSettings | None = None,
    ) -> Sequence[Completion]:
        ...


@dataclass(frozen=True)
class OpenAI:
    """Implementation of the LM protocol for OpenAI models.

    Arguments:
        url: The URL of the model.
        auth_provider: The authorization provider to use.
        encoding: The encoding of the model.
        default_completion_settings: The default completion settings to use.
    """

    if TYPE_CHECKING:

        def _check_protocol(self) -> LM:
            return self

    http_settings_provider: HttpSettingsProvider
    encoding: tiktoken.Encoding
    default_completion_settings: dict[str, Any]
    additional_headers: dict[str, str]
    capabilities: Capabilities
    request_limiter: limits.AdaptiveLimiter | None = None

    @staticmethod
    def create(
        http_settings_provider: HttpSettingsProvider,
        encoding_or_name: str | tiktoken.Encoding,
        capabilities: Capabilities,
        request_limiter: limits.AdaptiveLimiter | None = None,
        default_completion_settings: dict[str, Any] | None = None,
        additional_headers: dict[str, Any] | None = None,
    ) -> "OpenAI":
        encoding = (
            tiktoken.get_encoding(encoding_or_name)
            if isinstance(encoding_or_name, str)
            else encoding_or_name
        )

        return OpenAI(
            http_settings_provider,
            encoding,
            default_completion_settings or {},
            additional_headers or {},
            capabilities,
            request_limiter,
        )

    async def completions(
        self,
        prompt: str | Sequence[int] | Sequence[str] | Sequence[Sequence[int]],
        settings: CompletionsSettings | None = None,
    ) -> Sequence[Completion]:
        return await self._completions_maybe_limited(prompt, settings)

    @cached_property
    def _completions_maybe_limited(self) -> _CompletionsFunction:
        if self.request_limiter is None:
            return self._completions
        return self.request_limiter.wrap_async_callable(self._completions)

    async def _completions(
        self,
        prompt: str | Sequence[int] | Sequence[str] | Sequence[Sequence[int]],
        settings: CompletionsSettings | None = None,
    ) -> Sequence[Completion]:
        """
        Inner implementation of `completions` before `self.request_limiter` is applied.

        Must be called in an `async with client_session:` block where
        `client_session` is the same one used to construct this object.
        """
        params = self._make_params(prompt, settings)
        conn_settings = self.http_settings_provider()

        async with client_session.get().post(
            conn_settings.url,
            headers={**conn_settings.headers, **self.additional_headers},
            json=params,
        ) as response:
            if response.status != 200:
                if response.status in (408, 429, 500):
                    raise limits.RateLimitExceededError()
                else:
                    raise OpenAIAPIError(response.status, await response.text())
            resp = await response.json(content_type=None)
            if resp is None:
                raise limits.RateLimitExceededError()
        result = [
            extract_sampled_tokens(choice["logprobs"], self.encoding)
            for choice in resp["choices"]
        ]
        return result

    async def streaming_completions(
        self, prompt: str | Sequence[int], settings: CompletionsSettings | None = None
    ) -> Sequence[AsyncGenerator[SampledToken, None]]:
        """Please see docstring for LM.streaming_completions."""

        params = self._make_params(prompt, settings)
        n = params.get("n", 1)

        async def drop_unneeded(
            events: AsyncIterable[ServerSentEvent],
        ) -> AsyncIterator[dict[str, Any]]:
            """Process the stream of ServerSentEvents to drop the ones that we don't need.

            We drop:
            - The [DONE] event, which normally occurs one at the end
            - Unexpected events where there are no tokens sampled
            """
            async for event in events:
                assert event.data is not None
                if event.data == "[DONE]\n":
                    continue

                data = json.loads(event.data)
                assert len(data["choices"]) == 1
                choice = data["choices"][0]

                if len(choice["logprobs"].get("tokens", [])) == 0:
                    # Sometimes the API sends us an event even though no tokens were sampled.
                    # Ignore those cases.
                    continue

                yield choice

        def process_choice(
            choice: dict[str, Any]
        ) -> Iterator[tuple[int, SampledToken]]:
            """Extracts the data returned by OpenAI's API into the SampledText object."""
            # choice["text"] is always a valid Unicode string, and may correspond to multiple tokens.
            # Sometimes, choice["text"] is empty while choice["logprobs"]["tokens"] is not.
            # This seems to happen when choice["logprobs"]["tokens"] doesn't concatenate into valid UTF-8.
            # sampled_text =  SampledText(choice["text"], choice["finish_reason"])
            for sampled_tokens in extract_sampled_tokens(
                choice["logprobs"], self.encoding
            ):
                yield choice["index"], sampled_tokens

        if n == 1:

            async def gen_1() -> AsyncGenerator[SampledToken, None]:
                async with self.streaming_completions_client_response(
                    params
                ) as response:
                    # We use "utf-8" because
                    # https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation says:
                    # > Streams must be decoded using the UTF-8 decode algorithm.
                    events = parse_event_stream(
                        _bytes_to_str(response.content, "utf-8")
                    )
                    # `response.content` is an async iterable which returns sequences of
                    # bytes ending in b'\n'. That is slightly inappropriate because the SSE
                    # specification allows lines to end in b'\r\n' or b'\r' as well.
                    # But in practice, the GPT server never seems to use \r\n or \r.
                    async for c in drop_unneeded(events):
                        for _, item in process_choice(c):
                            yield item

            return [gen_1()]
        else:

            async def gen_n() -> AsyncGenerator[dict[str, Any], None]:
                async with self.streaming_completions_client_response(
                    params
                ) as response:
                    events = parse_event_stream(
                        _bytes_to_str(response.content, "utf-8")
                    )
                    async for event in drop_unneeded(events):
                        yield event

            return await asyncitertools.bucket(n, gen_n(), process_choice)

    @cached_property
    def streaming_completions_client_response(
        self,
    ) -> Callable[
        [dict[str, Any]], AbstractAsyncContextManager[aiohttp.ClientResponse]
    ]:
        """Helper function for `streaming_completions`.

        This method returns `aiohttp.ClientResponse`; the data retrieved from it is parsed by `streaming_completions`.
        """
        if self.request_limiter is None:
            return self._streaming_completions_client_response
        return self.request_limiter.wrap_async_context_manager_producer(
            self._streaming_completions_client_response
        )

    @asynccontextmanager
    async def _streaming_completions_client_response(
        self, params: dict[str, Any]
    ) -> AsyncIterator[aiohttp.ClientResponse]:
        """Call as: `with gpt_impl._streaming_completions_client_response(params) as response: ..."""

        conn_settings = self.http_settings_provider()
        async with client_session.get().post(
            conn_settings.url,
            headers={**conn_settings.headers, **self.additional_headers},
            json={**params, "stream": True},
        ) as response:
            if response.status != 200:
                if response.status in (408, 429, 500):
                    raise limits.RateLimitExceededError(
                        response.status, await response.text()
                    )
                else:
                    raise OpenAIAPIError(response.status, await response.text())
            yield response

    def _make_params(
        self,
        prompt: str | Sequence[int] | Sequence[str] | Sequence[Sequence[int]],
        settings: CompletionsSettings | None,
    ) -> dict[str, Any]:
        params = {"prompt": prompt, "logprobs": 0, **self.default_completion_settings}
        if settings is not None:
            params.update(_filter_none_values(dataclasses.asdict(settings)))
        return params


K = TypeVar("K")
V = TypeVar("V")


def _filter_none_values(d: dict[K, V | None]) -> dict[K, V]:
    return {k: v for k, v in d.items() if v is not None}


async def _bytes_to_str(
    bs: AsyncIterable[bytes], encoding: str
) -> AsyncGenerator[str, None]:
    async for b in bs:
        yield b.decode(encoding)


def openai_token_to_bytes(token: str) -> bytes:
    if token.startswith("bytes:"):
        return ast.literal_eval(f"b'{token[6:]}'")
    else:
        return token.encode("utf-8")


def extract_sampled_tokens(
    logprobs_info: dict, encoding: tiktoken.Encoding
) -> list[SampledToken]:
    tokens = logprobs_info["tokens"]
    token_logprobs = logprobs_info["token_logprobs"]
    top_logprobs = logprobs_info.get("top_logprobs")
    if top_logprobs is None:
        top_logprobs = [{}] * len(tokens)

    sampled_tokens: list[SampledToken] = []
    for token, token_logprob, top_logprobs_for_token in zip(
        tokens, token_logprobs, top_logprobs, strict=True
    ):
        if top_logprobs_for_token is None:
            top_logprobs_for_token = {}
        sampled_tokens.append(
            SampledToken(
                OpenAITokenWithLogprob(token, token_logprob, encoding),
                tuple(
                    OpenAITokenWithLogprob(t, lp, encoding)
                    for t, lp in top_logprobs_for_token.items()
                ),
            )
        )

    return sampled_tokens
