# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import dataclasses
from dataclasses import InitVar, dataclass
from typing import Protocol


class AuthorizationProvider(Protocol):
    def headers(self) -> dict[str, str]:
        """Returns the headers to be used for authorization."""
        ...


@dataclass
class OpenAiApiKey(AuthorizationProvider):
    key: InitVar[str]
    _headers: dict[str, str] = dataclasses.field(init=False)

    def __post_init__(self, key: str) -> None:
        self._headers = {"Authorization": f"Bearer {key}"}

    def headers(self) -> dict[str, str]:
        return self._headers


@dataclass
class AoaiApiKey(AuthorizationProvider):
    key: InitVar[str]
    _headers: dict[str, str] = dataclasses.field(init=False)

    def __post_init__(self, key: str) -> None:
        self._headers = {"api-key": key}

    def headers(self) -> dict[str, str]:
        return self._headers
