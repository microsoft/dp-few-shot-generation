# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from typing import Protocol


@dataclass
class HttpSettings:
    """Settings to use while making an HTTP connection.

    This should include the settings necessary to authenticate with the endpoint.
    """

    url: str
    headers: dict[str, str]


class HttpSettingsProvider(Protocol):
    def __call__(self) -> HttpSettings:
        ...


def fixed_url_with_bearer_token(url: str, key: str) -> HttpSettingsProvider:
    return lambda: HttpSettings(url, {"Authorization": f"Bearer {key}"})
