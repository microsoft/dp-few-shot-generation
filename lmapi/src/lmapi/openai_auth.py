# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from lmapi.http_settings import (
    HttpSettings,
    HttpSettingsProvider,
    fixed_url_with_bearer_token,
)

openai_with_api_key = fixed_url_with_bearer_token


def aoai_with_api_key(url: str, key: str) -> HttpSettingsProvider:
    return lambda: HttpSettings(url, {"api-key": key})
