# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import TypeVar, cast

import numpy as np
import scipy.special

T = TypeVar("T")


def log_normalize(logprobs: dict[T, float]) -> dict[T, float]:
    """Normalize a log probability distribution so that all probabilities sum to 1.

    The input is a sparse distribution represented as a dictionary mapping from token IDs to log probabilities.
    """

    normalizer = cast(float, scipy.special.logsumexp(list(logprobs.values())))
    return {k: v - normalizer for k, v in logprobs.items()}


def log_max_normalize(logprobs: dict[T, float]) -> dict[T, float]:
    """Normalize a log probability distribution so that maxium probabilities is to 1.

    The input is a sparse distribution represented as a dictionary mapping from token IDs to log probabilities.
    """

    normalizer = cast(float, max(list(logprobs.values())))
    return {k: v - normalizer for k, v in logprobs.items()}


def densify(vocab_size: int, logprobs: dict[int, float]) -> np.ndarray:
    """Convert a sparse log-probability distribution into a dense one.

    The dense distribution is represented as a 1D tensor of size `vocab_size`.
    """

    assert len(logprobs) > 0
    result = np.full((vocab_size,), -np.inf)
    for k, v in logprobs.items():
        result[k] = v
    return result


MINIMUM_MISSING_PROB = 1e-6


def remove_logit_bias(
    biased_logprobs: dict[T, float], logit_bias: dict[T, float] | dict[T, int]
) -> dict[T, float]:
    """Undo the effects of logit_bias.

    This function is useful to get log probabilities of arbitrary tokens,
    as a workaround for how OpenAI's API only returns the top K most likely tokens.
    Give those arbitary tokens a large logit_bias (about 70 works well in practice).

    For this function to work properly:
    - The logit_bias should be set so that logprobs for all tokens in the logit_bias
      are provided in biased_logprobs. For example, they should all be large positive numbers.
    - The logit_bias should not be too big, because of limited floating point precision.
      OpenAI's API will not return probabilities smaller than np.log(np.finfo(np.float32).smallest_normal.
    - biased_logprobs should have more elements than logit_bias. Otherwise it's
      not possible to recover the original log probabilities.
    """
    if not (logit_bias.keys() <= biased_logprobs.keys()):
        raise ValueError("logit_bias must be a subset of biased_logprobs")

    missing_key = object()
    missing_prob = 1 - np.exp(scipy.special.logsumexp(list(biased_logprobs.values())))
    missing_logprob = float(
        np.log(missing_prob) if missing_prob > MINIMUM_MISSING_PROB else -np.inf
    )

    debiased_logits: dict[T | object, float] = {
        k: v - logit_bias.get(k, 0) for k, v in biased_logprobs.items()
    }
    debiased_logits[missing_key] = missing_logprob

    normalized = log_normalize(debiased_logits)
    del normalized[missing_key]
    return cast(dict[T, float], normalized)
