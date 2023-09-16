# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from prv_accountant.dpsgd import DPSGDAccountant
import numpy as np
from prv_accountant.privacy_random_variables import PoissonSubsampledGaussianMechanism, GaussianMechanism, LaplaceMechanism
from prv_accountant import PRVAccountant


dataset = "MIT-G"
assert dataset in ["AGNEWS", "DBPedia", "MIT-D", "MIT-G"]
sigma_list = []

if dataset == "AGNEWS":
    full_train_num = 30000
    n = 20
    max_token_cnt = 100
    sigma_list = [0.51, 0.46, 0.39, 0.31]
elif dataset == "DBPedia":
    full_train_num = 40000
    n = 80
    max_token_cnt = 100
    sigma_list = [0.63, 0.54, 0.45, 0.36]
elif dataset == "MIT-G":
    full_train_num = 2953
    n = 80
    max_token_cnt = 80
    sigma_list = [1.08, 0.81, 0.64, 0.5]
elif dataset == "MIT-D":
    full_train_num = 1561
    n = 80
    max_token_cnt = 80
    sigma_list = [1.52, 1.04, 0.77, 0.58]

sample_rate = n / full_train_num
print(dataset)
print(n, full_train_num)
print("sample rate", sample_rate)
print("steps", max_token_cnt)

for sigma in sigma_list:
    prv_0 = PoissonSubsampledGaussianMechanism(noise_multiplier=sigma, sampling_probability=sample_rate)
    print("sample rate", sample_rate)

    accountant = PRVAccountant(
        prvs=[prv_0, ],
        max_self_compositions=[1000],
        eps_error=0.01,
        delta_error=1e-10
    )
    eps_low, eps_est, eps_up = accountant.compute_epsilon(delta=1/full_train_num, num_self_compositions=[max_token_cnt])

    print(sigma, eps_low, eps_est, eps_up)
