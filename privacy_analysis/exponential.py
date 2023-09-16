# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from prv_accountant.dpsgd import DPSGDAccountant
import numpy as np
from prv_accountant.privacy_random_variables import PureDPMechanism
from prv_accountant import PRVAccountant
import math

###assumming sensitivity as 1.
dataset = "DBPedia"
assert dataset in ["DBPedia"]
sigma_list = []

if dataset == "DBPedia":
    full_train_num = 40000
    n = 80
    max_len_token = 100
    sigma_list = [2.73, 3.34, 3.95, 4.57]


sample_rate = n / full_train_num
print(n, full_train_num)
print("sample rate", sample_rate)
print("steps", max_len_token)

for sigma in sigma_list:
    sigma_bar = math.log(1+sample_rate*(math.exp(sigma)-1))
    print(sigma_bar)
    prv_0 = PureDPMechanism(sigma_bar)

    accountant = PRVAccountant(
        prvs=[prv_0, ],
        max_self_compositions=[max_len_token+1],
        eps_error=0.01,
        delta_error=1e-10
    )
    eps_low, eps_est, eps_up = accountant.compute_epsilon(delta=1/full_train_num, num_self_compositions=[max_len_token])

    print(sigma, eps_low, eps_est, eps_up)
