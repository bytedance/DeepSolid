# Copyright 2020 DeepMind Technologies Limited.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file may have been modified by Bytedance Inc. (“Bytedance Modifications”).
# All Bytedance Modifications are Copyright 2022 Bytedance Inc.
"""Module for anything that an end user would use."""

from DeepSolid.utils.kfac_ferminet_alpha.loss_functions import register_normal_predictive_distribution
from DeepSolid.utils.kfac_ferminet_alpha.loss_functions import register_squared_error_loss
from DeepSolid.utils.kfac_ferminet_alpha.optimizer import Optimizer
