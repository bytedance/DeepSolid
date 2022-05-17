# Copyright 2020 DeepMind Technologies Limited.
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

import functools
import jax
import jax.numpy as jnp
from jax import core
from typing import TypeVar

T = TypeVar("T")

PMAP_AXIS_NAME = 'qmc_pmap_axis'

pmap = functools.partial(jax.pmap, axis_name=PMAP_AXIS_NAME)
broadcast_all_local_devices = jax.pmap(lambda x: x)
p_split = jax.pmap(lambda key: tuple(jax.random.split(key)))


def wrap_if_pmap(p_func):
    def p_func_if_pmap(obj, axis_name):
        try:
            core.axis_frame(axis_name)
            return p_func(obj, axis_name)
        except NameError:
            return obj

    return p_func_if_pmap


pmean_if_pmap = wrap_if_pmap(jax.lax.pmean)
psum_if_pmap = wrap_if_pmap(jax.lax.psum)


def replicate_all_local_devices(obj: T) -> T:
    n = jax.local_device_count()
    obj_stacked = jax.tree_map(lambda x: jnp.stack([x] * n, axis=0), obj)
    return broadcast_all_local_devices(obj_stacked)


def make_different_rng_key_on_all_devices(rng: jnp.ndarray) -> jnp.ndarray:
    rng = jax.random.fold_in(rng, jax.host_id())
    rng = jax.random.split(rng, jax.local_device_count())
    return broadcast_all_local_devices(rng)
