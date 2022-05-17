# Copyright 2020, 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# enable x64 on jax
# must be done at startup.

# This file may have been modified by Bytedance Inc. (“Bytedance Modifications”).
# All Bytedance Modifications are Copyright 2022 Bytedance Inc.

import jax
import jax.numpy as jnp
from jax.tree_util import (
    tree_map,
    tree_multimap,
)

def vjp_rc(
    fun, *primals, has_aux: bool = False, conjugate: bool = False):
    '''
    realize the vjp of R->C function
    :param fun:
    :param primals:
    :param has_aux:
    :param conjugate:
    :return:
    '''
    if has_aux:

        def real_fun(*primals):
            val, aux = fun(*primals)
            real_val = jax.tree_map(lambda x:x.real, val)
            return real_val, aux

        def imag_fun(*primals):
            val, aux = fun(*primals)
            imag_val = jax.tree_map(lambda x: x.imag, val)
            return imag_val, aux

        vals_r, vjp_r_fun, aux = jax.vjp(real_fun, *primals, has_aux=True)
        vals_j, vjp_j_fun, _ = jax.vjp(imag_fun, *primals, has_aux=True)

    else:
        real_fun = lambda *primals: fun(*primals).real
        imag_fun = lambda *primals: fun(*primals).imag

        vals_r, vjp_r_fun = jax.vjp(real_fun, *primals, has_aux=False)
        vals_j, vjp_j_fun = jax.vjp(imag_fun, *primals, has_aux=False)

    primals_out = jax.tree_multimap(lambda x,y:x + 1j*y, vals_r, vals_j)

    def vjp_fun(ȳ):
        """
        function computing the vjp product for a R->C function.
        """
        ȳ_r = jax.tree_map(lambda x:x.real, ȳ)
        # ȳ_r = jax.tree_map(lambda x:jnp.asarray(x, dtype=vals_r.dtype), ȳ_r)
        ȳ_j = jax.tree_map(lambda x:x.imag, ȳ)
        # ȳ_j = jax.tree_map(lambda x:jnp.asarray(x, dtype=vals_j.dtype), ȳ_j)

        # val = vals_r + vals_j
        vr_jr = vjp_r_fun(jax.tree_map(lambda x,v:jnp.asarray(x, dtype=v.dtype), ȳ_r, vals_r))
        vj_jr = vjp_r_fun(jax.tree_map(lambda x,v:jnp.asarray(x, dtype=v.dtype), ȳ_j, vals_r))
        vr_jj = vjp_j_fun(jax.tree_map(lambda x,v:jnp.asarray(x, dtype=v.dtype), ȳ_r, vals_j))
        vj_jj = vjp_j_fun(jax.tree_map(lambda x,v:jnp.asarray(x, dtype=v.dtype), ȳ_j, vals_j))

        r = tree_multimap(
            lambda re, im: re + 1j * im,
            vr_jr,
            vj_jr,
        )
        i = tree_multimap(lambda re, im: re + 1j * im, vr_jj, vj_jj)
        out = tree_multimap(lambda re, im: re + 1j * im, r, i)

        if conjugate:
            out = tree_map(jnp.conjugate, out)

        return out

    if has_aux:
        return primals_out, vjp_fun, aux
    else:
        return primals_out, vjp_fun