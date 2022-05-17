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

import jax
import jax.numpy as jnp
from DeepSolid import ewaldsum
from DeepSolid import network


def local_kinetic_energy(f):
    '''
    holomorphic mode, which seems dangerous since many op don't support complex number now.
    :param f:
    :return:
    '''
    def _lapl_over_f(params, x):
        ne = x.shape[-1]
        eye = jnp.eye(ne)
        grad_f = jax.grad(f, argnums=1, holomorphic=True)
        grad_f_closure = lambda y: grad_f(params, y)

        def _body_fun(i, val):
            primal, tangent = jax.jvp(grad_f_closure, (x + 0j,), (eye[i] + 0j,))
            return val + tangent[i] + primal[i] ** 2

        return -0.5 * jax.lax.fori_loop(0, ne, _body_fun, 0.0)

    return _lapl_over_f


def local_kinetic_energy_real_imag(f):
    '''
    evaluate real and imaginary part of laplacian, which is slower than holomorphic mode but is much safer.
    :param f:
    :return:
    '''
    def _lapl_over_f(params, x):
        ne = x.shape[-1]
        eye = jnp.eye(ne)
        grad_f_real = jax.grad(lambda p, y: f(p, y).real, argnums=1)
        grad_f_imag = jax.grad(lambda p, y: f(p, y).imag, argnums=1)
        grad_f_real_closure = lambda y: grad_f_real(params, y)
        grad_f_imag_closure = lambda y: grad_f_imag(params, y)

        def _body_fun(i, val):
            primal_real, tangent_real = jax.jvp(grad_f_real_closure, (x,), (eye[i],))
            primal_imag, tangent_imag = jax.jvp(grad_f_imag_closure, (x,), (eye[i],))
            kine_real = val[0] + tangent_real[i] + primal_real[i] ** 2 - primal_imag[i] ** 2
            kine_imag = val[1] + tangent_imag[i] + 2 * primal_real[i] * primal_imag[i]
            return [kine_real, kine_imag]

        result = jax.lax.fori_loop(0, ne, _body_fun, [0.0, 0.0])
        complex = [1., 1j]
        return [-0.5 * re * com for re, com in zip(result, complex)]

    return lambda p, y: _lapl_over_f(p, y)


def local_kinetic_energy_real_imag_dim_batch(f):

    def _lapl_over_f(params, x):
        ne = x.shape[-1]
        eye = jnp.eye(ne)
        grad_f_real = jax.grad(lambda p, y: f(p, y).real, argnums=1)
        grad_f_imag = jax.grad(lambda p, y: f(p, y).imag, argnums=1)
        grad_f_real_closure = lambda y: grad_f_real(params, y)
        grad_f_imag_closure = lambda y: grad_f_imag(params, y)

        def _body_fun(dummy_eye):
            primal_real, tangent_real = jax.jvp(grad_f_real_closure, (x,), (dummy_eye,))
            primal_imag, tangent_imag = jax.jvp(grad_f_imag_closure, (x,), (dummy_eye,))
            kine_real = ((tangent_real + primal_real ** 2 - primal_imag ** 2) * dummy_eye).sum()
            kine_imag = ((tangent_imag + 2 * primal_real * primal_imag) * dummy_eye).sum()
            return [kine_real, kine_imag]

        # result = jax.lax.fori_loop(0, ne, _body_fun, [0.0, 0.0])
        result = jax.vmap(_body_fun, in_axes=0)(eye)
        result = [re.sum() for re in result]
        complex = [1., 1j]
        return [-0.5 * re * com for re, com in zip(result, complex)]

    return lambda p, y: _lapl_over_f(p, y)


def local_kinetic_energy_real_imag_hessian(f):
    '''
    Use jax.hessian to evaluate laplacian, which requires huge amount of memory.
    :param f:
    :return:
    '''
    def _lapl_over_f(params, x):
        ne = x.shape[-1]
        grad_f_real = jax.grad(lambda p, y: f(p, y).real, argnums=1)
        grad_f_imag = jax.grad(lambda p, y: f(p, y).imag, argnums=1)
        hessian_f_real = jax.hessian(lambda p, y: f(p, y).real, argnums=1)
        hessian_f_imag = jax.hessian(lambda p, y: f(p, y).imag, argnums=1)
        v_grad_f_real = grad_f_real(params, x)
        v_grad_f_imag = grad_f_imag(params, x)
        real_kinetic = jnp.trace(hessian_f_real(params, x),) + jnp.sum(v_grad_f_real**2) - jnp.sum(v_grad_f_imag**2)
        imag_kinetic = jnp.trace(hessian_f_imag(params, x),) + jnp.sum(2 * v_grad_f_real * v_grad_f_imag)

        complex = [1., 1j]
        return [-0.5 * re * com for re, com in zip([real_kinetic, imag_kinetic], complex)]

    return lambda p, y: _lapl_over_f(p, y)


def local_kinetic_energy_partition(f, partition_number=3):
  '''
  Try to parallelize the evaluation of laplacian
  :param f:
  :param partition_number:
  :return:
  '''
  vjvp = jax.vmap(jax.jvp, in_axes=(None, None, 0))

  def _lapl_over_f(params, x):
    n = x.shape[0]
    eye = jnp.eye(n)
    grad_f_real = jax.grad(lambda p, y: f(p, y).real, argnums=1)
    grad_f_imag = jax.grad(lambda p, y: f(p, y).imag, argnums=1)
    grad_f_closure_real = lambda y: grad_f_real(params, y)
    grad_f_closure_imag = lambda y: grad_f_imag(params, y)

    eyes = jnp.asarray(jnp.array_split(eye, partition_number))
    def _body_fun(val, e):
        primal_real, tangent_real = vjvp(grad_f_closure_real, (x,), (e,))
        primal_imag, tangent_imag = vjvp(grad_f_closure_imag, (x,), (e,))
        return val, ([primal_real, primal_imag], [tangent_real, tangent_imag])
    _, (plist, tlist) = \
        jax.lax.scan(_body_fun, None, eyes)
    primal = [primal.reshape((-1, primal.shape[-1])) for primal in plist]
    tangent = [tangent.reshape((-1, tangent.shape[-1])) for tangent in tlist]

    real_kinetic = jnp.trace(tangent[0]) + jnp.trace(primal[0]**2).sum() - jnp.trace(primal[1]**2).sum()
    imag_kinetic = jnp.trace(tangent[1]) + jnp.trace(2 * primal[0] * primal[1]).sum()
    return [-0.5 * real_kinetic, -0.5 * 1j * imag_kinetic]

  return _lapl_over_f



def local_ewald_energy(simulation_cell):
    ewald = ewaldsum.EwaldSum(simulation_cell)
    assert jnp.allclose(simulation_cell.energy_nuc(),
                        (ewald.ion_ion + ewald.ii_const),
                        rtol=1e-8, atol=1e-5)
    ## check pyscf madelung constant agrees with DeepSolid

    def _local_ewald_energy(x):
        energy = ewald.energy(x)
        return sum(energy)

    return _local_ewald_energy


def local_energy(f, simulation_cell):
    ke = local_kinetic_energy(f)
    ew = local_ewald_energy(simulation_cell)

    def _local_energy(params, x):
        kinetic = ke(params, x)
        ewald = ew(x)
        return kinetic + ewald

    return _local_energy


def local_energy_seperate(f, simulation_cell, mode='for', partition_number=3):

    if mode == 'for':
        ke_ri = local_kinetic_energy_real_imag(f)
    elif mode == 'hessian':
        ke_ri = local_kinetic_energy_real_imag_hessian(f)
    elif mode == 'dim_batch':
        ke_ri = local_kinetic_energy_real_imag_dim_batch(f)
    elif mode == 'partition':
        ke_ri = local_kinetic_energy_partition(f, partition_number=partition_number)
    else:
        raise ValueError('Unrecognized laplacian evaluation mode.')
    ke = lambda p, y: sum(ke_ri(p, y))
    # ke = local_kinetic_energy(f)
    ew = local_ewald_energy(simulation_cell)

    def _local_energy(params, x):
        kinetic = ke(params, x)
        ewald = ew(x)
        return kinetic, ewald

    return _local_energy
