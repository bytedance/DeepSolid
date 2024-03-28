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

"""Implementation of Fermionic Neural Network in JAX."""
import functools
from collections import namedtuple
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple, Union

from DeepSolid import curvature_tags_and_blocks
from DeepSolid import distance

import jax
import jax.numpy as jnp

_MAX_POLY_ORDER = 5  # highest polynomial used in envelopes

FermiLayers = Tuple[Tuple[int, int], ...]
# Recursive types are not yet supported in pytype - b/109648354.
# pytype: disable=not-supported-yet
ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], Mapping[Any, 'ParamTree']]
# pytype: enable=not-supported-yet
# init(key) -> params
FermiNetInit = Callable[[jnp.ndarray], ParamTree]
# network(params, x) -> sign_out, log_out
FermiNetApply = Callable[[ParamTree, jnp.ndarray], Tuple[jnp.ndarray,
                                                         jnp.ndarray]]

def enforce_pbc(latvec, epos):
    """
    Enforces periodic boundary conditions on a set of configs.

    :param latvec: orthogonal lattice vectors defining 3D torus: (3,3)
    :param epos: attempted new electron coordinates: (N_ele, 3)
    :return: final electron coordinates with PBCs imposed: (N_ele, 3)
    """

    # Writes epos in terms of (lattice vecs) fractional coordinates
    recpvecs = jnp.linalg.inv(latvec)
    epos_lvecs_coord = jnp.einsum("ij,jk->ik", epos, recpvecs)
    wrap = epos_lvecs_coord // 1
    final_epos = jnp.matmul(epos_lvecs_coord - wrap, latvec)

    return final_epos, wrap


def init_solid_fermi_net_params(
    key: jnp.ndarray,
    data,
    atoms: jnp.ndarray,
    spins: Tuple[int, int],
    envelope_type: str = 'full',
    bias_orbitals: bool = False,
    use_last_layer: bool = False,
    eps: float = 0.01,
    full_det: bool = True,
    hidden_dims: FermiLayers = ((256, 32), (256, 32), (256, 32)),
    determinants: int = 16,
    after_determinants: Union[int, Tuple[int, ...]] = 1,
    distance_type='nu',
):
    """Initializes parameters for the Fermionic Neural Network.

    Args:
      key: JAX RNG state.
      atoms: (natom, 3) array of atom positions.
      spins: Tuple of the number of spin-up and spin-down electrons.
      envelope_type: Envelope to use to impose orbitals go to zero at infinity.
        See solid_fermi_net_orbitals.
      bias_orbitals: If true, include a bias in the final linear layer to shape
        the outputs into orbitals.
      use_last_layer: If true, the outputs of the one- and two-electron streams
        are combined into permutation-equivariant features and passed into the
        final orbital-shaping layer. Otherwise, just the output of the
        one-electron stream is passed into the orbital-shaping layer.
      hf_solution: If present, initialise the parameters to match the Hartree-Fock
        solution. Otherwise a random initialisation is use.
      eps: If hf_solution is present, scale all weights and biases except the
        first layer by this factor such that they are initialised close to zero.
      full_det: If true, evaluate determinants over all electrons. Otherwise,
        block-diagonalise determinants into spin channels.
      hidden_dims: Tuple of pairs, where each pair contains the number of hidden
        units in the one-electron and two-electron stream in the corresponding
        layer of the FermiNet. The number of layers is given by the length of the
        tuple.
      determinants: Number of determinants to use.
      after_determinants: currently ignored.

    Returns:
      PyTree of network parameters.
    """
    # after_det is from the legacy QMC TF implementation. Reserving for future
    # use.
    del after_determinants
    del data

    natom = atoms.shape[0]
    if distance_type == 'nu':
        in_dims = (natom * 4, 4)
    elif distance_type == 'tri':
        in_dims = (natom * 7, 7)
    else:
        raise ValueError('Unrecognized distance function.')

    active_spin_channels = [spin for spin in spins if spin > 0]
    nchannels = len(active_spin_channels)
    # The input to layer L of the one-electron stream is from
    # construct_symmetric_features and shape (nelectrons, nfeatures), where
    # nfeatures is i) output from the previous one-electron layer; ii) the mean
    # for each spin channel from each layer; iii) the mean for each spin channel
    # from each two-electron layer. We don't create features for spin channels
    # which contain no electrons (i.e. spin-polarised systems).
    dims_one_in = (
            [(nchannels + 1) * in_dims[0] + nchannels * in_dims[1]] +
            [(nchannels + 1) * hdim[0] + nchannels * hdim[1] for hdim in hidden_dims])
    if not use_last_layer:
        dims_one_in[-1] = hidden_dims[-1][0]
    dims_one_out = [hdim[0] for hdim in hidden_dims]
    dims_two = [in_dims[1]] + [hdim[1] for hdim in hidden_dims]

    len_double = len(hidden_dims) if use_last_layer else len(hidden_dims) - 1
    params = {
        'single': [{} for _ in range(len(hidden_dims))],
        'double': [{} for _ in range(len_double)],
        'orbital': [],
        'envelope': [{} for _ in active_spin_channels],
    }

    # params['envelope'] = [{} for _ in active_spin_channels]
    for i, spin in enumerate(active_spin_channels):
        nparam = sum(spins) * determinants if full_det else spin * determinants
        params['envelope'][i]['pi'] = jnp.ones((natom, nparam))
        if envelope_type == 'isotropic':
            params['envelope'][i]['sigma'] = jnp.ones((natom, nparam))
        elif envelope_type == 'diagonal':
            params['envelope'][i]['sigma'] = jnp.ones((natom, 3, nparam))
        elif envelope_type == 'full':
            params['envelope'][i]['sigma'] = jnp.tile(
                jnp.eye(3)[..., None, None], [1, 1, natom, nparam])

    for i in range(len(hidden_dims)):
        key, subkey = jax.random.split(key)
        params['single'][i]['w'] = (jax.random.normal(
            subkey, shape=(dims_one_in[i], dims_one_out[i])) /
                                    jnp.sqrt(float(dims_one_in[i])))

        key, subkey = jax.random.split(key)
        params['single'][i]['b'] = jax.random.normal(
            subkey, shape=(dims_one_out[i],))

        if i < len_double:
            key, subkey = jax.random.split(key)
            params['double'][i]['w'] = (jax.random.normal(
                subkey, shape=(dims_two[i], dims_two[i + 1])) /
                                        jnp.sqrt(float(dims_two[i])))

            key, subkey = jax.random.split(key)
            params['double'][i]['b'] = jax.random.normal(subkey,
                                                         shape=(dims_two[i + 1],))

    for i, spin in enumerate(active_spin_channels):
        nparam = sum(spins) * determinants if full_det else spin * determinants
        key, subkey = jax.random.split(key)
        params['orbital'].append({})
        params['orbital'][i]['w'] = (jax.random.normal(
            subkey, shape=(dims_one_in[-1], 2 * nparam)) /
                                     jnp.sqrt(float(dims_one_in[-1])))
        if bias_orbitals:
            key, subkey = jax.random.split(key)
            params['orbital'][i]['b'] = jax.random.normal(
                subkey, shape=(2 * nparam,))

    return params


def scaled_f(w):
    """
    see Phys. Rev. B 94, 035157
    :param w: projection of position vectors on reciprocal vectors.
    :return: function f in the ref.
    """
    return jnp.abs(w) * (1 - jnp.abs(w / jnp.pi) ** 3 / 4.)


def scaled_g(w):
    """
    see Phys. Rev. B 94, 035157
    :param w: projection of position vectors on reciprocal vectors.
    :return: function g in the ref.
    """
    return w * (1 - 3. / 2. * jnp.abs(w / jnp.pi) + 1. / 2. * jnp.abs(w / jnp.pi) ** 2)


def nu_distance(xea, a, b):
    """
    see Phys. Rev. B 94, 035157
    :param xea: relative distance between electrons and atoms
    :param a: lattice vectors of primitive cell divided by 2\pi.
    :param b: reciprocal vectors of primitive cell.
    :return: periodic generalized relative and absolute distance of xea.
    """
    w = jnp.einsum('...ijk,lk->...ijl', xea, b)
    mod = (w + jnp.pi) // (2 * jnp.pi)
    w = (w - mod * 2 * jnp.pi)
    r1 = (jnp.linalg.norm(a, axis=-1) * scaled_f(w)) ** 2
    sg = scaled_g(w)
    rel = jnp.einsum('...i,ij->...j', sg, a)
    r2 = jnp.einsum('ij,kj->ik', a, a) * (sg[..., :, None] * sg[..., None, :])
    result = jnp.sum(r1, axis=-1) + jnp.sum(r2 * (jnp.ones(r2.shape[-2:]) - jnp.eye(r2.shape[-1])), axis=[-1, -2])
    sd = result ** 0.5
    return sd, rel


def tri_distance(xea, a, b):
    """
    see Phys. Rev. Lett. 130, 036401 (2023).
    :param xea: relative distance between electrons and atoms
    :param a: lattice vectors of primitive cell divided by 2\pi.
    :param b: reciprocal vectors of primitive cell.
    :return: periodic generalized relative and absolute distance of xea.
    """
    w = jnp.einsum('...ijk,lk->...ijl', xea, b)
    sg = jnp.sin(w)
    cg = jnp.cos(w)
    rel_sin = jnp.einsum('...i,ij->...j', sg, a)
    rel_cos = jnp.einsum('...i,ij->...j', cg, a)
    rel = jnp.concatenate([rel_sin, rel_cos], axis=-1)
    metric = jnp.einsum('ij,kj->ik', a, a)
    vector_sin = sg[..., :, None] * sg[..., None, :]
    vector_cos = (1-cg[..., :, None]) * (1-cg[..., None, :])
    vector = vector_cos + vector_sin
    sd = jnp.einsum('...ij,ij->...', vector, metric) ** 0.5
    return sd, rel


def construct_periodic_input_features(
    x: jnp.ndarray,
    atoms: jnp.ndarray,
    simulation_cell=None,
    ndim: int = 3,
    distance_type: str = 'nu',
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Constructs a periodic generalized inputs to Fermi Net from raw electron and atomic positions.
    see Phys. Rev. B 94, 035157
      Args:
        x: electron positions. Shape (nelectrons*ndim,).
        atoms: atom positions. Shape (natoms, ndim).
        ndim: dimension of system. Change only with caution.
      Returns:
        ae, ee, r_ae, r_ee tuple, where:
          ae: atom-electron vector. Shape (nelectron, natom, 3).
          ee: atom-electron vector. Shape (nelectron, nelectron, 3).
          r_ae: atom-electron distance. Shape (nelectron, natom, 1).
          r_ee: electron-electron distance. Shape (nelectron, nelectron, 1).
        The diagonal terms in r_ee are masked out such that the gradients of these
        terms are also zero.
      """
    if distance_type == 'nu':
        distance_func = nu_distance
    elif distance_type == 'tri':
        distance_func = tri_distance
    else:
        raise ValueError('Unrecognized distance function.')

    primitive_cell = simulation_cell.original_cell
    x = x.reshape(-1, ndim)
    n = x.shape[0]
    prim_x, _ = enforce_pbc(primitive_cell.a, x)

    # prim_xea = minimal_imag.dist_i(atoms.ravel(), prim_x.ravel())
    prim_xea = prim_x[..., None, :] - atoms
    prim_periodic_sea, prim_periodic_xea = distance_func(prim_xea,
                                                       primitive_cell.AV,
                                                       primitive_cell.BV)
    prim_periodic_sea = prim_periodic_sea[..., None]

    sim_x, _ = enforce_pbc(simulation_cell.a, x)
    # sim_xee = sim_minimal_imag.dist_matrix(sim_x.ravel())
    sim_xee = sim_x[:, None, :] - sim_x[None, :, :]

    sim_periodic_see, sim_periodic_xee = distance_func(sim_xee + jnp.eye(n)[..., None],
                                                     simulation_cell.AV,
                                                     simulation_cell.BV)
    sim_periodic_see = sim_periodic_see * (1.0 - jnp.eye(n))
    sim_periodic_see = sim_periodic_see[..., None]

    sim_periodic_xee = sim_periodic_xee * (1.0 - jnp.eye(n))[..., None]

    return prim_periodic_xea, sim_periodic_xee, prim_periodic_sea, sim_periodic_see


def construct_symmetric_features(h_one: jnp.ndarray, h_two: jnp.ndarray,
                                 spins: Tuple[int, int]) -> jnp.ndarray:
    """Combines intermediate features from rank-one and -two streams.
    Args:
      h_one: set of one-electron features. Shape: (nelectrons, n1), where n1 is
        the output size of the previous layer.
      h_two: set of two-electron features. Shape: (nelectrons, nelectrons, n2),
        where n2 is the output size of the previous layer.
      spins: number of spin-up and spin-down electrons.
    Returns:
      array containing the permutation-equivariant features: the input set of
      one-electron features, the mean of the one-electron features over each
      (occupied) spin channel, and the mean of the two-electron features over each
      (occupied) spin channel. Output shape (nelectrons, 3*n1 + 2*n2) if there are
      both spin-up and spin-down electrons and (nelectrons, 2*n1, n2) otherwise.
    """
    # Split features into spin up and spin down electrons
    h_ones = jnp.split(h_one, spins[0:1], axis=0)
    h_twos = jnp.split(h_two, spins[0:1], axis=0)

    # Construct inputs to next layer
    # h.size == 0 corresponds to unoccupied spin channels.
    g_one = [jnp.mean(h, axis=0, keepdims=True) for h in h_ones if h.size > 0]
    g_two = [jnp.mean(h, axis=0) for h in h_twos if h.size > 0]

    g_one = [jnp.tile(g, [h_one.shape[0], 1]) for g in g_one]

    return jnp.concatenate([h_one] + g_one + g_two, axis=1)


def isotropic_envelope(ae, params):
    """Computes an isotropic exponentially-decaying multiplicative envelope."""
    return jnp.sum(jnp.exp(-jnp.abs(ae * params['sigma'])) * params['pi'], axis=1)


def diagonal_envelope(ae, params):
    """Computes a diagonal exponentially-decaying multiplicative envelope."""
    r_ae = jnp.linalg.norm(ae[..., None] * params['sigma'], axis=2)
    return jnp.sum(jnp.exp(-r_ae) * params['pi'], axis=1)


vdot = jax.vmap(jnp.dot, (0, 0))


def apply_covariance(x, y):
    """Equivalent to jnp.einsum('ijk,kmjn->ijmn', x, y)."""
    i, _, _ = x.shape
    k, m, j, n = y.shape
    x = x.transpose((1, 0, 2))
    y = y.transpose((2, 0, 1, 3)).reshape((j, k, m * n))
    return vdot(x, y).reshape((j, i, m, n)).transpose((1, 0, 2, 3))


def full_envelope(ae, params):
    """Computes a fully anisotropic exponentially-decaying multiplicative envelope."""
    r_ae = apply_covariance(ae, params['sigma'])
    r_ae = curvature_tags_and_blocks.register_qmc1(r_ae, ae, params['sigma'],
                                                   type='full')
    r_ae = jnp.linalg.norm(r_ae, axis=2)
    return jnp.sum(jnp.exp(-r_ae) * params['pi'], axis=1)


def output_envelope(ae, params):
    """Fully anisotropic envelope, but only one output."""
    sigma = jnp.expand_dims(params['sigma'], -1)
    ae_sigma = jnp.squeeze(apply_covariance(ae, sigma), axis=-1)
    r_ae = jnp.linalg.norm(ae_sigma, axis=2)
    return jnp.sum(jnp.log(jnp.sum(jnp.exp(-r_ae + params['pi']), axis=1)))


def slogdet_op(x):
    """Computes sign and log of determinants of matrices.

    This is a jnp.linalg.slogdet with a special (fast) path for small matrices.

    Args:
      x: square matrix.

    Returns:
      sign, (natural) logarithm of the determinant of x.
    """
    if x.shape[-1] == 1:
        sign = jnp.exp(1j*jnp.angle(x[..., 0, 0]))
        logdet = jnp.log(jnp.abs(x[..., 0, 0]))
    else:
        sign, logdet = jnp.linalg.slogdet(x)

    return sign, logdet


def logdet_matmul(xs: Sequence[jnp.ndarray],
                  w: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Combines determinants and takes dot product with weights in log-domain.

    We use the log-sum-exp trick to reduce numerical instabilities.

    Args:
      xs: FermiNet orbitals in each determinant. Either of length 1 with shape
        (ndet, nelectron, nelectron) (full_det=True) or length 2 with shapes
        (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta) (full_det=False,
        determinants are factorised into block-diagonals for each spin channel).
      w: weight of each determinant. If none, a uniform weight is assumed.

    Returns:
      sum_i w_i D_i in the log domain, where w_i is the weight of D_i, the i-th
      determinant (or product of the i-th determinant in each spin channel, if
      full_det is not used).
    """
    slogdets = [slogdet_op(x) for x in xs]
    sign_in, slogdet = functools.reduce(
        lambda a, b: (a[0] * b[0], a[1] + b[1]), slogdets)
    max_idx = jnp.argmax(slogdet)
    # sign_in_max = sign_in[max_idx]
    slogdet_max = slogdet[max_idx]
    # log-sum-exp trick
    det = sign_in * jnp.exp(slogdet-slogdet_max)
    if w is None:
        result = jnp.sum(det)
    else:
        result = jnp.matmul(det, w)[0]
    sign_out = jnp.exp(1j*jnp.angle(result))
    slog_out = jnp.log(jnp.abs(result)) + slogdet_max
    return sign_out, slog_out


def linear_layer(x, w, b=None):
    """Evaluates a linear layer, x w + b.

    Args:
      x: inputs.
      w: weights.
      b: optional bias. Only x w is computed if b is None.

    Returns:
      x w + b if b is given, x w otherwise.
    """
    y = jnp.dot(x, w)
    y = y + b if b is not None else y
    return curvature_tags_and_blocks.register_repeated_dense(y, x, w, b)


vmap_linear_layer = jax.vmap(linear_layer, in_axes=(0, None, None), out_axes=0)


def eval_phase(x, klist, ndim=3, spins=None, full_det=False):
    x = x.reshape([-1, ndim])
    xs = jnp.split(x, spins[0:1], axis=-2)
    if full_det:
        klist = jnp.concatenate(klist, axis=0)
        kdot_xs = [jnp.matmul(x, klist.T) for x, ne in zip(xs, spins) if ne > 0]
    else:
        kdot_xs = [jnp.matmul(x, kpt.T) for x, kpt, ne in zip(xs, klist, spins) if ne > 0]
    phases = [jnp.exp(1j * kdot_x) for kdot_x in kdot_xs]
    return phases


def solid_fermi_net_orbitals(params, x,
                             simulation_cell=None,
                             klist=None,
                             atoms=None,
                             spins=(None, None),
                             envelope_type=None,
                             full_det=False,
                             distance_type='nu'):
    """Forward evaluation of the Solid Neural Network up to the orbitals.
     Args:
       params: A dictionary of parameters, containing fields:
         `single`: a list of dictionaries with params 'w' and 'b', weights for the
           one-electron stream of the network.
         `double`: a list of dictionaries with params 'w' and 'b', weights for the
           two-electron stream of the network.
         `orbital`: a list of two weight matrices, for spin up and spin down (no
           bias is necessary as it only adds a constant to each row, which does
           not change the determinant).
         `dets`: weight on the linear combination of determinants
         `envelope`: a dictionary with fields `sigma` and `pi`, weights for the
           multiplicative envelope.
       x: The input data, a 3N dimensional vector.
       simulation_cell: pyscf object of simulation cell.
       klist: Tuple with occupied k points of the spin up and spin down electrons
       in simulation cell.
       spins: Tuple with number of spin up and spin down electrons.
       envelope_type: a string that specifies kind of envelope. One of:
         `isotropic`: envelope is the same in every direction
       full_det: If true, the determinants are dense, rather than block-sparse.
         True by default, false is still available for backward compatibility.
         Thus, the output shape of the orbitals will be (ndet, nalpha+nbeta,
         nalpha+nbeta) if True, and (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta)
         if False.
     Returns:
       One (two matrices if full_det is False) that exchange columns under the
       exchange of inputs, and additional variables that may be needed by the
       envelope, depending on the envelope type.
     """

    ae_, ee_, r_ae, r_ee = construct_periodic_input_features(
        x, atoms, simulation_cell=simulation_cell, distance_type=distance_type
    )
    ae = jnp.concatenate((r_ae, ae_), axis=2)
    ae = jnp.reshape(ae, [jnp.shape(ae)[0], -1])
    ee = jnp.concatenate((r_ee, ee_), axis=2)

    # which variable do we pass to envelope?
    to_env = r_ae if envelope_type == 'isotropic' else ae_

    if envelope_type == 'isotropic':
        envelope = isotropic_envelope
    elif envelope_type == 'diagonal':
        envelope = diagonal_envelope
    elif envelope_type == 'full':
        envelope = full_envelope

    h_one = ae  # single-electron features
    h_two = ee  # two-electron features
    residual = lambda x, y: (x + y) / jnp.sqrt(2.0) if x.shape == y.shape else y
    for i in range(len(params['double'])):
        h_one_in = construct_symmetric_features(h_one, h_two, spins)

        # Execute next layer
        h_one_next = jnp.tanh(linear_layer(h_one_in, **params['single'][i]))
        h_two_next = jnp.tanh(vmap_linear_layer(h_two, params['double'][i]['w'],
                                                params['double'][i]['b']))
        h_one = residual(h_one, h_one_next)
        h_two = residual(h_two, h_two_next)
    if len(params['double']) != len(params['single']):
        h_one_in = construct_symmetric_features(h_one, h_two, spins)
        h_one_next = jnp.tanh(linear_layer(h_one_in, **params['single'][-1]))
        h_one = residual(h_one, h_one_next)
        h_to_orbitals = h_one
    else:
        h_to_orbitals = construct_symmetric_features(h_one, h_two, spins)
    # Note split creates arrays of size 0 for spin channels without any electrons.
    h_to_orbitals = jnp.split(h_to_orbitals, spins[0:1], axis=0)

    active_spin_channels = [spin for spin in spins if spin > 0]
    orbitals = [linear_layer(h, **p)
                for h, p in zip(h_to_orbitals, params['orbital'])]

    for i, spin in enumerate(active_spin_channels):
        nparams = params['orbital'][i]['w'].shape[-1] // 2
        orbitals[i] = orbitals[i][..., :nparams] + 1j * orbitals[i][..., nparams:]

    if envelope_type in ['isotropic', 'diagonal', 'full']:
        orbitals = [envelope(te, param) * orbital for te, orbital, param in
                    zip(jnp.split(to_env, active_spin_channels[:-1], axis=0),
                        orbitals, params['envelope'])]
    # Reshape into matrices and drop unoccupied spin channels.
    orbitals = [jnp.reshape(orbital, [spin, -1, sum(spins) if full_det else spin])
                for spin, orbital in zip(active_spin_channels, orbitals) if spin > 0]
    orbitals = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals]
    phases = eval_phase(x, klist=klist, ndim=3, spins=spins, full_det=full_det)

    orbitals = [orb * p[None, :, :] for orb, p in zip(orbitals, phases)]
    if full_det:
        orbitals = [jnp.concatenate(orbitals, axis=1)]
    return orbitals, to_env


def eval_func(params, x,
              klist=None,
              simulation_cell=None,
              atoms=None,
              spins=(None, None),
              envelope_type='full',
              full_det=False,
              distance_type='nu',
              method_name='eval_slogdet'):
    '''
    generates the wavefunction of simulation cell.
    :param params: parameter dict
    :param x: The input data, a 3N dimensional vector.
    :param simulation_cell: pyscf object of simulation cell.
    :param klist: Tuple with occupied k points of the spin up and spin down electrons
    in simulation cell.
    :param atoms: array of atom positions in the primitive cell.
    :param spins: Tuple with number of spin up and spin down electrons.
    :param full_det: specify the mode of wavefunction, spin diagonalized or not.
    :param method_name: specify the returned function of wavefunction
    :return: required wavefunction
    '''

    orbitals, to_env = solid_fermi_net_orbitals(params, x,
                                                klist=klist,
                                                simulation_cell=simulation_cell,
                                                atoms=atoms,
                                                spins=spins,
                                                envelope_type=envelope_type,
                                                distance_type=distance_type,
                                                full_det=full_det)
    if method_name == 'eval_slogdet':
        _, result = logdet_matmul(orbitals)
    elif method_name == 'eval_logdet':
        sign, slogdet = logdet_matmul(orbitals)
        result = jnp.log(sign) + slogdet
    elif method_name == 'eval_phase_and_slogdet':
        result = logdet_matmul(orbitals)
    elif method_name == 'eval_mats':
        result = orbitals
    else:
        raise ValueError('Unrecognized method name')

    return result


def make_solid_fermi_net(
    envelope_type: str = 'full',
    bias_orbitals: bool = False,
    use_last_layer: bool = False,
    klist=None,
    simulation_cell=None,
    full_det: bool = True,
    hidden_dims: FermiLayers = ((256, 32), (256, 32), (256, 32)),
    determinants: int = 16,
    after_determinants: Union[int, Tuple[int, ...]] = 1,
    distance_type='nu',
    method_name='eval_logdet',
):
    '''
    generates the wavefunction of simulation cell.
    :param envelope_type: specify envelope
    :param bias_orbitals: whether to contain bias in the last layer of orbitals
    :param use_last_layer: wheter to use two-electron feature in the last layer
    :param klist: occupied k points from HF
    :param simulation_cell: simulation cell
    :param full_det: specify the mode of wavefunction, spin diagonalized or not.
    :param hidden_dims: specify the dimension of one-electron and two-electron layer
    :param determinants: the number of determinants used
    :param after_determinants: deleted
    :param method_name: specify the returned function
    :return: a haiku like module which contain init and apply method. init is used to initialize the parameter of
    network and apply method perform the calculation.
    '''
    if method_name not in ['eval_slogdet', 'eval_logdet', 'eval_mats', 'eval_phase_and_slogdet']:
        raise ValueError('Method name is not in class dir.')

    method = namedtuple('method', ['init', 'apply'])
    init = functools.partial(
        init_solid_fermi_net_params,
        atoms=simulation_cell.original_cell.atom_coords(),
        spins=simulation_cell.nelec,
        envelope_type=envelope_type,
        bias_orbitals=bias_orbitals,
        use_last_layer=use_last_layer,
        full_det=full_det,
        hidden_dims=hidden_dims,
        determinants=determinants,
        after_determinants=after_determinants,
        distance_type=distance_type,
    )
    network = functools.partial(
        eval_func,
        simulation_cell=simulation_cell,
        klist=klist,
        atoms=simulation_cell.original_cell.atom_coords(),
        spins=simulation_cell.nelec,
        envelope_type=envelope_type,
        full_det=full_det,
        distance_type=distance_type,
        method_name=method_name,
    )
    method.init = init
    method.apply = network
    return method
