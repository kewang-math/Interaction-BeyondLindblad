import quimb as qu
import numpy as np

import scipy.sparse as sp
from scipy.linalg import expm
from numpy.polynomial.hermite import hermgauss
from tqdm import tqdm

op_kws = {'sparse': True, 'stype': 'coo'}
ikron_kws = {'sparse': True, 'stype': 'coo',
             'coo_build': True, 'ownership': None}

# Construct Hamiltonians
def hubbard_spinful_fermi(L, t=1.0, U=4.0, mu=0.0, Delta=0.0,
                    op_kws=None, ikron_kws=None):
    """
    construct Hamiltonian of 1D spinful Fermi–Hubbard Hamiltonian with open boundaries and optional
    particle-nonconserving pairing terms.
    """
    if op_kws is None:
        op_kws = {'sparse': True, 'stype': 'coo'}
    if ikron_kws is None:
        ikron_kws = {'sparse': True, 'stype': 'coo', 'coo_build': True}

    # local ops
    c     = qu.gen.operators.destroy(2, **op_kws)
    cdag  = qu.gen.operators.create(2, **op_kws)
    n     = qu.gen.operators.num(2, **op_kws)
    # I2    = scipy.sparse.eye(2, format='coo')
    I2    = sp.eye(2, format='coo')
    P     = I2 - 2 * n  # parity = (-1)^n

    Nm   = 2 * L
    dims = [2] * Nm
    H = sp.coo_matrix((2**Nm, 2**Nm), dtype=complex)

    iu  = lambda j: 2 * j
    idn = lambda j: 2 * j + 1

    # hopping (preserves particle number)
    for j in range(L - 1):
        # ↑ hop: u_j <-> u_{j+1} crosses d_j
        H += qu.ikron([-t * cdag, P, c], dims, (iu(j), idn(j), iu(j+1)), **ikron_kws)
        H += qu.ikron([-t * c,    P, cdag], dims, (iu(j), idn(j), iu(j+1)), **ikron_kws)

        # ↓ hop: d_j <-> d_{j+1} crosses u_{j+1}
        H += qu.ikron([-t * cdag, P, c], dims, (idn(j), iu(j+1), idn(j+1)), **ikron_kws)
        H += qu.ikron([-t * c,    P, cdag], dims, (idn(j), iu(j+1), idn(j+1)), **ikron_kws)

    # on-site interaction U n_u n_d
    for j in range(L):
        H += qu.ikron([U * n, n], dims, (iu(j), idn(j)), **ikron_kws)

    # chemical potential (kept since N not conserved)
        H += qu.ikron(-mu * n, dims, iu(j),  **ikron_kws)
        H += qu.ikron(-mu * n, dims, idn(j), **ikron_kws)

    # pairing term Δ (breaks particle-number conservation)
        if abs(Delta) > 1e-12:
            H += qu.ikron([Delta * cdag, cdag], dims, (iu(j), idn(j)), **ikron_kws)
            H += qu.ikron([Delta.conjugate() * c, c], dims, (iu(j), idn(j)), **ikron_kws)

    return H.tocoo()

def h_annni(L, J1=1.0, J2=0.5, g=1.0, h=0.0, cyclic=False):
    """
    Construct the 1D ANNNI Hamiltonian in quimb (sparse matrix).
    """
    sz = qu.pauli('Z')
    sx = qu.pauli('X')

    H = 0
    dims = [2] * L
    # Nearest-neighbor term: -J1 Z_i Z_{i+1}
    for i in range(L):
        j = (i + 1) % L if cyclic else i + 1
        if j >= L:
            continue
        Zi = qu.ikron(sz, dims, i)
        Zj = qu.ikron(sz, dims, j)
        H += -J1 * (Zi @ Zj)

    # Next-nearest-neighbor term: -J2 Z_i Z_{i+2}
    for i in range(L):
        j = (i + 2) % L if cyclic else i + 2
        if j >= L:
            continue
        Zi = qu.ikron(sz, dims, i)
        Zj = qu.ikron(sz, dims, j)
        H += -J2 * (Zi @ Zj)

    # Transverse field: -g X_i
    for i in range(L):
        Xi = qu.ikron(sx, dims, i)
        H += -g * Xi

    # Longitudinal field: -h Z_i
    if abs(h) > 0:
        for i in range(L):
            Zi = qu.ikron(sz, dims, i)
            H += -h * Zi

    return qu.qu(H, sparse=True)

# Construct system operators in the interaction term
def jw_fermion_ops(L, op_kws=None, ikron_kws=None):
    """
    Build global annihilation/creation operators (with Jordan–Wigner strings)
    for a spinful chain of length L with interleaved ordering:
        modes = [u0, d0, u1, d1, ..., u_{L-1}, d_{L-1}]
    Returns:
        a_list  : [c_{0,↑}, c_{0,↓}, c_{1,↑}, c_{1,↓}, ...]
        adag_list: matching creations
    """
    if op_kws is None:
        op_kws = {'sparse': True, 'stype': 'coo'}
    if ikron_kws is None:
        ikron_kws = {'sparse': True, 'stype': 'coo',
                     'coo_build': True, 'ownership': None}

    # local single-mode ops
    c_local    = qu.gen.operators.destroy(2, **op_kws)  # annihilation
    cdag_local = qu.gen.operators.create(2,  **op_kws)  # creation
    n_local    = qu.gen.operators.num(2,     **op_kws)
    # I2         = scipy.sparse.eye(2, format='coo')
    I2         = sp.eye(2, format='coo')
    P_local    = I2 - 2 * n_local                     # (-1)^n

    Nm   = 2 * L
    dims = [2] * Nm

    a_list, adag_list = [], []

    def mode_index(j, sigma):  # sigma=0: up, 1: down
        return 2*j + sigma

    for j in range(L):
        for sigma in (0, 1):
            m = mode_index(j, sigma)

            if m == 0:
                # no preceding modes -> no JW string
                a_m    = qu.ikron(c_local,    dims, m, **ikron_kws)
                adag_m = qu.ikron(cdag_local, dims, m, **ikron_kws)
            else:
                # JW string over all previous modes 0..m-1, then local op at m
                idxs = tuple(range(m)) + (m,)
                a_m    = qu.ikron([P_local]*m + [c_local],    dims, idxs, **ikron_kws)
                adag_m = qu.ikron([P_local]*m + [cdag_local], dims, idxs, **ikron_kws)

            a_list.append(a_m)
            adag_list.append(adag_m)

    return a_list, adag_list

def jw_fermion_ops_flat(L, **kwargs):
    a_list, adag_list = jw_fermion_ops(L, **kwargs)
    flat = []
    for j in range(L):
        for sigma in (0, 1):
            m = 2*j + sigma
            flat.extend([a_list[m], adag_list[m]])
    return flat

def jw_spin_ops(Nm, op_kws=None, ikron_kws=None):
    #Nm: total number of spins
    if op_kws is None:
        op_kws = {'sparse': True, 'stype': 'coo'}
    if ikron_kws is None:
        ikron_kws = {'sparse': True, 'stype': 'coo',
                     'coo_build': True, 'ownership': None}

    # local single-mode ops
    sx_local = qu.spin_operator('x', **op_kws)
    sy_local = qu.spin_operator('y', **op_kws)
    sz_local = qu.spin_operator('z', **op_kws)
    dims = [2] * Nm

    sx_list, sy_list, sz_list, flat_list = [], [], [], []

    for j in range(Nm):
        sx_m = qu.ikron(sx_local, dims, j, **ikron_kws)
        sy_m = qu.ikron(sy_local, dims, j, **ikron_kws)
        sz_m = qu.ikron(sz_local, dims, j, **ikron_kws)
        sx_list.append(sx_m)
        sy_list.append(sy_m)
        sz_list.append(sz_m)
        flat_list.extend([sx_m, sy_m, sz_m])
        
    return flat_list



