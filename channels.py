import quimb as qu
import numpy as np

from scipy.linalg import expm
from numpy.polynomial.hermite import hermgauss
from tqdm import tqdm

op_kws = {'sparse': True, 'stype': 'coo'}
ikron_kws = {'sparse': True, 'stype': 'coo',
             'coo_build': True, 'ownership': None}

def f_gaussian(t, sigma):
    """
    filter function f(t) in the interaction term
    """
    prefactor = 1 / ((2 * np.pi)**(1/4) * sigma**0.5)
    exponent = -t**2 / (4 * sigma**2)
    return prefactor * np.exp(exponent)


def _expm_from_eigh(Q, w, factor):
    """
    Build exp(factor * Q diag(w) Q†) efficiently as (Q * exp(factor*w)) @ Q†
    without forming the diagonal explicitly. Q: (n,n), w: (n,), factor: scalar
    """
    phases = np.exp(factor * w)              # (n,)
    return (Q * phases) @ Q.conj().T         # scale columns of Q, then matmul

def _sym_herm(A):
    return 0.5 * (A + A.conj().T)


def construct_unified_channel(Hamsys_base, As_ops, alpha=0.05, beta=1, sigma=5, dTime=0.1):
    """
    Construct the averaged quantum channel.

    Randomness sources:
    1. Random bath frequency ω ~ Gaussin
    2. Random choice of system operator (2 choices) in the interaction term
    3. Random sign (±1)
    """
    print(f"Constructing unified channel " f"( β = {beta})...")

    # System parameters and Hamiltonian
    N = Hamsys_base.shape[0].bit_length() - 1
    dims = [2] * (N+1)
    d_sys = 2**N
    Hamsys = qu.kron(Hamsys_base, qu.eye(2)).toarray()

    # Bath interaction operator
    B_op = (qu.spin_operator('x', **op_kws) + 1j*qu.spin_operator('y', **op_kws))
    Z_bath = qu.spin_operator("z", **op_kws).toarray()
    Hambath_base = np.kron(np.eye(d_sys), Z_bath)

    # Pulse parameters
    Ss = 5 * sigma
    nsub = 2*round(Ss/dTime) + 1
    tgrid = (np.arange(nsub) * dTime) - Ss
    dt_half = -1j * dTime / 2.0
    dt_full = -1j * dTime

    # Precompute Gaussian pulse values
    fvals = np.asarray([f_gaussian(t, sigma) for t in tgrid], dtype=np.float64)
    amps = fvals * alpha

    #  bath frequency distribution:  truncated Gaussian
    if beta == np.inf:
        BB = 5
        n_omega_samples = BB * 10
        omega_samples = np.linspace(0, BB, n_omega_samples)
        omega_weights = 1.0 / n_omega_samples * np.ones(len(omega_samples), dtype=np.float64)
    else:
        a_freq = 2 - beta**2 / (4 * sigma**2)
        omega_mean = -1 / beta
        omega_std = np.sqrt(a_freq) / beta
        n_omega_samples = 201
        omega_cutoff_std = 5.0
        omega_min = omega_mean - omega_cutoff_std * omega_std
        omega_max = omega_mean + omega_cutoff_std * omega_std
        omega_samples = np.linspace(omega_min, omega_max, n_omega_samples)
        Z = (1 / beta) * np.sqrt(2 * np.pi * a_freq)
        g_vals = (1 / Z) * np.exp(-((beta * omega_samples + 1)**2) / (2 * a_freq))
        domega = omega_samples[1] - omega_samples[0]
        omega_weights = g_vals * domega
        omega_weights = omega_weights / np.sum(omega_weights)   


    # Initialize channel matrix
    channel_matrix = np.zeros((d_sys**2, d_sys**2), dtype=np.complex128)

    # Loop over all random configurations
    for omega, w_omega in tqdm(zip(omega_samples, omega_weights),
                            total=len(omega_samples),
                            desc="Bath frequencies"):
        # Bath setup for this frequency
        Hambath_local = -omega * Z_bath 
        Hambath = -omega * Hambath_base 
        if beta == np.inf:
            rhobath = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        else:
            rhobath = expm(-beta * Hambath_local)
            rhobath = rhobath / np.trace(rhobath)

        # Total Hamiltonian
        H0 = _sym_herm(Hamsys + Hambath)
        w0, Q0 = np.linalg.eigh(H0)
        U0_half = _expm_from_eigh(Q0, w0, factor=dt_half)

        for A in As_ops:
            interact_base = qu.kron(A, B_op, parallel=False).toarray()
            interact_base = _sym_herm(interact_base)
            wv_base, Qv = np.linalg.eigh(interact_base)
            for sign in [1, -1]:
                wv = sign * wv_base

                # Time evolution
                U = np.eye(2**(N+1), dtype=np.complex128)
                for amp in amps:
                    Uv = _expm_from_eigh(Qv, wv, factor=dt_full * amp)
                    dU = U0_half @ Uv @ U0_half
                    U = U @ dU

                # Build channel map
                for i in range(d_sys):
                    for j in range(d_sys):
                        rho_in_sys = np.zeros((d_sys, d_sys), dtype=np.complex128)
                        rho_in_sys[i, j] = 1.0
                        rho_in_total = np.kron(rho_in_sys, rhobath)

                        rho_evolved = U @ rho_in_total @ U.conj().T
                        rho_out_sys = qu.ptr(rho_evolved, dims, list(range(N)))

                        input_idx = i * d_sys + j
                        output_vec = rho_out_sys.flatten()
                        channel_matrix[:, input_idx] += output_vec * w_omega / (len(As_ops) * 2)

    return channel_matrix

def construct_unified_channel_ground(Hamsys_base, As_ops, alpha=0.05, sigma=5, BB = 2,  dTime=0.1):
    """
    Construct the averaged quantum channel.

    Randomness sources:
    1. Random bath frequency ω ~ Gaussin
    2. Random choice of system operator (2 choices) in the interaction term
    3. Random sign (±1)
    """
    print(f"Constructing unified channel " f"( β = {beta})...")

    # System parameters and Hamiltonian
    N = Hamsys_base.shape[0].bit_length() - 1
    dims = [2] * (N+1)
    d_sys = 2**N
    Hamsys = qu.kron(Hamsys_base, qu.eye(2)).toarray()

    # Bath interaction operator
    B_op = (qu.spin_operator('x', **op_kws) + 1j*qu.spin_operator('y', **op_kws))
    Z_bath = qu.spin_operator("z", **op_kws).toarray()
    Hambath_base = np.kron(np.eye(d_sys), Z_bath)

    # Pulse parameters
    Ss = 5 * sigma
    nsub = 2*round(Ss/dTime) + 1
    tgrid = (np.arange(nsub) * dTime) - Ss
    dt_half = -1j * dTime / 2.0
    dt_full = -1j * dTime

    # Precompute Gaussian pulse values
    fvals = np.asarray([f_gaussian(t, sigma) for t in tgrid], dtype=np.float64)
    amps = fvals * alpha

    #  bath frequency distribution
    n_omega_samples = BB * 10
    omega_samples = np.linspace(0, BB, n_omega_samples)
    w_omega = 1.0 / n_omega_samples
    # a_freq = 2 - beta**2 / (4 * sigma**2)
    # omega_mean = -1 / beta
    # omega_std = np.sqrt(a_freq) / beta
    # n_omega_samples = 201
    # omega_cutoff_std = 5.0
    # omega_min = omega_mean - omega_cutoff_std * omega_std
    # omega_max = omega_mean + omega_cutoff_std * omega_std
    # omega_samples = np.linspace(omega_min, omega_max, n_omega_samples)

    # Z = (1 / beta) * np.sqrt(2 * np.pi * a_freq)
    # g_vals = (1 / Z) * np.exp(-((beta * omega_samples + 1)**2) / (2 * a_freq))

    # domega = omega_samples[1] - omega_samples[0]
    # omega_weights = g_vals * domega
    # omega_weights = omega_weights / np.sum(omega_weights)   


    # Initialize channel matrix
    channel_matrix = np.zeros((d_sys**2, d_sys**2), dtype=np.complex128)

    # Loop over all random configurations
    for omega in tqdm(zip(omega_samples),
                            total=len(omega_samples),
                            desc="Bath frequencies"):
        # Bath setup for this frequency
        Hambath_local = -omega * Z_bath 
        Hambath = -omega * Hambath_base 
        if beta == np.inf:
            rhobath = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        else:
            rhobath = expm(-beta * Hambath_local)
            rhobath = rhobath / np.trace(rhobath)

        # Total Hamiltonian
        H0 = _sym_herm(Hamsys + Hambath)
        w0, Q0 = np.linalg.eigh(H0)
        U0_half = _expm_from_eigh(Q0, w0, factor=dt_half)

        for A in As_ops:
            interact_base = qu.kron(A, B_op, parallel=False).toarray()
            interact_base = _sym_herm(interact_base)
            wv_base, Qv = np.linalg.eigh(interact_base)
            for sign in [1, -1]:
                wv = sign * wv_base

                # Time evolution
                U = np.eye(2**(N+1), dtype=np.complex128)
                for amp in amps:
                    Uv = _expm_from_eigh(Qv, wv, factor=dt_full * amp)
                    dU = U0_half @ Uv @ U0_half
                    U = U @ dU

                # Build channel map
                for i in range(d_sys):
                    for j in range(d_sys):
                        rho_in_sys = np.zeros((d_sys, d_sys), dtype=np.complex128)
                        rho_in_sys[i, j] = 1.0
                        rho_in_total = np.kron(rho_in_sys, rhobath)

                        rho_evolved = U @ rho_in_total @ U.conj().T
                        rho_out_sys = qu.ptr(rho_evolved, dims, list(range(N)))

                        input_idx = i * d_sys + j
                        output_vec = rho_out_sys.flatten()
                        channel_matrix[:, input_idx] += output_vec * w_omega / (len(As_ops) * 2)

    return channel_matrix

def analyze_fixed_point(channel, protocol_name, target_gibbs_state):
    """Analyze the fixed point of a quantum channel and compute spectral gap"""
    print(f"\n=== {protocol_name} Fixed Point Analysis ===")
    d_sys = target_gibbs_state.shape[0]
    # Find fixed point (eigenvalue closest to 1)
    eigenvals, eigenvecs = np.linalg.eig(channel)
    eigenvals_abs = np.abs(eigenvals)
    idx_fixed = np.argmin(np.abs(eigenvals - 1.0))
    eigenval_fixed = eigenvals[idx_fixed]
    eigenvec_fixed = eigenvecs[:, idx_fixed]
    # Reshape to density matrix and normalize
    rho_fixed = eigenvec_fixed.reshape(d_sys, d_sys)
    rho_fixed = 0.5 * (rho_fixed + rho_fixed.conj().T)  # Make Hermitian
    if np.allclose(rho_fixed.imag, 0, atol=1e-12):
        rho_fixed = rho_fixed.real
    rho_fixed = rho_fixed / np.trace(rho_fixed)  # Normalize

    # Check validity
    eigenvals_fixed = np.sort(np.linalg.eigvals(rho_fixed))
    is_positive = np.all(eigenvals_fixed >= -1e-10)
    if not is_positive:
        print("Warning: Fixed point density matrix is not positive semidefinite!")

    if np.abs(eigenval_fixed-1) > 1e-12:
        print(f"Warning: Fixed point eigenvalue deviates from 1 by {np.abs(eigenval_fixed-1):.2e}")

    # Compute spectral gap 
    eigenvals_sorted = np.sort(eigenvals_abs)[::-1]  
    spectral_gap = eigenvals_sorted[0] - eigenvals_sorted[1]

    ## for thermal state
    fidelity = min(qu.fidelity(rho_fixed, target_gibbs_state), 1)

    return rho_fixed, fidelity, spectral_gap

def random_density_orthogonal_to(sigma, tol=1e-12, approx=False, rng=None):
    """
    Generate a random density matrix with minimal overlap with a target PSD matrix sigma.

    If sigma has a nontrivial nullspace, the function samples a random density
    matrix supported in ker(sigma), so that Tr(rho sigma) = 0.

    If sigma is full rank, such a rho does not exist. In that case:
        - if approx=False, raise ValueError;
        - if approx=True, return the pure state on the eigenvector of sigma
          with the smallest eigenvalue, which minimizes Tr(rho sigma).
    """
    rng = np.random.default_rng() if rng is None else rng
    # Hermitian eigen-decomposition
    w, U = np.linalg.eigh(sigma)
    null_mask = w <= tol
    m = int(null_mask.sum())

    if m == 0:
        if not approx:
            raise ValueError("sigma is full-rank ⇒ no PSD, trace-1 rho can satisfy Tr(rho @ sigma)=0. "
                             "Use approx=True to get the minimizer.")
        # Best you can do: projector onto eigenvector of smallest eigenvalue
        v = U[:, np.argmin(w)]
        rho = np.outer(v, v.conj())  # |v><v|
        # sanity: this achieves Tr(rho @ sigma) = lambda_min
        return rho

    # Nullspace basis U0 (columns span ker(sigma))
    U0 = U[:, null_mask]                      # shape (d, m)
    # Sample Ginibre on the nullspace and make a random density there
    X = (rng.standard_normal((m, m)) + 1j*rng.standard_normal((m, m))) / np.sqrt(2)
    A = X @ X.conj().T                        # PSD on ker(sigma)
    rho0 = A / np.trace(A)                    # trace-1 on ker(sigma)
    # Embed back to the full space
    rho = U0 @ rho0 @ U0.conj().T

    return rho

def validate_channel(channel, d_sys, tol=1e-10):
    """
    A simple numerical debugging check for the constructed channel.
    Validate the two necessary properties:
    1. Trace preservation: tr(Phi(rho)) = 1;
    2. Positivity: Phi(rho) should remain positive semidefinite.
    """
    
    test_rho = np.random.randn(d_sys, d_sys) + 1j * np.random.randn(d_sys, d_sys)
    test_rho = test_rho @ test_rho.conj().T
    test_rho /= np.trace(test_rho)

    rho_out = (channel @ test_rho.flatten()).reshape(d_sys, d_sys)
    trace_error = abs(np.trace(rho_out) - 1.0)

    rho_out = 0.5 * (rho_out + rho_out.conj().T)
    evals = np.linalg.eigvalsh(rho_out)
    is_positive = np.all(evals >= -tol)

    return trace_error, is_positive, np.min(evals)

def trace_out_last_qubit(state_vec):
    dim = state_vec.shape[0]
    n_qubits = int(np.log2(dim))
    psi = state_vec.reshape(2**(n_qubits - 1), 2)

    probs = np.sum(np.abs(psi)**2, axis=0)
    probs = probs / np.sum(probs)

    idx = np.random.choice(2, p=probs)
    return psi[:, idx] / np.sqrt(probs[idx])

def suzuki_trotter_vbasis(phi0, fvals, alpha, dTime, Qv, wv, U0_half):
    """
    2nd-order ST: exp[-i(H0 + amp*H_I) dt] ≈ U0_half · U_I(amp) · U0_half,
    but do *all* steps in the H_I-eigenbasis to avoid per-step Qv transforms.
    """
    QvH = Qv.conj().T

    # move everything into V-basis once
    phi_e = QvH @ phi0                  # state in V-basis
    U0_half_e = QvH @ U0_half @ Qv      # H0 half-step in V-basis
    U0_full_e = U0_half_e @ U0_half_e   # H0 full-step in V-basis

    # first half-step once
    phi_e = U0_half_e @ phi_e

    # precompute wv * (-i*dTime) to speed elementwise exp
    scale = (-1j * dTime) * wv

    nsub = len(fvals)
    for k in range(nsub):
        amp = fvals[k] * alpha
        # Uv(amp) is diagonal in V-basis → elementwise multiply
        np.multiply(phi_e, np.exp(amp * scale), out=phi_e)

        if k < nsub - 1:
            # fuse interior halves → one full H0 step (in V-basis)
            phi_e = U0_full_e @ phi_e

    # final half-step once
    phi_e = U0_half_e @ phi_e

    # transform back once
    phi = Qv @ phi_e
    return phi