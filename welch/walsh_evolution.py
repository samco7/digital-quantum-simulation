from walsh_transform import *
import numpy as np
from scipy.fft import fft, ifft
from tqdm import tqdm
from sympy.combinatorics import GrayCode
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
from scipy.interpolate import RegularGridInterpolator

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import state_fidelity
from qiskit import Aer
from qiskit.tools.visualization import plot_histogram, plot_state_city
from qiskit.circuit.library import QFT
from qiskit.extensions import Initialize
from qiskit.quantum_info import Statevector


def walsh_evolve_classical(n, K, terms_kept=None, ord=2):
    # Constants
    hbar = 1.0  # Reduced Planck's constant
    m = 1.0     # Mass of the particle

    # Problem setup
    T = .6
    N = 2**n
    x0 = -3.0  # Initial position
    p0 = 15.0  # Initial momentum
    sigma = 0.5
    L = 5    # Spatial domain [-L, L]

    def initial_wave_function(x, x0, sigma, p0):
        return np.exp(-(x - x0)**2 / (2 * sigma**2) + 1j*p0*(x - x0))

    potential = lambda x: 100/np.cosh(x/.5)

    x_grid = np.linspace(-L, L, N)
    t_grid = np.linspace(0, T, K)
    dx, dt = x_grid[1] - x_grid[0], t_grid[1] - t_grid[0]

    # Initialize the initial wave function
    desired_vector = initial_wave_function(x_grid, x0, sigma, p0)
    psi = desired_vector/np.linalg.norm(desired_vector)

    a = wft(potential, n, x_grid)
    potential_walsh = iwft(a, n, terms_kept=terms_kept)

    amplitudes = []
    progress = tqdm(total=K, desc='working on time evolution')
    for i in range(K):
        # Propogate potential
        if ord == 1:
            psi = psi*np.exp(-1j*potential_walsh*dt/hbar)
        elif ord == 2:
            psi = psi*np.exp(-1j*potential_walsh*dt/2/hbar)
        else: raise(ValueError('Only first and second order supported.'))

        # Propogate kinetic
        psi = fft(psi)
        p = 2 * np.pi * np.fft.fftfreq(N, d=dx)  # Momentum grid
        psi = psi*np.exp(-1j*dt*hbar*p**2/(2*m))
        psi = ifft(psi)

        # Another half step of potential if second order
        if ord == 2:
            psi = psi*np.exp(-1j*potential_walsh*dt/2/hbar)

        amplitudes.append(np.abs(psi)**2)
        progress.update(1)
    progress.close()
    amplitudes = np.array(amplitudes)/dx
    return amplitudes, t_grid, x_grid


# Function that gives the partitioned sets by the most significant non-zero bit (MSB) from the number of qubits
def gray_partitions(n):
    # List the Gray code
    g = GrayCode(n)
    gray_list = list(g.generate_gray())[1:]

    # Create a dictionary collecting the lists
    partitions = [[] for _ in range(n)]

    # Figure out the MSB and arrange the partiton
    for entry in gray_list:
        index = entry.find('1')
        partitions[n - 1 - index].append(entry)
    return partitions


# Function that gives the position of the targeted part of the CNOT
def get_control(bitstring1, bitstring2, n):
    xor = ''.join([str(int(bit1) ^ int(bit2)) for bit1, bit2 in zip(bitstring1, bitstring2)])
    return n - 1 - xor.find('1')


# Function that implements the unitary diagonals
def unitary_circuit(f, n, dt, x_grid, terms_kept=None):
    circ = QuantumCircuit(n)
    a = wft(f, n, x_grid)
    a_kept = np.copy(a)
    if terms_kept is not None:
        sorted_indices = np.argsort(np.abs(a))[::-1]
        dropped_indices = sorted_indices[terms_kept:]
        a_kept[dropped_indices] = 0
    partitions = gray_partitions(n)
    for partition, target in zip(partitions, range(n)):
        if len(partition) == 1:
            index = eval('0b' + partition[0])
            theta = a_kept[index]
            if np.abs(theta) > 0:
                circ.rz(2*theta*dt, target)
            continue
        for i in range(len(partition)):
            index = eval('0b' + partition[i])
            theta = a_kept[index]
            control = get_control(partition[i - 1], partition[i], n)
            circ.cnot(control, target)
            if np.abs(theta) > 0:
                circ.rz(2*theta*dt, target)
    circ = transpile(circ, optimization_level=3)
    return circ


def trotter_step(n_q, dx, dt):
    """Apply the kinetic term of a single iteration of the Zalka-Wiesner algorithm.
    Args:
        n_q: number of qubits that define the grid.
        d: limits of the grid, i.e., x is defined in [-d, d).
        dt: duration of each discrete time step.

    Returns:
        qc: quantum circuit right after this step.
    """
    qc = QuantumCircuit(n_q)
    N = 2**n_q

    p_vals = (2 * np.pi * np.fft.fftfreq(N, d=dx))[[2**k for k in range(n_q)]]
    p_sum = sum(p_vals)

    for j in range(n_q):
        alpha_j = -(dt / 2) * p_vals[j] ** 2 - (dt / 2) * p_vals[j] * (p_sum - p_vals[j])
        qc.rz(alpha_j, j)

    for j in range(n_q):
        for l in range(j + 1, n_q):
                gamma_jl = (dt / 2) * p_vals[j] * p_vals[l]
                qc.cx(j, l)
                qc.rz(gamma_jl, l)
                qc.cx(j, l)
    return qc


def walsh_evolve_quantum(n, K, terms_kept=None, ord=2):
    # Constants
    hbar = 1.0  # Reduced Planck's constant
    m = 1.0     # Mass of the particle

    # Problem setup
    T = .6
    N = 2**n
    x0 = -3.0  # Initial position
    p0 = 15.0  # Initial momentum
    sigma = 0.5
    L = 5    # Spatial domain [-L, L]

    def initial_wave_function(x, x0, sigma, p0):
        return np.exp(-(x - x0)**2 / (2*sigma**2) + 1j*p0*(x - x0))

    potential = lambda x: 100/np.cosh(x/.5)

    x_grid = np.linspace(-L, L, N)
    t_grid = np.linspace(0, T, K)
    dx, dt = x_grid[1] - x_grid[0], t_grid[1] - t_grid[0]

    # Initializing the quantum circuit
    qc = QuantumCircuit(n)

    # Initialize the initial wave function
    desired_vector = initial_wave_function(x_grid, x0, sigma, p0)
    wave_func = desired_vector/np.linalg.norm(desired_vector)

    # Initialize(params = wave_func)
    qc.prepare_state(state=wave_func)

    if ord == 1:
        potential_step = unitary_circuit(potential, n, dt, x_grid, terms_kept=terms_kept)
    elif ord == 2:
        potential_step = unitary_circuit(potential, n, dt/2, x_grid, terms_kept=terms_kept)
    else: raise ValueError('Only first and second order supported.')

    kinetic_step = trotter_step(n, dx, dt)

    iqft = QFT(num_qubits=n, inverse=True).to_gate()
    qft = QFT(num_qubits=n).to_gate()

    states = []
    progress = tqdm(total=K, desc='working on time evolution')
    for i in range(K):
        # Gives the potential energy
        qc.append(potential_step, qargs=[i for i in range(n)][::-1])

        # Apply the inverse QFT
        qc.append(iqft, qargs=[i for i in range(n)])

        # Gives the kinetic energy
        qc.append(kinetic_step, qargs=[i for i in range(n)])

        # Apply the QFT
        qc.append(qft, qargs=[i for i in range(n)])

        # Another half step of potential if second order
        if ord == 2:
            qc.append(potential_step, qargs=[i for i in range(n)][::-1])

        progress.update(1)
        states.append(Statevector.from_instruction(qc))
    progress.close()
    amplitudes = np.abs(np.array(states))**2/dx
    return amplitudes, t_grid, x_grid


def plot_time_evolution(amplitudes, t_grid, x_grid, interpolate_plot=True):
    T, X = np.meshgrid(t_grid, x_grid)
    C = amplitudes.T

    if interpolate_plot:
        grid_size = 1000
        t_interp, x_interp = np.linspace(T[0, 0], T[0, -1], grid_size), np.linspace(X[0, 0], X[-1, 0], grid_size)
        T_new, X_new = np.meshgrid(t_interp, x_interp)
        interp = RegularGridInterpolator((T[0, :], X[:, 0]), C.T, method='linear')
        plt.pcolor(T_new, -X_new, interp((T_new, X_new)), cmap='magma')
        plt.show()
    else:
        plt.pcolor(T, -X, C, cmap='magma')
        plt.yticks([-5, 0, 5], [5, 0, -5])
        plt.show()
