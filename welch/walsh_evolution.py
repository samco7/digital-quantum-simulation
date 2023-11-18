from walsh_transform import *
import numpy as np
from scipy.fft import fft, ifft, fftfreq
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


def walsh_evolve_1(potential, initial_wave_function, n, L, K, T, D=1/2, h_bar=1, m=1, terms_kept=None, verbose=True):
    N = 2**n
    dx, dt = 2*L/N, T/K
    x_grid = np.arange(-L, L - dx/2, dx)
    t_grid = np.arange(0, T + dt/2, dt)

    # Initialize the initial wave function
    psi = initial_wave_function(x_grid)
    a = wft(potential, n, x_grid, verbose=verbose)
    potential_walsh = iwft(a, n, terms_kept=terms_kept, verbose=verbose)

    out = True
    states = [psi]
    if verbose: progress = tqdm(total = K, desc='working on time evolution')

    for i in range(K):
        # propagate kinetic
        psi = fft(psi)
        p = 2*np.pi*fftfreq(N, d=dx)  # Momentum grid
        psi = psi*np.exp(-1j*D*dt*h_bar*p**2/m)
        psi = ifft(psi)

        # propagate potential
        psi = psi*np.exp(-1j*potential_walsh*dt/h_bar)

        if out:
            states.append(psi)
        if verbose: progress.update(1)
    if verbose: progress.close()

    states = np.array(states)
    return states, t_grid, x_grid

def walsh_evolve_2(potential, initial_wave_function, n, L, K, T, D=1/2, h_bar=1, m=1, terms_kept=None, verbose=True):
    N = 2**n
    dx, dt = 2*L/N, T/K
    x_grid = np.arange(-L, L - dx/2, dx)
    t_grid = np.arange(0, T + dt/2, dt)

    # Initialize the initial wave function
    psi = initial_wave_function(x_grid)
    a = wft(potential, n, x_grid, verbose=verbose)
    potential_walsh = iwft(a, n, terms_kept=terms_kept, verbose=verbose)

    out = True
    states = [psi]
    if verbose: progress = tqdm(total = K, desc='working on time evolution')

    # propagate potential half step to start
    psi = psi*np.exp(-1j*potential_walsh*dt/2/h_bar)

    for i in range(K - 1):
        # propagate kinetic
        psi = fft(psi)
        p = 2*np.pi*fftfreq(N, d=dx)  # Momentum grid
        psi = psi*np.exp(-1j*D*dt*h_bar*p**2/m)
        psi = ifft(psi)

        # propagate potential
        psi = psi*np.exp(-1j*potential_walsh*dt/h_bar)

        if out:
            states.append(psi*np.exp(1j*potential_walsh*dt/2/h_bar))
        if verbose: progress.update(1)

    # propagate kinetic
    psi = fft(psi)
    p = 2*np.pi*fftfreq(N, d=dx)  # Momentum grid
    psi = psi*np.exp(-1j*D*dt*h_bar*p**2/m)
    psi = ifft(psi)

    # Final potential half step
    psi = psi*np.exp(-1j*potential_walsh*dt/2/h_bar)

    if out:
        states.append(psi)
    if verbose: progress.update(1)
    if verbose: progress.close()

    states = np.array(states, dtype='complex')
    return states, t_grid, x_grid

def walsh_evolve_4(potential, initial_wave_function, n, L, K, T, D=1/2, h_bar=1, m=1, terms_kept=None, verbose=True):
    N = 2**n
    dx, dt = 2*L/N, T/K
    x_grid = np.arange(-L, L - dx/2, dx)
    t_grid = np.arange(0, T + dt/2, dt)
    s = 1/(4-4**(1/3))

    # Initialize the initial wave function
    psi = initial_wave_function(x_grid)
    a = wft(potential, n, x_grid, verbose=verbose)
    potential_walsh = iwft(a, n, terms_kept=terms_kept, verbose=verbose)

    out = True
    states = [psi]
    if verbose: progress = tqdm(total = K, desc='working on time evolution')

    # propagate potential half step to start
    psi = psi*np.exp(-1j*potential_walsh*s*dt/2/h_bar)

    for i in range(K - 1):
        # propagate kinetic
        psi = fft(psi)
        p = 2*np.pi*fftfreq(N, d=dx)  # Momentum grid
        psi = psi*np.exp(-1j*D*s*dt*h_bar*p**2/m)
        psi = ifft(psi)

        # propagate potential
        psi = psi*np.exp(-1j*potential_walsh*s*dt/h_bar)
        
        # propagate kinetic
        psi = fft(psi)
        p = 2*np.pi*fftfreq(N, d=dx)  # Momentum grid
        psi = psi*np.exp(-1j*D*s*dt*h_bar*p**2/m)
        psi = ifft(psi)

        # propagate potential
        psi = psi*np.exp(-1j*potential_walsh*(1-3*s)/2*dt/h_bar)
        
        # propagate kinetic
        psi = fft(psi)
        p = 2*np.pi*fftfreq(N, d=dx)  # Momentum grid
        psi = psi*np.exp(-1j*D*(1-4*s)*dt*h_bar*p**2/m)
        psi = ifft(psi)

        # propagate potential
        psi = psi*np.exp(-1j*potential_walsh*(1-3*s)/2*dt/h_bar)
        
        # propagate kinetic
        psi = fft(psi)
        p = 2*np.pi*fftfreq(N, d=dx)  # Momentum grid
        psi = psi*np.exp(-1j*D*s*dt*h_bar*p**2/m)
        psi = ifft(psi)

        # propagate potential
        psi = psi*np.exp(-1j*potential_walsh*s*dt/h_bar)
        
        # propagate kinetic
        psi = fft(psi)
        p = 2*np.pi*fftfreq(N, d=dx)  # Momentum grid
        psi = psi*np.exp(-1j*D*s*dt*h_bar*p**2/m)
        psi = ifft(psi)

        # propagate potential
        psi = psi*np.exp(-1j*potential_walsh*s*dt/h_bar)

        if out:
            states.append(psi*np.exp(1j*potential_walsh*s*dt/2/h_bar))
        if verbose: progress.update(1)

    # propagate kinetic
    psi = fft(psi)
    p = 2*np.pi*fftfreq(N, d=dx)  # Momentum grid
    psi = psi*np.exp(-1j*D*s*dt*h_bar*p**2/m)
    psi = ifft(psi)

    # propagate potential
    psi = psi*np.exp(-1j*potential_walsh*s*dt/h_bar)

    # propagate kinetic
    psi = fft(psi)
    p = 2*np.pi*fftfreq(N, d=dx)  # Momentum grid
    psi = psi*np.exp(-1j*D*s*dt*h_bar*p**2/m)
    psi = ifft(psi)

    # propagate potential
    psi = psi*np.exp(-1j*potential_walsh*(1-3*s)/2*dt/h_bar)

    # propagate kinetic
    psi = fft(psi)
    p = 2*np.pi*fftfreq(N, d=dx)  # Momentum grid
    psi = psi*np.exp(-1j*D*(1-4*s)*dt*h_bar*p**2/m)
    psi = ifft(psi)

    # propagate potential
    psi = psi*np.exp(-1j*potential_walsh*(1-3*s)/2*dt/h_bar)

    # propagate kinetic
    psi = fft(psi)
    p = 2*np.pi*fftfreq(N, d=dx)  # Momentum grid
    psi = psi*np.exp(-1j*D*s*dt*h_bar*p**2/m)
    psi = ifft(psi)

    # propagate potential
    psi = psi*np.exp(-1j*potential_walsh*s*dt/h_bar)

    # propagate kinetic
    psi = fft(psi)
    p = 2*np.pi*fftfreq(N, d=dx)  # Momentum grid
    psi = psi*np.exp(-1j*D*s*dt*h_bar*p**2/m)
    psi = ifft(psi)

    # propagate potential
    psi = psi*np.exp(-1j*potential_walsh*s*dt/2/h_bar)

    if out:
        states.append(psi)
            
    if verbose: progress.update(1)
    if verbose: progress.close()

    states = np.array(states, dtype='complex')
    return states, t_grid, x_grid


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
def unitary_circuit(f, n, dt, x_grid, terms_kept=None, verbose=True):
    circ = QuantumCircuit(n)
    a = wft(f, n, x_grid, verbose=verbose)
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
    circ = transpile(circ, optimization_level=1)
    return circ


def kinetic(n_q, dx, dt, D):
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
        alpha_j = -(dt*D) * p_vals[j] ** 2 - (dt*D) * p_vals[j] * (p_sum - p_vals[j])
        qc.rz(alpha_j, j)

    for j in range(n_q):
        for l in range(j + 1, n_q):
                gamma_jl = (dt*D) * p_vals[j] * p_vals[l]
                qc.cx(j, l)
                qc.rz(gamma_jl, l)
                qc.cx(j, l)
    return qc


def walsh_evolve_quantum_1(potential, initial_wave_function, n, L, K, T, D=1/2, terms_kept=None, verbose=True, gate_count_only=False):
    N = 2**n
    dx, dt = 2*L/N, T/K
    x_grid = np.arange(-L, L, dx)
    t_grid = np.arange(0, T + dt/2, dt)

    # Initializing the quantum circuit
    qc = QuantumCircuit(n)

    # Initialize the initial wave function
    desired_vector = initial_wave_function(x_grid)
    if not gate_count_only: qc.prepare_state(state=desired_vector)

    potential_step = unitary_circuit(potential, n, dt, x_grid, terms_kept=terms_kept, verbose=False)
    kinetic_step = kinetic(n, dx, dt, D)

    iqft = QFT(num_qubits=n, inverse=True).decompose().to_gate()
    qft = QFT(num_qubits=n).decompose().to_gate()

    if gate_count_only: out = False
    else:
        out = True
        states = [Statevector.from_instruction(qc)]

    if verbose: progress = tqdm(total=K, desc='working on time evolution')
    for i in range(K):
        # Kinetic Step
        qc.append(qft, qargs=[i for i in range(n)])
        qc.append(kinetic_step, qargs=[i for i in range(n)])
        qc.append(iqft, qargs=[i for i in range(n)])

        # Potential Step
        qc.append(potential_step, qargs=[i for i in range(n)][::-1])

        if out:
            states.append(Statevector.from_instruction(qc))
        if verbose: progress.update(1)
    if verbose: progress.close()

    if out: states = np.array(states, dtype='complex')
    qc = qc.decompose()
    if gate_count_only: return qc.count_ops()
    return states, t_grid, x_grid

def walsh_evolve_quantum_2(potential, initial_wave_function, n, L, K, T, D=1/2, terms_kept=None, verbose=True, gate_count_only=False):
    N = 2**n
    dx, dt = 2*L/N, T/K
    x_grid = np.arange(-L, L - dx/2, dx)
    t_grid = np.arange(0, T + dt/2, dt)

    # Initializing the quantum circuit
    qc = QuantumCircuit(n)

    # Initialize the initial wave function
    desired_vector = initial_wave_function(x_grid)
    if not gate_count_only: qc.prepare_state(state=desired_vector)

    potential_step = unitary_circuit(potential, n, dt, x_grid, terms_kept=terms_kept, verbose=False)
    half_potential_step = unitary_circuit(potential, n, dt/2, x_grid, terms_kept=terms_kept, verbose=False)
    kinetic_step = kinetic(n, dx, dt, D)

    a = wft(potential, n, x_grid, verbose=False)
    potential_walsh = iwft(a, n, terms_kept=terms_kept, verbose=False)

    iqft = QFT(num_qubits=n, inverse=True).decompose().to_gate()
    qft = QFT(num_qubits=n).decompose().to_gate()

    if gate_count_only: out = False
    else:
        out = True
        states = [Statevector.from_instruction(qc)]

    qc.append(half_potential_step, qargs=[i for i in range(n)][::-1])
    # propagate potential half step to start
    if verbose: progress = tqdm(total=K, desc='working on time evolution')
    for i in range(K - 1):
        # Propagate kinetic
        qc.append(qft, qargs=[i for i in range(n)])
        qc.append(kinetic_step, qargs=[i for i in range(n)])
        qc.append(iqft, qargs=[i for i in range(n)])

        # Propagate potential
        qc.append(potential_step, qargs=[i for i in range(n)][::-1])
        if out:
            states.append(np.array(Statevector.from_instruction(qc))*np.exp(1j*potential_walsh*dt/2))
        if verbose: progress.update(1)

    # Propagate kinetic
    qc.append(qft, qargs=[i for i in range(n)])
    qc.append(kinetic_step, qargs=[i for i in range(n)])
    qc.append(iqft, qargs=[i for i in range(n)])

    # Ending potential half step
    qc.append(half_potential_step, qargs=[i for i in range(n)][::-1])

    if out:
        states.append(Statevector.from_instruction(qc))
    if verbose: progress.update(1)
    if verbose: progress.close()

    if out: states = np.array(states, dtype='complex')
    qc = qc.decompose()
    if gate_count_only: return qc.count_ops()
    return states, t_grid, x_grid

def walsh_evolve_quantum_4(potential, initial_wave_function, n, L, K, T, D=1/2, terms_kept=None, verbose=True, gate_count_only=False):
    N = 2**n
    dx, dt = 2*L/N, T/K
    x_grid = np.arange(-L, L - dx/2, dx)
    t_grid = np.arange(0, T + dt/2, dt)
    s = 1/(4-4**(1/3))

    # Initializing the quantum circuit
    qc = QuantumCircuit(n)

    # Initialize the initial wave function
    desired_vector = initial_wave_function(x_grid)
    if not gate_count_only: qc.prepare_state(state=desired_vector)

    potential_step_s = unitary_circuit(potential, n, s*dt, x_grid, terms_kept=terms_kept, verbose=False)
    half_potential_step_s = unitary_circuit(potential, n, s*dt/2, x_grid, terms_kept=terms_kept, verbose=False)
    interm_potential_step_s = unitary_circuit(potential, n, (1-3*s)*dt/2, x_grid, terms_kept=terms_kept, verbose=False)
    kinetic_step_s = kinetic(n, dx, s*dt, D)
    interm_kinetic_step_s = kinetic(n, dx, (1-4*s)*dt, D)

    a = wft(potential, n, x_grid, verbose=False)
    potential_walsh = iwft(a, n, terms_kept=terms_kept, verbose=False)

    iqft = QFT(num_qubits=n, inverse=True).decompose().to_gate()
    qft = QFT(num_qubits=n).decompose().to_gate()

    if gate_count_only: out = False
    else:
        out = True
        states = [Statevector.from_instruction(qc)]

    qc.append(half_potential_step_s, qargs=[i for i in range(n)][::-1])
    # propagate potential half step to start
    if verbose: progress = tqdm(total=K, desc='working on time evolution')
    for i in range(K - 1):
        # Propagate kinetic
        qc.append(qft, qargs=[i for i in range(n)])
        qc.append(kinetic_step_s, qargs=[i for i in range(n)])
        qc.append(iqft, qargs=[i for i in range(n)])

        # Propagate potential
        qc.append(potential_step_s, qargs=[i for i in range(n)][::-1])
        
        # Propagate kinetic
        qc.append(qft, qargs=[i for i in range(n)])
        qc.append(kinetic_step_s, qargs=[i for i in range(n)])
        qc.append(iqft, qargs=[i for i in range(n)])
        
        # Propagate potential
        qc.append(interm_potential_step_s, qargs=[i for i in range(n)][::-1])
        
        # Propagate kinetic
        qc.append(qft, qargs=[i for i in range(n)])
        qc.append(interm_kinetic_step_s, qargs=[i for i in range(n)])
        qc.append(iqft, qargs=[i for i in range(n)])
        
        # Propagate potential
        qc.append(interm_potential_step_s, qargs=[i for i in range(n)][::-1])
        
        # Propagate kinetic
        qc.append(qft, qargs=[i for i in range(n)])
        qc.append(kinetic_step_s, qargs=[i for i in range(n)])
        qc.append(iqft, qargs=[i for i in range(n)])
        
        # Propagate potential
        qc.append(potential_step_s, qargs=[i for i in range(n)][::-1])
        
        # Propagate kinetic
        qc.append(qft, qargs=[i for i in range(n)])
        qc.append(kinetic_step_s, qargs=[i for i in range(n)])
        qc.append(iqft, qargs=[i for i in range(n)])
        
        # Propagate potential
        qc.append(potential_step_s, qargs=[i for i in range(n)][::-1])
        
        # Append the states
        if out:
            states.append(np.array(Statevector.from_instruction(qc))*np.exp(1j*potential_walsh*s*dt/2))
        if verbose: progress.update(1)

    # Propagate kinetic
    qc.append(qft, qargs=[i for i in range(n)])
    qc.append(kinetic_step_s, qargs=[i for i in range(n)])
    qc.append(iqft, qargs=[i for i in range(n)])

    # Propagate potential
    qc.append(potential_step_s, qargs=[i for i in range(n)][::-1])

    # Propagate kinetic
    qc.append(qft, qargs=[i for i in range(n)])
    qc.append(kinetic_step_s, qargs=[i for i in range(n)])
    qc.append(iqft, qargs=[i for i in range(n)])

    # Propagate potential
    qc.append(interm_potential_step_s, qargs=[i for i in range(n)][::-1])

    # Propagate kinetic
    qc.append(qft, qargs=[i for i in range(n)])
    qc.append(interm_kinetic_step_s, qargs=[i for i in range(n)])
    qc.append(iqft, qargs=[i for i in range(n)])

    # Propagate potential
    qc.append(interm_potential_step_s, qargs=[i for i in range(n)][::-1])

    # Propagate kinetic
    qc.append(qft, qargs=[i for i in range(n)])
    qc.append(kinetic_step_s, qargs=[i for i in range(n)])
    qc.append(iqft, qargs=[i for i in range(n)])

    # Propagate potential
    qc.append(potential_step_s, qargs=[i for i in range(n)][::-1])

    # Propagate kinetic
    qc.append(qft, qargs=[i for i in range(n)])
    qc.append(kinetic_step_s, qargs=[i for i in range(n)])
    qc.append(iqft, qargs=[i for i in range(n)])

    # Propagate potential
    qc.append(half_potential_step_s, qargs=[i for i in range(n)][::-1])

    # Append the states
    if out:
        states.append(np.array(Statevector.from_instruction(qc)))
    if verbose: progress.update(1)
    if verbose: progress.close()

    if out: states = np.array(states, dtype='complex')
    qc = qc.decompose()
    if gate_count_only: return qc.count_ops()
    return states, t_grid, x_grid


def plot_time_evolution(amplitudes, t_grid, x_grid, interpolate_plot=True):
    T, X = np.meshgrid(t_grid, x_grid)
    C = amplitudes.T

    if interpolate_plot:
        grid_size = 1000
        t_interp, x_interp = np.linspace(T[0, 0], T[0, -1], grid_size), np.linspace(X[0, 0], X[-1, 0], grid_size)
        T_new, X_new = np.meshgrid(t_interp, x_interp)
        interp = RegularGridInterpolator((T[0, :], X[:, 0]), C.T, method='linear')
        plt.pcolor(T_new, -X_new, interp((T_new, X_new)), cmap='magma')
        plt.colorbar()
        plt.show()
    else:
        plt.pcolor(T, -X, C, cmap='magma')
        plt.yticks([-5, 0, 5], [5, 0, -5])
        plt.colorbar()
        plt.show()
