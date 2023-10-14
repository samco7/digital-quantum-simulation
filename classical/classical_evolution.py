import numpy as np
from scipy.fft import fft, ifft, fftfreq
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')


def time_evolve_1(potential, initial_wave_function, N, L, K, T, f, D=1/2, h_bar=1, m=1):
    dx, dt = 2*L/N, T/K
    x_grid = np.arange(-L, L - dx/2, dx)
    t_grid = np.arange(0, T + dt/2, dt)

    # Initialize the initial wave function
    psi = initial_wave_function(x_grid)

    out = True
    states = [psi]
    progress = tqdm(total = K, desc='working on time evolution')

    for i in range(K):
        # Propogate kinetic
        psi = fft(psi)
        p = 2*np.pi*fftfreq(N, d=dx)  # Momentum grid
        psi = psi*np.exp(-1j*D*dt*h_bar*p**2/m)
        psi = ifft(psi)

        # Propogate potential
        ff = f(np.abs(psi)**2)
        psi = psi*np.exp(-1j*(potential(x_grid) + ff)*dt/h_bar)

        if out:
            states.append(psi)
        progress.update(1)
    progress.close()

    states = np.array(states)
    return states, t_grid, x_grid

def time_evolve_2(potential, initial_wave_function, N, L, K, T, f, D=1/2, h_bar=1, m=1):
    dx, dt = 2*L/N, T/K
    x_grid = np.arange(-L, L - dx/2, dx)
    t_grid = np.arange(0, T + dt/2, dt)

    # Initialize the initial wave function
    psi = initial_wave_function(x_grid)

    out = True
    states = [psi]
    progress = tqdm(total = K, desc='working on time evolution')

    # Propogate potential half step to start
    ff = f(np.abs(psi)**2)
    psi = psi*np.exp(-1j*(potential(x_grid) + ff)*dt/2/h_bar)

    for i in range(K - 1):
        # Propogate kinetic
        psi = fft(psi)
        p = 2*np.pi*fftfreq(N, d=dx)  # Momentum grid
        psi = psi*np.exp(-1j*D*dt*h_bar*p**2/m)
        psi = ifft(psi)

        # Propogate potential
        ff = f(np.abs(psi)**2)
        psi = psi*np.exp(-1j*(potential(x_grid) + ff)*dt/h_bar)

        if out:
            states.append(psi.copy()*np.exp(1j*(potential(x_grid) + ff)*dt/2/h_bar))
        progress.update(1)

    # Propogate kinetic
    psi = fft(psi)
    p = 2*np.pi*fftfreq(N, d=dx)  # Momentum grid
    psi = psi*np.exp(-1j*D*dt*h_bar*p**2/m)
    psi = ifft(psi)
    psi = psi*np.exp(-1j*(potential(x_grid) + ff)*dt/2/h_bar)

    if out:
        states.append(psi)
    progress.update(1)
    progress.close()

    states = np.array(states, dtype='complex')
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
        plt.show()
    else:
        plt.pcolor(T, -X, C, cmap='magma')
        plt.yticks([-5, 0, 5], [5, 0, -5])
        plt.show()
