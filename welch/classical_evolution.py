import numpy as np
from scipy.fft import fft, ifft
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')


def time_evolve(potential, initial_wave_function, N, L, K, T, f, ord=2, h_bar=1, m=1):
    x_grid = np.linspace(-L, L, N)
    t_grid = np.linspace(0, T, K)
    dx, dt = x_grid[1] - x_grid[0], t_grid[1] - t_grid[0]

    # Initialize the initial wave function
    psi = initial_wave_function(x_grid)

    amplitudes = []
    progress = tqdm(total = K, desc='working on time evolution')

    # Propogate potential half step to start
    if ord == 2:
        pass
    elif ord == 1:
        pass
    else:
        raise(Exception('Only 1st and 2nd order implemented.'))

    for i in range(K):
        # Propogate potential (half) step
        if ord == 2:
            psi = psi*np.exp(-1j*(potential(x_grid) + f(np.abs(psi)**2))*dt/2/h_bar)
        else:
            psi = psi*np.exp(-1j*(potential(x_grid) + f(np.abs(psi)**2))*dt/h_bar)

        # Propogate kinetic
        psi = fft(psi)
        p = 2*np.pi*np.fft.fftfreq(N, d=dx)  # Momentum grid
        psi = psi*np.exp(-1j*dt*h_bar*p**2/(2*m))
        psi = ifft(psi)

        # Propogate potential half step if second order
        if ord == 2:
            psi = psi*np.exp(-1j*(potential(x_grid) + f(np.abs(psi)**2))*dt/2/h_bar)

        amplitudes.append(np.abs(psi)**2)
        progress.update(1)
    progress.close()

    amplitudes = np.array(amplitudes)/dx
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
