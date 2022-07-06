import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
import numpy as np 
from scipy import sparse, stats
import os

plt.rcParams['image.cmap'] = 'seismic'

class vanillaRNN():
    """docstring for vanillaRNN."""
    def __init__(self, N: int, N_in: int, N_out: int, tau: float, g: float, p: float):
        super(vanillaRNN, self).__init__()
        self.N = N
        self.N_in = N_in
        self.N_out = N_out
        self.g = g
        self.tau = tau
        self.p = p
        self.__init_weights__()
        
        self.x = np.random.randn(1, self.N)
        self.NonLin = np.tanh

    def add_input(self, max_amp:float, start: float, end:float, dt:float, T:float):
        assert start <= end <= T
        I = np.zeros((int(T//dt), self.N_in))
        I[int(start//dt):int(end//dt), :] = max_amp
        self.I = I

    def __init_weights__(self):
        normal = stats.norm(loc=0, scale=self.g**2/(self.N*self.p)).rvs

        self.W_in = np.random.random((self.N_in, self.N))*2 - 1
        self.W = sparse.random(self.N, self.N, density=self.p, data_rvs=normal).A
        self.W_out = np.ones((self.N, self.N_out))
        self.W_fb = np.random.random((self.N_out, self.N))

    def _plot_weights(self, ax: np.ndarray): 

        ax[0].imshow(self.W_in, aspect="auto")
        ax[1].imshow(self.W, aspect="auto")
        ax[2].imshow(self.W_out, aspect="auto")

    def save_weights(self, save_path: str):
        # if os.path.exists(save_path):
        params_dict = {"N": self.N,
                       "N_in": self.N_in,
                       "g": self.g,
                       "tau": self.tau,
                       "p": self.p,
                       }
        weights_dict = {"W": self.W,
                        "W_in": self.W_in,
                        "W_out": self.W_out,
                        "W_fb": self.W_fb
                        }
        
        with open(save_path, 'wb') as f:
            np.save(f, arr={**params_dict, **weights_dict}, allow_pickle=True)


    def _update_simple(self, ts:float):

        self.dx = self.dt/self.tau * (-self.x + self.NonLin(self.x)@self.W + self.I[ts]@self.W_in)
        self.x = self.x + self.dx

    def train(self, dt: float, T: float):
        
        self.dt = dt
        self.T = T
        time_steps = np.arange(1, T, dt)
        self.x_list = np.zeros((len(time_steps), self.N))
        self.dx_list = np.zeros((len(time_steps), self.N))

        for i, ts in enumerate(time_steps):
            self._update_simple(i)
            self.x_list[i, :] = self.x
            self.dx_list[i, :] = self.dx
    #
    # def performance_error(self, target: np.ndarray, out: np.ndarray):
    #
    #     ep = 
    #
    #
