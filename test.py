from RNN import vanilla
import numpy as np 
import matplotlib.pyplot as plt
import os

data = "data/"

N_in = 1    # Number of input neurons
N_out = 1   # Number of outputs
N = 100     # Recurrent network neurons
g = 1.5     # connection strength (chaos factor)
tau = 0.1   # dynamical system time constant
dt = 0.01   # euler method time step
T = 2       # Total simulation time
p = 0.1     # connection probability

save_path = os.path.join(data, f"{N_in}-{N}-{N_out}-{T}s.npy")

RNN = vanilla.vanillaRNN(N=N, N_in=N_in, N_out=N_out, tau=tau, g=g, p=p)
RNN.add_input(max_amp=0.2, start=0.1, end=0.3, dt=dt, T=T)
RNN.save_weights(save_path)
RNN.train(dt=dt, T=T)

# plt.imshow(RNN.x_list)
# plt.imshow(RNN.dx_list)
# plt.show()
