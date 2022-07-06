from RNN import vanilla
import pytest
import numpy as np

def test_W():
    """
    When p=1, W.mean and W.std should be close to g**2/(Np)
    """
    N_in = 2    # Number of input neurons
    N_out = 1   # Number of outputs
    N = 100     # Recurrent network neurons
    g = 1.5     # connection strength (chaos factor)
    tau = 0.1   # dynamical system time constant
    p = 1     # connection probability
    RNN = vanilla.vanillaRNN(N=N, N_in=N_in, N_out=N_out, tau=tau, g=g, p=p)

    np.testing.assert_almost_equal(np.std(RNN.W), RNN.g**2/(RNN.N*RNN.p), decimal=2)



