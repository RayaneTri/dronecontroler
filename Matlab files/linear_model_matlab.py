import numpy as np
U_MAX = 1
def N(U):
    """Caractéristique statique : y en régime établi pour la commande U."""
    return -6.045*U**3 + 0.001055*U**2 + 26.12*U + -9.81

U_GRILLE = np.linspace(-U_MAX, U_MAX, 20001)

def N_inv(v):
    """Inverse de N sur [-U_MAX, U_MAX] : commande u donnant y = v en régime établi (saturée)."""
    return np.interp(v, N(U_GRILLE), U_GRILLE)

class LQR:
    def __init__(self, q=1, r=1, tau=0.22, TS=0.05, x0=-9.81, *args, **kwargs):
        self.q = q          # pondération sur l'état (reprise du MPC de rnn_mpc.ipynb)
        self.r = r          # pondération sur la commande (reprise du MPC de rnn_mpc.ipynb)
        self.tau = tau
        self.TS = TS
        self.a = np.exp(-self.TS / self.tau)
        self.x = x0
        self.u = 0.0
        # Solution scalaire de l'équation de Riccati, sous forme fermée
        P = (self.q * (1 - self.a) - self.r * (1 + self.a) + np.sqrt((self.r * (1 + self.a) - self.q * (1 - self.a))**2 + 4 * self.q * self.r)) / (2 * (1 - self.a))
        self.K = self.a * (1 - self.a) * P / (self.r + (1 - self.a)**2 * P)
        self.N_lq = 1 + self.K 

    def lqr(self, yc, x):
        return -self.K * x + self.N_lq * yc

    def step(self, yc):
        if isinstance(yc, (list, tuple, np.ndarray)):
            yc = float(yc[0])
        else:
            yc = float(yc)
        v = self.lqr(yc, self.x)
        self.u = float(N_inv(v))
        self.x = float(self.a * self.x + (1 - self.a) * N(self.u))
        return self.u

LinearModel = LQR