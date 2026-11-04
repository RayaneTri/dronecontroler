#site utilisé pour squelette EKF, author: Addison Sears-Collins: https://automaticaddison.com/extended-kalman-filter-ekf-with-python-code-example/
import numpy as np
import torch

class EKF:
    """
    EKF sur l'état augmenté (état caché du RNN, vitesse, position) :
        x_k = (hr_k;     (tanh(Wx hr_k_minus_1 + Wu u_k_minus_1 + bxu);
               v_k;   =   v_k_minus_1 + acc_k_minus_1 dt;                 + wb_k,   wb_k ~ N(0, Q_k)
               z_k)       z_k_minus_1 + v_k_minus_1 dt)
             with  acc_k_minus_1 = (Wf hr_k_minus_1 + bf) acc_std + acc_mean
        y_k = (acc_k;    (Wf hr_k + bf;
               v_k;   =   v_k;                                            + vb_k,   vb_k ~ N(0, R_k)
               z_k)       z_k)
             i.e.  y_k = H_k x_k + c + vb_k  with  c = (bf; 0; 0)
    acc = z''
    u est en unités physiques, acc en m.s-2, v en m.s-1, z en m à l'extérieur de la classe.
    À l'intérieur, u et acc sont normalisés, v et z restent en unités physiques.
    q et r peuvent être des scalaires ou des vecteurs (diagonales de Q_k et R_k).
    """
    def __init__(self, model,q=1.0, r=1.0, p0=0.1, dt=0.05, u_mean = -0.0023, acc_mean = -9.7708, u_std = 0.5731, acc_std = 12.5953):

        self.u_mean, self.u_std = u_mean, u_std
        self.acc_mean, self.acc_std = acc_mean, acc_std
        self.dt = dt

        #Extracting weights
        weight = model.state_dict()
        self.bxu = weight['rnn.bias_ih_l0'].cpu().numpy() + weight['rnn.bias_hh_l0'].cpu().numpy() # bxu : hidden layer bias vector (n_hr,)
        self.Wx = weight['rnn.weight_hh_l0'].cpu().numpy() # Wx : hidden to hidden layer weights
        self.Wu = weight['rnn.weight_ih_l0'].cpu().numpy() # Wu : input to hidden layer weights
        self.Wf = weight['head.weight'].cpu().numpy() # Wf : hidden to output layer weights
        self.bf = weight['head.bias'].cpu().numpy() # bf : output layer bias (n_acc,)

        self.n_hr = self.Wx.shape[0] # dimension de l'état caché du RNN
        self.n_x = self.n_hr + 2 # dimension de x_hat = (hr, v, z)
        self.n_y = self.bf.size + 2 # dimension de y = (acc, v, z)

        # State model noise covariance matrix Q_k
        self.Q_k = np.diag(np.broadcast_to(np.asarray(q, dtype=float), self.n_x))

        # Sensor measurement noise covariance matrix R_k
        self.R_k = np.diag(np.broadcast_to(np.asarray(r, dtype=float), self.n_y))

        # Measurement matrix H_k (linéaire) et biais c : y = H_k x + c
        self.H_k = np.zeros((self.n_y, self.n_x))
        self.H_k[:self.bf.size, :self.n_hr] = self.Wf
        self.H_k[-2, -2] = 1.0
        self.H_k[-1, -1] = 1.0
        self.c = np.concatenate([self.bf, np.zeros(2)])

        self.p0 = p0
        self.reset()

    def reset(self, x0=None):
        """Réinitialise l'estimation de l'état (hr, v, z) et sa covariance."""
        # The estimated state vector at time k-1
        self.x_hat_k_minus_1 = np.zeros(self.n_x) if x0 is None else np.asarray(x0, dtype=float)
        # State covariance matrix P_k_minus_1
        self.P_k_minus_1 = np.eye(self.n_x) * self.p0

    def f(self, x, u):
        """
        Modèle d'état : renvoie x_k = f(x_k_minus_1, u_k_minus_1) (u normalisé)
        """
        hr, v, z = x[:self.n_hr], x[-2], x[-1]
        acc = (self.Wf @ hr + self.bf).item() * self.acc_std + self.acc_mean
        hr_next = np.tanh(self.Wx @ hr + self.Wu @ np.atleast_1d(u) + self.bxu)
        return np.concatenate([hr_next, [v + acc * self.dt, z + v * self.dt]])

    def getF(self, x, u):
        """
        Calculates and returns the Jacobian F matrix (Jacobian of f according to x)
        """
        hr = x[:self.n_hr]
        s = self.Wx @ hr + self.Wu @ np.atleast_1d(u) + self.bxu
        F = np.zeros((self.n_x, self.n_x))
        F[:self.n_hr, :self.n_hr] = np.diag(1 - np.tanh(s)**2) @ self.Wx
        F[-2, :self.n_hr] = self.acc_std * self.dt * self.Wf
        F[-2, -2] = 1.0
        F[-1, -2] = self.dt
        F[-1, -1] = 1.0
        return F

    def getB(self, x, u):
        """
        Calculates and returns the Jacobian B matrix (Jacobian of f according to u)
        (not used but still here)
        """
        hr = x[:self.n_hr]
        s = self.Wx @ hr + self.Wu @ np.atleast_1d(u) + self.bxu
        B = np.zeros((self.n_x, self.Wu.shape[1]))
        B[:self.n_hr] = np.diag(1 - np.tanh(s)**2) @ self.Wu
        return B

    def ekf(self, y_k, u_k_minus_1):
        """
        Extended Kalman Filter at step k

        INPUT
            :param y_k Most recent sensor measurement (acc, v, z)
            :param u_k_minus_1 The control vector applied at time k-1

        OUTPUT
            :return x_hat_k near-optimal state estimate at time k
            :return P_k state covariance_estimate for time k
        """
        # Normalisation (seule l'accélération est normalisée)
        y_k = np.array(y_k, dtype=float)
        y_k[:self.bf.size] = (y_k[:self.bf.size] - self.acc_mean) / self.acc_std
        u_k_minus_1 = (u_k_minus_1 - self.u_mean) / self.u_std

        ######################### Predict #############################
        x_hat_k = self.f(self.x_hat_k_minus_1, u_k_minus_1)

        # Jacobian of the motion model, evaluated at the previous estimate
        F_k = self.getF(self.x_hat_k_minus_1, u_k_minus_1)

        P_k = F_k @ self.P_k_minus_1 @ F_k.T + self.Q_k

        ################### Update (Correct) ##########################
        measurement_residual_y_k = y_k - (self.H_k @ x_hat_k + self.c)

        S_k = self.H_k @ P_k @ self.H_k.T + self.R_k
        K_k = P_k @ self.H_k.T @ np.linalg.pinv(S_k)
        x_hat_k = x_hat_k + (K_k @ measurement_residual_y_k)
        P_k = P_k - (K_k @ self.H_k @ P_k)

        # Get ready for the next timestep
        self.x_hat_k_minus_1 = x_hat_k
        self.P_k_minus_1 = P_k

        return x_hat_k, P_k

    def step(self, y_k, u_k_minus_1):
        """
        Un pas d'EKF, renvoie la sortie estimée y_hat_k = (acc, v, z) en unités physiques et son écart-type sigma_y_hat_k
        """
        x_hat_k, P_k = self.ekf(y_k, u_k_minus_1)
        scale = np.concatenate([np.full(self.bf.size, self.acc_std), [1.0, 1.0]])
        offset = np.concatenate([np.full(self.bf.size, self.acc_mean), [0.0, 0.0]])
        y_hat_k = (self.H_k @ x_hat_k + self.c) * scale + offset
        var_y_hat_k = np.diag(self.H_k @ P_k @ self.H_k.T) * scale**2
        return y_hat_k, np.sqrt(var_y_hat_k)

    def h0(self):
        """État caché du RNN estimé au format torch (num_layers, batch, n_hr), ex. pour initialiser MPC.h."""
        return torch.tensor(self.x_hat_k_minus_1[:self.n_hr], dtype=torch.float32).reshape(1, 1, -1)
