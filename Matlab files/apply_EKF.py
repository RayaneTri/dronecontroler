#site utilisé pour squelette EKF, author: Addison Sears-Collins: https://automaticaddison.com/extended-kalman-filter-ekf-with-python-code-example/
import numpy as np
import torch
from mpc_controller import model_nn

class EKF:
    """
    EKF sur l'état caché du RNN :
        x_k = tanh(Wx x_k_minus_1 + Wu u_k_minus_1 + bxu) + w_k,  w_k ~ N(0, Q_k)
        y_k = H_k x_k + bf + v_k,                                 v_k ~ N(0, R_k)
    u et y sont en unités physiques à l'extérieur de la classe, normalisés à l'intérieur.
    """
    def __init__(self, model, q=1.0, r=1.0, p0=0.1, u_mean = -0.0023, y_mean = -9.7708, u_std = 0.5731, y_std = 12.5953):
        self.u_mean, self.u_std = u_mean, u_std
        self.y_mean, self.y_std = y_mean, y_std
        
        #Extracting weights
        weight = model.state_dict()
        self.bxu = weight['rnn.bias_ih_l0'].cpu().numpy() + weight['rnn.bias_hh_l0'].cpu().numpy() # bxu : hidden layer bias vector (n_x,)
        self.Wx = weight['rnn.weight_hh_l0'].cpu().numpy() # Wx : hidden to hidden layer weights
        self.Wu = weight['rnn.weight_ih_l0'].cpu().numpy() # Wu : input to hidden layer weights
        self.Wf = weight['head.weight'].cpu().numpy() # Wf : hidden to output layer weights
        self.bf = weight['head.bias'].cpu().numpy() # bf : output layer bias (n_y,)

        self.n_x = self.Wx.shape[0] # dimension de l'état caché

        # State model noise covariance matrix Q_k
        self.Q_k = q * np.eye(self.n_x)

        # Sensor measurement noise covariance matrix R_k
        self.R_k = r * np.eye(self.bf.size)

        # Measurement matrix H_k
        self.H_k = self.Wf

        self.p0 = p0
        self.reset()

    def reset(self, x0=None):
        """Réinitialise l'estimation de l'état et sa covariance."""
        # The estimated state vector at time k-1
        self.x_hat_k_minus_1 = np.zeros(self.n_x) if x0 is None else np.asarray(x0, dtype=float)
        # State covariance matrix P_k_minus_1
        self.P_k_minus_1 = np.eye(self.n_x) * self.p0

    def getF(self, x, u):
        """
        Calculates and returns the Jacobian F matrix (Jacobian of F according to x)
        """
        z = self.Wx @ x + self.Wu @ np.atleast_1d(u) + self.bxu
        return np.diag(1 - np.tanh(z)**2) @ self.Wx

    def getB(self, x, u):
        """
        Calculates and returns the Jacobian B matrix (Jacobian of F according to u)
        (not used but still here)
        """
        z = self.Wx @ x + self.Wu @ np.atleast_1d(u) + self.bxu
        return np.diag(1 - np.tanh(z)**2) @ self.Wu

    def ekf(self, y_k, u_k_minus_1):
        """
        Extended Kalman Filter at step k

        INPUT
            :param y_k Most recent sensor measurement
            :param u_k_minus_1 The control vector applied at time k-1

        OUTPUT
            :return x_hat_k near-optimal state estimate at time k
            :return P_k state covariance_estimate for time k
        """
        # Normalisation
        y_k = (np.atleast_1d(y_k) - self.y_mean) / self.y_std
        u_k_minus_1 = (u_k_minus_1 - self.u_mean) / self.u_std

        ######################### Predict #############################
        x_hat_k = np.tanh(
                        self.Wx @ self.x_hat_k_minus_1 +
                        self.Wu @ np.atleast_1d(u_k_minus_1) +
                        self.bxu)

        # Jacobian of the motion model, evaluated at the previous estimate
        F_k = self.getF(self.x_hat_k_minus_1, u_k_minus_1)

        P_k = F_k @ self.P_k_minus_1 @ F_k.T + self.Q_k

        ################### Update (Correct) ##########################
        measurement_residual_y_k = y_k - (self.H_k @ x_hat_k + self.bf)

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
        Un pas d'EKF, renvoie la sortie estimée y_hat_k et son écart-type sigma_y_hat_k
        """
        x_hat_k, P_k = self.ekf(y_k, u_k_minus_1)
        y_hat_k = ((self.H_k @ x_hat_k + self.bf) * self.y_std + self.y_mean).item()
        var_y_hat_k = (self.H_k @ P_k @ self.H_k.T).item() * self.y_std**2
        return y_hat_k, np.sqrt(var_y_hat_k)

    def h0(self):
        """État caché estimé au format torch (num_layers, batch, n_x), ex. pour initialiser MPC.h."""
        return torch.tensor(self.x_hat_k_minus_1, dtype=torch.float32).reshape(1, 1, -1)
    