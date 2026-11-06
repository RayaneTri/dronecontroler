#site utilisé pour squelette EKF, author: Addison Sears-Collins: https://automaticaddison.com/extended-kalman-filter-ekf-with-python-code-example/
import numpy as np
import torch

class EKF:
    """
    EKF sur l'état augmenté (état caché du RNN, vitesse, position) :
        x_k = (hr_k;     (tanh(Wx hr_k_minus_1 + Wu u_k_minus_1 + bxu);
               vx_k;       vx_k_minus_1 + ax_k dt;
               vy_k;   =    vy_k_minus_1 + ay_k dt;          + wb_k,   wb_k ~ N(0, Q_k)                            
               x_k;        x_k_minus_1 + vx_k dt;
               y_k;        y_k_minus_1 + vy_k dt;
               z_k)        z_k  (= sr_k[2], sortie directe du RNN))
             with  sr_k = (Wf hr_k + bf) sr_std + sr_mean,  (ax_k, ay_k) = sr_k[i_acc]  et  z_k = sr_k[2]
        Euler semi-implicite, même schéma que MPC.J : l'accélération vient du NOUVEL état caché hr_k,
        et la position est intégrée avec la NOUVELLE vitesse v_k.
        y_k = (sr_k;    (Wf hr_k + bf;
               vx_k;      vx_k;
               vy_k;      vy_k;              + vb_k,   vb_k ~ N(0, R_k)                                 
               px_k;       px_k;
               py_k;       py_k;
               pz_k)       pz_k)
             i.e.  y_k = H_k x_k + c + vb_k  with  c = (bf; 0; 0; 0; 0; 0)

    sr_k[0:3] = (acc_x, acc_y, pz) = (x'', y'', z), avec i_acc = [0, 1] et i_z = 2 (pas de vz dans l'état)
    
    À l'intérieur, u et acc sont normalisés, v et z restent en unités physiques.
    q et r peuvent être des scalaires ou des vecteurs (diagonales de Q_k et R_k).
    """
    #u_mean = -0.0023, sr_mean = -9.7708, u_std = 0.5731, sr_std = 12.5953
    def __init__(self, model, u_mean, sr_mean, u_std, sr_std, q=1.0, r=1.0, p0=0.1, dt=0.05):
        self.dt = dt

        #Extracting weights
        weight = model.state_dict()
        self.bxu = weight['rnn.bias_ih_l0'].cpu().numpy() + weight['rnn.bias_hh_l0'].cpu().numpy() # bxu : hidden layer bias vector (n_hr,)
        self.Wx = weight['rnn.weight_hh_l0'].cpu().numpy() # Wx : hidden to hidden layer weights
        self.Wu = weight['rnn.weight_ih_l0'].cpu().numpy() # Wu : input to hidden layer weights
        self.Wf = weight['head.weight'].cpu().numpy() # Wf : hidden to output layer weights
        self.bf = weight['head.bias'].cpu().numpy() # bf : output layer bias (n_acc,)

        self.n_hr = self.Wx.shape[0] # dimension de l'état caché du RNN
        self.n_sr = self.bf.size # dimension de la sortie du RNN (12)
        self.n_u = self.Wu.shape[1] # dimension de la commande (4)
        self.i_acc = [0, 1] # indices de (ax, ay) dans sr
        self.i_z = 2 # indice de z dans sr

        # Normalisation par canal (un scalaire est diffusé sur tous les canaux)
        self.u_mean = np.broadcast_to(np.asarray(u_mean, dtype=float), self.n_u)
        self.u_std = np.broadcast_to(np.asarray(u_std, dtype=float), self.n_u)
        self.sr_mean = np.broadcast_to(np.asarray(sr_mean, dtype=float), self.n_sr)
        self.sr_std = np.broadcast_to(np.asarray(sr_std, dtype=float), self.n_sr)

        self.n_extras = 5
        self.n_x = self.n_hr + self.n_extras # dimension de x_hat = (hr, vx, vy, x, y, z)
        self.n_y = self.bf.size + self.n_extras # dimension de y = (sr, vx, vy, px, py, pz)

        # State model noise covariance matrix Q_k
        self.Q_k = np.diag(np.broadcast_to(np.asarray(q, dtype=float), self.n_x))

        # Sensor measurement noise covariance matrix R_k
        self.R_k = np.diag(np.broadcast_to(np.asarray(r, dtype=float), self.n_y))

        # Measurement matrix H_k (linéaire) et biais c : y = H_k x + c
        self.H_k = np.zeros((self.n_y, self.n_x))
        self.H_k[:self.bf.size, :self.n_hr] = self.Wf
        for i in range(1,self.n_extras+1):
            self.H_k[-i,-i] = 1.0
        self.c = np.concatenate([self.bf, np.zeros(self.n_extras)])

        self.p0 = p0
        self.reset()

    def reset(self, x0=None):
        """Réinitialise l'estimation de l'état (hr, vx, vy, px, py, pz) et sa covariance."""
        # The estimated state vector at time k-1
        self.x_hat_k_minus_1 = np.zeros(self.n_x) if x0 is None else np.asarray(x0, dtype=float)
        # State covariance matrix P_k_minus_1
        self.P_k_minus_1 = np.eye(self.n_x) * self.p0

    def f(self, x, u):
        """
        Modèle d'état : renvoie x_k = f(x_k_minus_1, u_k_minus_1) (u normalisé)
        """
        hr, v, p = x[:self.n_hr], x[-5:-3], x[-3:-1]
        hr_next = np.tanh(self.Wx @ hr + self.Wu @ np.atleast_1d(u) + self.bxu)
        sr = (self.Wf @ hr_next + self.bf) * self.sr_std + self.sr_mean   # sortie du RNN avec le nouvel état caché
        a = sr[self.i_acc]                                                # (ax_k, ay_k)
        v_next = v + a * self.dt                                          # Euler semi-implicite (comme MPC.J) :
        p_next = p + v_next * self.dt                                     # la position utilise la nouvelle vitesse
        z_next = sr[self.i_z]                                             # z est directement une sortie du RNN
        return np.concatenate([hr_next, v_next, p_next, [z_next]])

    def getF(self, x, u):
        """
        Calculates and returns the Jacobian F matrix (Jacobian of f according to x)
        """
        hr = x[:self.n_hr]
        s = self.Wx @ hr + self.Wu @ np.atleast_1d(u) + self.bxu
        A = np.diag(1 - np.tanh(s)**2) @ self.Wx                # d hr_k / d hr_k_minus_1
        F = np.zeros((self.n_x, self.n_x))
        F[:self.n_hr, :self.n_hr] = A
        for j, i_a in enumerate(self.i_acc): # axes x, y
            iv, ip = -5 + j, -3 + j # lignes de v_j et p_j
            d_a = self.sr_std[i_a] * (self.Wf[i_a] @ A)         # d a_j_k / d hr_k_minus_1 (a_j_k dépend de hr_k)
            # v_j_k = v_j_k_minus_1 + a_j_k dt
            F[iv, :self.n_hr] = self.dt * d_a
            F[iv, iv] = 1.0
            # p_j_k = p_j_k_minus_1 + v_j_k dt = p_j_k_minus_1 + v_j_k_minus_1 dt + a_j_k dt²
            F[ip, :self.n_hr] = self.dt**2 * d_a
            F[ip, iv] = self.dt
            F[ip, ip] = 1.0
        # z_k = sr_k[2] dépend seulement de hr_k
        F[-1, :self.n_hr] = self.sr_std[self.i_z] * (self.Wf[self.i_z] @ A)
        return F

    def getB(self, x, u):
        """
        Calculates and returns the Jacobian B matrix (Jacobian of f according to u)
        (not used but still here)
        """
        hr = x[:self.n_hr]
        s = self.Wx @ hr + self.Wu @ np.atleast_1d(u) + self.bxu
        dhr = np.diag(1 - np.tanh(s)**2) @ self.Wu              # d hr_k / d u_k_minus_1
        B = np.zeros((self.n_x, self.Wu.shape[1]))
        B[:self.n_hr] = dhr
        for j, i_a in enumerate(self.i_acc): # axes x, y
            d_a = self.sr_std[i_a] * (self.Wf[i_a] @ dhr)       # d a_j_k / d u_k_minus_1
            B[-5 + j] = self.dt * d_a
            B[-3 + j] = self.dt**2 * d_a
        B[-1] = self.sr_std[self.i_z] * (self.Wf[self.i_z] @ dhr)
        return B

    def ekf(self, y_k, u_k_minus_1):
        """
        Extended Kalman Filter at step k

        INPUT
            :param y_k Most recent sensor measurement (sr, vx, vy, px, py, pz)
            :param u_k_minus_1 The control vector applied at time k-1

        OUTPUT
            :return x_hat_k near-optimal state estimate at time k
            :return P_k state covariance_estimate for time k
        """
        # Normalisation (seuls sr et u sont normalisés, v et p restent en unités physiques)
        y_k = np.array(y_k, dtype=float)
        y_k[:self.n_sr] = (y_k[:self.n_sr] - self.sr_mean) / self.sr_std
        u_k_minus_1 = (np.asarray(u_k_minus_1, dtype=float) - self.u_mean) / self.u_std

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
        Un pas d'EKF, renvoie la sortie estimée y_hat_k = (sr, vx, vy, px, py, pz) en unités physiques et son écart-type sigma_y_hat_k
        """
        x_hat_k, P_k = self.ekf(y_k, u_k_minus_1)
        scale = np.concatenate([self.sr_std, np.ones(self.n_extras)])
        offset = np.concatenate([self.sr_mean, np.zeros(self.n_extras)])
        y_hat_k = (self.H_k @ x_hat_k + self.c) * scale + offset
        var_y_hat_k = np.diag(self.H_k @ P_k @ self.H_k.T) * scale**2
        return y_hat_k, np.sqrt(var_y_hat_k)

    def x0(self):
        """
        État estimé courant pour le MPC (après ekf(), x_hat_k_minus_1 contient l'estimation à l'instant k) :
            h : état caché du RNN au format torch (num_layers, batch, n_hr)
            a : accélérations (ax, ay) en m.s-2, numpy (2,) : elles ne sont pas dans x,
                on les recalcule à partir de hr_k, comme dans la mesure y_k = Wf hr_k + bf
            v : vitesses (vx, vy) en m.s-1, numpy (2,)
            p : positions (x, y, z) en m, numpy (3,)
        """
        x = self.x_hat_k_minus_1
        hr = x[:self.n_hr]
        h = torch.tensor(hr, dtype=torch.float32).reshape(1, 1, -1)
        sr = (self.Wf @ hr + self.bf) * self.sr_std + self.sr_mean   # sortie du RNN dénormalisée
        a = sr[self.i_acc]
        return h, a, x[-5:-3].copy(), x[-3:].copy()