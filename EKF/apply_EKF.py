#site utilisé pour squelette EKF, author: Addison Sears-Collins: https://automaticaddison.com/extended-kalman-filter-ekf-with-python-code-example/
import numpy as np
from pathlib import Path
import torch
import matplotlib.pyplot as plt

# Données d'experience :
DATA_DIR = Path(__file__).resolve().parent.parent
bdd_in = DATA_DIR / "bdd_in_mat_05_postfiltre.csv"
bdd_out = DATA_DIR / "bdd_out_mat_05_postfiltre.csv"

class EKF:
    """
    EKF sur l'état augmenté (état caché du RNN, vitesse, position) :
        x_k = (hr_k;     (tanh(Wx hr_k_minus_1 + Wu u_k_minus_1 + bxu);
               vx_k;       vx_k_minus_1 + sr_k_minus_1[6] dt; 
               vy_k;       vy_k_minus_1 + sr_k_minus_1[7] dt;
               vz_k;   =   vz_k_minus_1 + sr_k_minus_1[8] dt;                 + wb_k,   wb_k ~ N(0, Q_k) 
               x_k;        x_k_minus_1 + vx_k_minus_1 dt;
               y_k;        y_k_minus_1 + vy_k_minus_1 dt;
               z_k)        z_k_minus_1 + vz_k_minus_1 dt)
             with  sr_k_minus_1 = (Wf hr_k_minus_1 + bf) sr_std + sr_mean
        y_k = (sr_k;    (Wf hr_k + bf;
               vx_k;      vx_k;
               vy_k;      vy_k;
               vz_k;   =  vz_k;                                            + vb_k,   vb_k ~ N(0, R_k)
               px_k;       px_k;
               py_k;       py_k;
               pz_k)       pz_k)
             i.e.  y_k = H_k x_k + c + vb_k  with  c = (bf; 0; 0)

    sr_k_minus_1[6] = acc_x = x'', sr_k_minus_1[7] = acc_y = y'', sr_k_minus_1[8] = acc_z = z''
    
    À l'intérieur, u et acc sont normalisés, v et z restent en unités physiques.
    q et r peuvent être des scalaires ou des vecteurs (diagonales de Q_k et R_k).
    """
    #u_mean = -0.0023, sr_mean = -9.7708, u_std = 0.5731, sr_std = 12.5953
    def __init__(self, model_path, model_nn, u_mean, sr_mean, u_std, sr_std, q=1.0, r=1.0, p0=0.1, dt=0.05):
        self.model = model_nn.load_from_checkpoint(model_path, map_location="cpu")
        self.model.eval()
        self.model.requires_grad_(False)

        self.dt = dt

        #Extracting weights
        weight = self.model.state_dict()
        self.bxu = weight['rnn.bias_ih_l0'].cpu().numpy() + weight['rnn.bias_hh_l0'].cpu().numpy() # bxu : hidden layer bias vector (n_hr,)
        self.Wx = weight['rnn.weight_hh_l0'].cpu().numpy() # Wx : hidden to hidden layer weights
        self.Wu = weight['rnn.weight_ih_l0'].cpu().numpy() # Wu : input to hidden layer weights
        self.Wf = weight['head.weight'].cpu().numpy() # Wf : hidden to output layer weights
        self.bf = weight['head.bias'].cpu().numpy() # bf : output layer bias (n_acc,)

        self.n_hr = self.Wx.shape[0] # dimension de l'état caché du RNN
        self.n_sr = self.bf.size # dimension de la sortie du RNN (12)
        self.n_u = self.Wu.shape[1] # dimension de la commande (4)
        self.i_acc = [6, 7, 8] # indices de (ax, ay, az) dans sr

        # Normalisation par canal (un scalaire est diffusé sur tous les canaux)
        self.u_mean = np.broadcast_to(np.asarray(u_mean, dtype=float), self.n_u)
        self.u_std = np.broadcast_to(np.asarray(u_std, dtype=float), self.n_u)
        self.sr_mean = np.broadcast_to(np.asarray(sr_mean, dtype=float), self.n_sr)
        self.sr_std = np.broadcast_to(np.asarray(sr_std, dtype=float), self.n_sr)

        self.n_extras = 6
        self.n_x = self.n_hr + self.n_extras # dimension de x_hat = (hr, vx, vy, vz, x, y, z)
        self.n_y = self.bf.size + self.n_extras # dimension de y = (sr, vx, vy, vz, x, y, z)

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
        """Réinitialise l'estimation de l'état (hr, vx, vy, vz, px, py, pz) et sa covariance."""
        # The estimated state vector at time k-1
        self.x_hat_k_minus_1 = np.zeros(self.n_x) if x0 is None else np.asarray(x0, dtype=float)
        # State covariance matrix P_k_minus_1
        self.P_k_minus_1 = np.eye(self.n_x) * self.p0

    def f(self, x, u):
        """
        Modèle d'état : renvoie x_k = f(x_k_minus_1, u_k_minus_1) (u normalisé)
        """
        hr, vx, vy, vz, px, py, pz = x[:self.n_hr], x[-6], x[-5], x[-4], x[-3], x[-2], x[-1]
        sr = (self.Wf @ hr + self.bf) * self.sr_std + self.sr_mean
        ax, ay, az = sr[self.i_acc]
        hr_next = np.tanh(self.Wx @ hr + self.Wu @ np.atleast_1d(u) + self.bxu)
        extras_next = [vx + ax * self.dt,
                       vy + ay * self.dt,
                       vz + az * self.dt,
                       px + vx * self.dt,
                       py + vy * self.dt,
                       pz + vz * self.dt]
        return np.concatenate([hr_next, extras_next])

    def getF(self, x, u):
        """
        Calculates and returns the Jacobian F matrix (Jacobian of f according to x)
        """
        hr = x[:self.n_hr]
        s = self.Wx @ hr + self.Wu @ np.atleast_1d(u) + self.bxu
        F = np.zeros((self.n_x, self.n_x))
        F[:self.n_hr, :self.n_hr] = np.diag(1 - np.tanh(s)**2) @ self.Wx
        for j, i_a in enumerate(self.i_acc): # axes x, y, z
            iv, ip = -6 + j, -3 + j # lignes de v_j et p_j
            # v_j_next = v_j + (Wf[i_a] hr + bf[i_a]) sr_std[i_a] dt + sr_mean[i_a] dt
            F[iv, :self.n_hr] = self.sr_std[i_a] * self.dt * self.Wf[i_a]
            F[iv, iv] = 1.0
            # p_j_next = p_j + v_j dt
            F[ip, iv] = self.dt
            F[ip, ip] = 1.0
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
            :param y_k Most recent sensor measurement (sr, vx, vy, vz, px, py, pz)
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
        Un pas d'EKF, renvoie la sortie estimée y_hat_k = (sr, vx, vy, vz, px, py, pz) en unités physiques et son écart-type sigma_y_hat_k
        """
        x_hat_k, P_k = self.ekf(y_k, u_k_minus_1)
        scale = np.concatenate([self.sr_std, np.ones(self.n_extras)])
        offset = np.concatenate([self.sr_mean, np.zeros(self.n_extras)])
        y_hat_k = (self.H_k @ x_hat_k + self.c) * scale + offset
        var_y_hat_k = np.diag(self.H_k @ P_k @ self.H_k.T) * scale**2
        return y_hat_k, np.sqrt(var_y_hat_k)

    def h0(self):
        """État caché du RNN estimé au format torch (num_layers, batch, n_hr), ex. pour initialiser MPC.h."""
        return torch.tensor(self.x_hat_k_minus_1[:self.n_hr], dtype=torch.float32).reshape(1, 1, -1)

def test_EKF():
    """
    Boucle EKF de test sur les données 3D (APRBS) de generate_training_Data
    """
    import sys
    from scipy.io import loadmat
    sys.path.append(str(DATA_DIR / "Matlab files"))
    from mpc_controller import model_nn

    # Checkpoint du RNN entraîné sur les données 3D (4 entrées -> 12 sorties)
    model_path = DATA_DIR / "Matlab files" / "model.ckpt"

    # inputs : essais x N x 4 (U1..U4), outputs : essais x N x 12 (phi..w_dot_z)
    data = loadmat(DATA_DIR / "generate_training_Data" / "data" / "prbs_data.mat")
    u = data["inputs"].astype(float)
    sr = data["outputs"].astype(float)

    # Normalisation par canal (à remplacer par celle utilisée à l'entraînement si différente)
    u_mean, u_std = u.mean(axis=(0, 1)), u.std(axis=(0, 1))
    sr_mean, sr_std = sr.mean(axis=(0, 1)), sr.std(axis=(0, 1))

    # definition de l'EKF
    filtre = EKF(model_path, model_nn, u_mean, sr_mean, u_std, sr_std, q=1, r=5)
    dt = filtre.dt

    essai = 0
    u_e, sr_e = u[essai], sr[essai]  # N x 4, N x 12

    # Pas de mesure de v et p dans les données : on les reconstruit en intégrant (ax, ay, az) (même schéma que le modèle)
    acc = sr_e[:, filtre.i_acc]  # N x 3
    v = np.vstack([np.zeros(3), np.cumsum(acc[:-1], axis=0) * dt])
    p = np.vstack([np.zeros(3), np.cumsum(v[:-1], axis=0) * dt])
    y = np.hstack([sr_e, v, p])  # N x 18

    # Sortie estimée et son écart-type avec EKF
    y_hat = []
    sigma_y_hat = []

    # Simulation du modèle seul (RNN + intégration)
    x_rnn = np.zeros(filtre.n_x)
    y_rnn = []

    # Start at k=1
    for k in range(1, y.shape[0]):

        # Un pas d'EKF : mesure à k, commande à k-1 (unités physiques)
        y_hat_k, sigma_y_hat_k = filtre.step(y[k], u_e[k-1])
        y_hat.append(y_hat_k)
        sigma_y_hat.append(sigma_y_hat_k)

        # Pas du modèle seul : même prédiction que l'EKF, mais sans mise à jour
        u_norm = (u_e[k-1] - filtre.u_mean) / filtre.u_std
        x_rnn = filtre.f(x_rnn, u_norm)
        sr_rnn = (filtre.Wf @ x_rnn[:filtre.n_hr] + filtre.bf) * filtre.sr_std + filtre.sr_mean
        y_rnn.append(np.concatenate([sr_rnn, x_rnn[-filtre.n_extras:]]))

    # Tracé y mesuré vs y estimé : lignes acc / v / p, colonnes x / y / z
    y_hat = np.array(y_hat)
    sigma_y_hat = np.array(sigma_y_hat)
    y_rnn = np.array(y_rnn)
    t = np.arange(1, y.shape[0])
    n_sr = filtre.n_sr
    lignes = [("acc", "m/s²", filtre.i_acc),
              ("v", "m/s", [n_sr, n_sr + 1, n_sr + 2]),
              ("p", "m", [n_sr + 3, n_sr + 4, n_sr + 5])]
    fig, axes = plt.subplots(3, 3, sharex=True, figsize=(14, 9))
    for r, (nom, unite, idx) in enumerate(lignes):
        for c, axe in enumerate("xyz"):
            i = idx[c]
            ax = axes[r, c]
            ax.plot(t, y[1:, i], label="mesuré")
            ax.plot(t, y_hat[:, i], label="estimé (EKF)")
            ax.fill_between(t, y_hat[:, i] - sigma_y_hat[:, i], y_hat[:, i] + sigma_y_hat[:, i], alpha=0.3, label="±σ")
            ax.plot(t, y_rnn[:, i], "--", label="prédit (modèle seul)")
            ax.set_ylabel(f"{nom}_{axe} ({unite})")
    axes[0, 0].legend()
    for ax in axes[-1]:
        ax.set_xlabel("k")
    fig.suptitle(f"EKF - essai {essai}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_EKF()
