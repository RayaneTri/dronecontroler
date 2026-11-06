import torch
from torch import nn
import lightning as L
import numpy as np
#import matplotlib.pyplot as plt
from apply_EKF import EKF

class model_nn(L.LightningModule):
    """
    Sequential neural network model (RNN/LSTM/GRU) for aligned sequence-to-sequence mapping U->Y.
    """
    def __init__(
        self,
        input_size: int =   1,
        output_size: int =  1,
        rnn_type: str = "RNN",  # "RNN", "LSTM", "GRU"
        hidden_size: int =  16,
        num_layers: int =  1,
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        bidirectional: bool = False,
    ):
        """
        Initialize the model with the specified architecture and hyperparameters.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
            rnn_type (str): Type of recurrent layer ("RNN", "LSTM", "GRU").
            hidden_size (int): Number of hidden units in the RNN layer.
            num_layers (int): Number of recurrent layers.
            dropout (float): Dropout probability between RNN layers.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay (L2 penalty) for the optimizer.
            bidirectional (bool): If True, becomes a bidirectional RNN.
        """
        super().__init__()
        self.save_hyperparameters()
        # Select and initialize the RNN type
        if rnn_type.upper() == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif rnn_type.upper() == "GRU":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif rnn_type.upper() == "RNN":
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity='tanh',
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:
            raise ValueError("rnn_type must be one of {'RNN','LSTM','GRU'}")
        # Linear head for output projection
        d = 2 if bidirectional else 1
        self.head = nn.Linear(d * hidden_size, output_size)
        self.loss_fn = nn.MSELoss()  # Mean squared error loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, T, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch, T, output_size).
        """
        h, _ = self.rnn(x)              # h: (batch, T, d*hidden_size)
        y = self.head(h)                # y: (batch, T, output_size)
        return y

    def training_step(self, batch, batch_idx):
        """
        Training step: forward pass, loss computation, and logging.

        Args:
            batch: Tuple of (input, target) tensors.
            batch_idx: Index of the current batch.

        Returns:
            torch.Tensor: Computed loss.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step: forward pass, loss computation, and logging.

        Args:
            batch: Tuple of (input, target) tensors.
            batch_idx: Index of the current batch.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        Test step: forward pass, loss computation, and logging.

        Args:
            batch: Tuple of (input, target) tensors.
            batch_idx: Index of the current batch.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: Adam optimizer instance.
        """
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return opt

class MPC:
    """
    MPC sur le modèle RNN : entrées u = consignes [z_ref, phi_ref, theta_ref, psi_ref] du contrôleur interne,
    sorties du RNN = [ax, ay, z] : ax, ay sont intégrées en vitesses v = [vx, vy] et positions x, y ; z est lue directement.
    Coût : sum_k  Q ||p_k - r_k||^2 + R ||u_k - u_k_minus_1||^2
    """
    def __init__(self, model_path, Q, R, use_ekf, N=20, V=1, W=1):
        self.model = model_nn.load_from_checkpoint(model_path, map_location="cpu")
        self.model.eval()
        self.model.requires_grad_(False)
        self.use_ekf = use_ekf
        self.n_u = self.model.hparams.input_size                       # 4 consignes
        self.n_a = self.model.hparams.output_size                      # 3 sorties [ax, ay, z]

        # Statistiques de normalisation de l'entraînement (version_5, affichées par rnn_extended.ipynb > Display results)
        self.u_m = torch.tensor([1.502333, -0.000792, 0.002138, -0.002166])   # z_ref, phi_ref, theta_ref, psi_ref
        self.u_s = torch.tensor([0.577639, 0.172674, 0.173350, 0.289042])
        self.a_m = torch.tensor([0.019727, 0.009098, 1.504876])               # ax, ay, z
        self.a_s = torch.tensor([1.595419, 1.602934, 0.544539])

        # Bornes des consignes : plages vues à l'entraînement (au-delà, le RNN extrapole)
        self.u_min = torch.tensor([0.5, -0.3, -0.3, -0.5])
        self.u_max = torch.tensor([2.5, 0.3, 0.3, 0.5])

        self.Te = 0.05
        self.filter = EKF(self.model, q=V, r=W, dt=self.Te,
                          u_mean=self.u_m.numpy(), sr_mean=self.a_m.numpy(), u_std=self.u_s.numpy(), sr_std=self.a_s.numpy())

        self.N = N
        self.Q, self.R = Q, R
        u_0 = (self.u_min + self.u_max) / 2                            # consigne de départ : z_ref = 1.5, angles nuls
        self.U = u_0.repeat(N, 1).requires_grad_(True)                 # (N, n_u)

        # État interne : h_k (état caché du RNN, normalisé) ; a_k (ax, ay), v_k (vx, vy), p_k (x, y, z) en unités physiques
        self.h_k = torch.zeros(1, 1, self.model.hparams.hidden_size)
        self.a_k = np.zeros(2)
        self.v_k = np.zeros(2)
        self.p_k = np.zeros(3)

        self.u_prev = u_0.clone()                                      # commande appliquée au pas précédent (n_u,)
        self.opt = torch.optim.Adam([self.U], lr=0.05)

    def step_torch(self, h, u):
        """Un pas du RNN : u normalisée (n_u,) -> nouvel état caché et accélérations normalisées (n_a,)"""
        x = u.reshape(1, 1, -1)
        _, h_new = self.model.rnn(x, h)
        y = self.model.head(h_new[-1]).reshape(-1)
        return h_new, y

    def J(self, U, r, h0, v_0, p_0):
        h_k = h0
        v_k = torch.as_tensor(v_0, dtype=torch.float32)
        p_k = torch.as_tensor(p_0, dtype=torch.float32)
        cout = 0.0
        for k in range(self.N):
            h_k, a_k = self.step_torch(h_k, (U[k] - self.u_m) / self.u_s)

            sr_k = a_k * self.a_s + self.a_m                              # [ax, ay, z]
            v_k = v_k + sr_k[:2] * self.Te
            p_k = torch.cat([p_k[:2] + v_k * self.Te, sr_k[2:]])

            U_prec = self.u_prev if k == 0 else U[k-1]
            cout_q = torch.sum((p_k - r[k]) ** 2)
            cout = cout + self.Q * cout_q + self.R * torch.sum((U[k] - U_prec) ** 2)
        return cout

    def internal_state(self):
        """État interne courant [ax, ay, az, vx, vy, vz, x, y, z] en unités physiques
        (az et vz ne sont plus dans le modèle : renvoyés à 0 pour garder le format 9x1)."""
        return np.concatenate([self.a_k, [0.0], self.v_k, [0.0], self.p_k]).astype(np.float64)

    def step(self, r, a, v, p):
        """
        r : consigne de position sur l'horizon, (N, 3) [x, y, z] ou liste à plat x1 y1 z1 x2 y2 z2 ...
        a, v, p : mesures [ax, ay(, az)], [vx, vy(, vz)], [x, y, z] (az et vz éventuels ignorés)
        Renvoie u (n_u,) : consignes [z_ref, phi_ref, theta_ref, psi_ref] à appliquer
        """
        # 1. État interne à l'instant k, à partir de l'état en k-1 et de la commande appliquée au pas précédent u_{k-1}
        if self.use_ekf:
            # EKF : prédiction par le modèle + correction avec y = (ax, ay, z, vx, vy, px, py, pz)
            a, v, p = np.ravel(a), np.ravel(v), np.ravel(p)
            y = np.concatenate([a[:2], p[2:3], v[:2], p])
            self.filter.ekf(y, self.u_prev.numpy())
            self.h_k, self.a_k, self.v_k, self.p_k = self.filter.x0()
        else:
            # Boucle ouverte : prédiction par le modèle seul (mêmes équations que J), les mesures sont ignorées
            with torch.no_grad():
                self.h_k, a_n = self.step_torch(self.h_k, (self.u_prev - self.u_m) / self.u_s)
            sr = (a_n * self.a_s + self.a_m).numpy().astype(np.float64)   # [ax, ay, z]
            self.a_k = sr[:2]
            self.v_k = self.v_k + self.a_k * self.Te
            self.p_k = np.concatenate([self.p_k[:2] + self.v_k * self.Te, sr[2:]])

        # 2. Optimisation de la séquence de commandes à partir de l'état estimé
        r = torch.as_tensor(np.asarray(r, dtype=np.float32)).reshape(-1, self.n_a)   # (N, 3)
        for _ in range(50):
            self.opt.zero_grad()
            L = self.J(self.U, r, self.h_k, self.v_k, self.p_k)
            L.backward()
            self.opt.step()
            with torch.no_grad():
                self.U.clamp_(self.u_min, self.u_max)
        u = self.U[0].detach().clone()
        self.u_prev = u
        with torch.no_grad():
            self.U[:-1] = self.U[1:].clone()                           # warm start pour le pas suivant
        return u.numpy().astype(np.float64)


if __name__ == "__main__":
    # Test ---------------------------------------------------------------------------------------------------------------------------------------------
    # Le vrai drone est dans Simulink : ici le "système" est simulé par le RNN lui-même,
    # avec son propre état h_sys, séparé de l'état interne du MPC.
    from tqdm.auto import tqdm
    import matplotlib.pyplot as plt

    import os
    # RNN [z_ref, phi_ref, theta_ref, psi_ref] -> [ax, ay, z] (version_5), chemin relatif à ce fichier
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.ckpt')
    mpc = MPC(model_path, Q=1, R=1, use_ekf=True, N=20)     # N=10 (0.5 s) est trop court : le MPC diverge
    Te = mpc.Te
    T_sim = 200                                         # nombre de pas simulés (T_sim * Te = 10 s)
    sigma_mes = 0.01                                    # bruit de mesure sur a, v, p

    def systeme(h, v, p, u):
        """Un pas du système simulé : applique u (consignes physiques) -> nouvel état.
        a = (ax, ay), v = (vx, vy), p = (x, y, z) : mêmes équations que MPC.J."""
        with torch.no_grad():
            h, y_n = mpc.step_torch(h, (torch.as_tensor(u, dtype=torch.float32) - mpc.u_m) / mpc.u_s)
        sr = (y_n * mpc.a_s + mpc.a_m).numpy().astype(np.float64)       # [ax, ay, z]
        a = sr[:2]
        v = v + a * Te
        p = np.concatenate([p[:2] + v * Te, sr[2:]])
        return h, a, v, p

    # État initial du système : on applique la commande initiale du MPC (u_prev), comme le MPC le fait à son premier pas
    rng = np.random.default_rng(0)
    h_sys = torch.zeros(1, 1, mpc.model.hparams.hidden_size)
    h_sys, a_sys, v_sys, p_sys = systeme(h_sys, np.zeros(2), np.zeros(3), mpc.u_prev.numpy())

    # Sorties du RNN : [ax, ay, z]. x, y sont intégrés depuis 0 (déplacement par rapport au point de départ),
    # z est une sortie directe du RNN, donc en absolu : on part de l'altitude initiale du système (~1 m, comme dans le dataset)
    z_0 = p_sys[2]
    # Consigne de position [x, y, z], prolongée de N pas pour que la fenêtre de prédiction ne déborde pas à la fin
    t_full = np.arange(T_sim + mpc.N) * Te
    r_full = np.stack([
        np.interp(t_full, [0, 1, 4, 10], [0, 0, 1, 1]),                     # x : +1 m
        np.interp(t_full, [0, 3, 6, 10], [0, 0, -1, -1]),                   # y : -1 m
        np.interp(t_full, [0, 1, 3, 10], [z_0, z_0, z_0 + 0.5, z_0 + 0.5]), # z : +0.5 m (absolu)
    ], axis=1)

    hist = {'p': [], 'p_int': [], 'u': []}
    for t in tqdm(range(T_sim)):
        # Mesures bruitées de l'état à l'instant t : a (2,), v (2,), p (3,) (MPC.step n'utilise que a[:2] et v[:2])
        a_mes, v_mes, p_mes = (x + sigma_mes * rng.standard_normal(x.shape) for x in (a_sys, v_sys, p_sys))
        u = mpc.step(r_full[t:t + mpc.N], a_mes, v_mes, p_mes)

        hist['p'].append(p_sys.copy())                  # position réelle à l'instant t
        hist['p_int'].append(mpc.internal_state()[6:9]) # position estimée par le MPC à l'instant t
        hist['u'].append(u)

        # On applique u au système
        h_sys, a_sys, v_sys, p_sys = systeme(h_sys, v_sys, p_sys, u)

    p_hist, p_int_hist, u_hist = (np.array(hist[k]) for k in ('p', 'p_int', 'u'))
    for i, nom in enumerate('xyz'):
        e = p_hist[:, i] - r_full[:T_sim, i]
        print(f"{nom} : RMSE {np.sqrt(np.mean(e**2)):.3f} m, erreur finale {e[-1]:+.3f} m, "
              f"écart état interne / réel max {np.abs(p_int_hist[:, i] - p_hist[:, i]).max():.4f} m")

    # Affichage --------------------------------------------------------------------------------------------------------------------------------------
    t_axis = np.arange(T_sim) * Te
    fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for i, (ax, nom) in enumerate(zip(axs.flat[:3], 'xyz')):
        ax.plot(t_axis, r_full[:T_sim, i], 'k--', label="Consigne r")
        ax.plot(t_axis, p_hist[:, i], color='tab:orange', label="Position réelle")
        ax.plot(t_axis, p_int_hist[:, i], color='tab:green', linestyle=':', linewidth=2, label="État interne MPC")
        ax.set_ylabel(f"{nom} [m]")
        ax.grid(True, alpha=0.3)
    axs[0, 0].legend()
    axs[0, 0].set_title("MPC sur le modèle RNN")

    ax = axs[1, 1]
    for i, nom in enumerate(['z_ref [m]', 'phi_ref [rad]', 'theta_ref [rad]', 'psi_ref [rad]']):
        ax.step(t_axis, u_hist[:, i], where='post', label=nom)
    ax.set_title("Commandes u")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    for ax in axs[1]:
        ax.set_xlabel("Temps [s]")

    plt.tight_layout()
    plt.show()
