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

        self.u_m = torch.tensor([1.499473, -0.000128, 0.000055, -0.000685])   # z_ref, phi_ref, theta_ref, psi_ref
        self.u_s = torch.tensor([0.576533, 0.173159, 0.173025, 0.288451])
        self.a_m = torch.tensor([-0.000286, 0.000850, -0.002871])             # ax, ay, az
        self.a_s = torch.tensor([1.298574, 1.310540, 2.559979])

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
        a, v, p : mesures (3,) [ax, ay, az], [vx, vy, vz], [x, y, z] (az et vz ignorés)
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