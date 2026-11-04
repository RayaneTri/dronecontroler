import torch
from torch import nn
import lightning as L
import numpy as np
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
    def __init__(self, model_path, Q, R, use_ekf, N=20, V=1, W=1):
        self.model = model_nn.load_from_checkpoint(model_path, map_location="cpu")
        self.model.eval()
        self.model.requires_grad_(False)
        self.use_ekf = use_ekf

        self.u_m, self.u_s = -0.0023, 0.5731                           # normalisation de la commande u
        self.az_m, self.az_s = -9.7708, 12.5953                        # normalisation de az (sortie du RNN)
        self.Te = 0.05
        self.filter = EKF(self.model, q=V, r=W, dt=self.Te,
                          u_mean=self.u_m, u_std=self.u_s, az_mean=self.az_m, az_std=self.az_s)

        self.N = N
        self.Q, self.R = Q, R
        self.U = torch.zeros(N, requires_grad=True)

        # État interne : h_k (état caché du RNN, normalisé) ; az_k, vz_k, z_k en unités physiques
        self.h_k = torch.zeros(1, 1, 16)
        self.az_k = 0.0
        self.vz_k = 0.0
        self.z_k = 0.0

        self.u_prev = 0.0                                              # commande appliquée au pas précédent
        self.opt = torch.optim.Adam([self.U], lr=0.05)

    def step_torch(self, h, u):
        x = u.reshape(1, 1, 1)
        _, h_new = self.model.rnn(x, h)
        y = self.model.head(h_new[-1])
        return h_new, y

    def J(self, U, r, h0, vz_0, z_0):
        h_k = h0
        vz_k = vz_0
        z_k = z_0
        cout = 0.0
        for k in range(self.N):
            h_k, az_k = self.step_torch(h_k, (U[k] - self.u_m) / self.u_s)
            az_k = az_k * self.az_s + self.az_m
            vz_k = vz_k + az_k * self.Te
            z_k = z_k + vz_k * self.Te

            cout += self.Q * (z_k - r[k]) ** 2 + self.R * U[k] ** 2
        return cout

    def internal_state(self):
        """État interne courant [az, vz, z] en unités physiques."""
        return [float(self.az_k), float(self.vz_k), float(self.z_k)]

    def step(self, r, az, vz, z):
        # 1. État interne à l'instant k, à partir de l'état en k-1 et de la commande appliquée au pas précédent u_{k-1}
        if self.use_ekf:
            # EKF : prédiction par le modèle + correction avec les mesures (az, vz, z)
            self.filter.ekf([az, vz, z], self.u_prev)
            self.h_k, self.az_k, self.vz_k, self.z_k = self.filter.x0()
        else:
            # Boucle ouverte : prédiction par le modèle seul (mêmes équations que J), les mesures sont ignorées
            with torch.no_grad():
                self.h_k, az_n = self.step_torch(self.h_k, torch.tensor((self.u_prev - self.u_m) / self.u_s))
            self.az_k = az_n.item() * self.az_s + self.az_m
            self.vz_k = self.vz_k + self.az_k * self.Te
            self.z_k = self.z_k + self.vz_k * self.Te

        # 2. Optimisation de la séquence de commandes à partir de l'état estimé
        r = torch.as_tensor(r, dtype=torch.float32).flatten()
        for _ in range(50):
            self.opt.zero_grad()
            L = self.J(self.U, r, self.h_k, self.vz_k, self.z_k)
            L.backward()
            self.opt.step()
            with torch.no_grad():
                self.U.clamp_(-1, 1)
        u = self.U[0].detach().item()
        self.u_prev = u
        with torch.no_grad():
            self.U[:-1] = self.U[1:].clone()                           # warm start pour le pas suivant
        return u