import torch
from torch import nn
import lightning as L
import numpy as np
import matplotlib.pyplot as plt

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

u_mean = torch.tensor(-0.0023)
y_mean = torch.tensor(-9.7708)
u_std = torch.tensor(0.5731)
y_std = torch.tensor(12.5953)

class MPC:
    def __init__(self, model_path, Q, R, N=20):
        self.model = model_nn.load_from_checkpoint(model_path, map_location="cpu")
        self.model.eval()
        self.model.requires_grad_(False)

        self.u_m, self.u_s = u_mean.item(), u_std.item()
        self.y_m, self.y_s = y_mean.item(), y_std.item()

        self.N = N
        self.Q, self.R = Q, R
        self.U = torch.zeros(N, requires_grad=True)                   
        self.h = torch.zeros(1, 1, 16)
        self.y = 0
        self.opt = torch.optim.Adam([self.U], lr=0.05)     

    def step_torch(self, h, u):
        x = u.reshape(1, 1, 1)
        _, h_new = self.model.rnn(x, h)
        y = self.model.head(h_new[-1])      
        return h_new, y

    def J(self, U, r, h0):
        h = h0
        cout = 0.0
        for k in range(self.N):
            h, y = self.step_torch(h, (U[k] - self.u_m) / self.u_s)
            cout += self.Q * (y * self.y_s + self.y_m  - r[k]) ** 2 + self.R * U[k] ** 2
        return cout

    def internal_state(self):
        return self.y * self.y_s + self.y_m
    
    def step(self, r):
        r = torch.as_tensor(r, dtype=torch.float32).flatten()
        for _ in range(50):
            self.opt.zero_grad()
            L = self.J(self.U, r, self.h)
            L.backward()
            self.opt.step()
            with torch.no_grad():
                self.U.clamp_(-1, 1)
        u = float(self.U[0])
        with torch.no_grad():
            self.h, self.y = self.step_torch(self.h, torch.tensor((u - self.u_m) / self.u_s))
            self.U[:-1] = self.U[1:].clone()
        return u
    
Te = 0.05
# Boucle principale MPC ------------------------------------------------------------------------------------------------------------------------------
# Le vrai drone est dans Simulink (UAV_model.slx) : ici le "système" est simulé par le RNN lui-même,
# avec son propre état h_sys, séparé de l'état interne du MPC.
from tqdm.auto import tqdm

model_path = 'model_rnn/lightning_logs/version_0/model.ckpt'        # Path to the trained model checkpoint       # To be completed
mpc = MPC(model_path, Q=1, R=1, N=10)

T_sim = 200                                         # nombre de pas simulés (T_sim * Te = 10 s)

# Consigne sur y (accélération verticale) : stationnaire -> montée -> stationnaire -> descente -> stationnaire
# Prolongée de N pas pour que la fenêtre de prédiction ne déborde pas à la fin
f, A = 1, 2
t_full = np.arange(T_sim + mpc.N) * Te
r_full = A * np.sin(2 * np.pi * f * t_full)

h_sys = torch.zeros(1, 1, 16)                       # état du système simulé
u_hist, y_hist, y_int_hist = [], [], []

for t in tqdm(range(T_sim)):
    r_window = torch.tensor(r_full[t:t + mpc.N], dtype=torch.float32)
    u = mpc.step(r_window)                          # commande optimale (premier élément de U)

    # État interne du MPC (sortie prédite, déjà dénormalisée par internal_state)
    y_int = float(mpc.internal_state())
    y_int_hist.append(y_int)

    # On applique u au système et on mesure y
    with torch.no_grad():
        h_sys, y_n = mpc.step_torch(h_sys, torch.tensor((u - mpc.u_m) / mpc.u_s))
    y = y_n.item()

    u_hist.append(u)
    y_hist.append(y)

# Affichage ------------------------------------------------------------------------------------------------------------------------------------------
t_axis = np.arange(T_sim) * Te

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax1.plot(t_axis, r_full[:T_sim], 'k--', label="Consigne r")
ax1.plot(t_axis, y_hist, color='tab:orange', label="Sortie y")
ax1.plot(t_axis, y_int_hist, color='tab:green', linestyle=':', linewidth=2, label="État interne MPC (internal_state)")
ax1.set_ylabel("Accélération verticale")
ax1.set_title("MPC sur le modèle RNN")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.step(t_axis, u_hist, where='post', color='tab:blue', label="Commande u")
ax2.axhline(1, color='gray', linestyle=':', linewidth=1)
ax2.axhline(-1, color='gray', linestyle=':', linewidth=1)
ax2.set_xlabel("Temps [s]")
ax2.set_ylabel("u")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()