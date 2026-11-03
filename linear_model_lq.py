import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # chemins relatifs au dossier du script
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TS = 0.05      # période d'échantillonnage (s)

# Commandes et sorties filtrées (une ligne = un essai)
u = np.loadtxt("bdd_in_mat_05_postfiltre.csv", delimiter=",")
y = np.loadtxt("bdd_out_mat_05_postfiltre.csv", delimiter=",")

# Paramètres identifiés dans data_analysis.py (relus depuis echelons_longs.csv)
df = pd.read_csv("echelons_longs.csv")
a3, a2, a1, a0 = np.polyfit(df["amplitude"], df["y_fin"], 3)
tau = df["tau"].mean()
a = np.exp(-TS / tau)

# Commande LQ à horizon infini sur x[k+1] = a*x[k] + (1-a)*v[k], v[k] = N(u[k]), y = x
q = 4          # pondération sur l'état (reprise du MPC de rnn_mpc.ipynb)
r = 1          # pondération sur la commande (reprise du MPC de rnn_mpc.ipynb)
U_MAX = 1      # saturation de la commande : domaine d'identification de N (V)

# Solution scalaire de l'équation de Riccati, sous forme fermée
P = (q * (1 - a) - r * (1 + a)
     + np.sqrt((r * (1 + a) - q * (1 - a))**2 + 4 * q * r)) / (2 * (1 - a))
K = a * (1 - a) * P / (r + (1 - a)**2 * P)
N_lq = 1 + K   # précompensation pour un gain statique unitaire

def lqr(yc, x):
    """Commande LQ (dans l'espace de y, non saturée) pour atteindre la consigne yc depuis l'état x[k]."""
    return -K * x + N_lq * yc

def N(U):
    """Caractéristique statique : y en régime établi pour la commande U."""
    return a3 * U**3 + a2 * U**2 + a1 * U + a0

U_GRILLE = np.linspace(-U_MAX, U_MAX, 20001)

def N_inv(v):
    """Inverse de N sur [-U_MAX, U_MAX] : commande u donnant y = v en régime établi (saturée)."""
    return np.interp(v, N(U_GRILLE), U_GRILLE)

def simule(u, a):
    """Sortie du modèle de Hammerstein pour une matrice de commandes (un essai par ligne)."""
    v = N(u)
    x = np.empty_like(v)
    x[:, 0] = v[:, 0]  # départ en régime établi sur la première commande
    for k in range(u.shape[1] - 1):
        x[:, k + 1] = a * x[:, k] + (1 - a) * v[:, k]
    return x

y_pred = simule(u, a)
np.savetxt("bdd_pred_hammerstein.csv", y_pred, delimiter=",")

rmse = np.sqrt(np.mean((y - y_pred)**2))
print(f"tau = {tau:.4g} s, a = {a:.4g}")
print(f"Prédiction {y_pred.shape} enregistrée, RMSE = {rmse:.4g}")

def simule_lq(yc, a):
    """Boucle fermée LQ sur le modèle linéaire (un essai par ligne) : commandes et état prédit."""
    x = np.empty_like(yc)
    u_cmd = np.empty_like(yc)
    x[:, 0] = yc[:, 0]  # départ en régime établi sur la première consigne
    for k in range(yc.shape[1]):
        u_cmd[:, k] = N_inv(lqr(yc[:, k], x[:, k]))  # commande réelle u = N⁻¹(v), saturée
        if k + 1 < yc.shape[1]:
            x[:, k + 1] = a * x[:, k] + (1 - a) * N(u_cmd[:, k])
    return u_cmd, x

yc = np.loadtxt("consigne.csv", delimiter=",", ndmin=2)  # consigne (une ligne), sa longueur fixe celle de la commande
u_cmd, x_lq = simule_lq(yc, a)
np.savetxt("bdd_cmd_lq.csv", u_cmd, delimiter=",")
rmse_lq = np.sqrt(np.mean((yc - x_lq)**2))
print(f"Commandes LQ {u_cmd.shape} enregistrées, {np.mean(np.abs(u_cmd) >= U_MAX):.1%} saturées, "
      f"RMSE consigne/état = {rmse_lq:.4g}")

# Comparaison mesure / modèle sur un essai
i = 10
t = np.arange(u.shape[1]) * TS
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(t, u[i])
ax1.set_ylabel("entrée u")
ax2.plot(t, y[i], label="mesure")
ax2.plot(t, y_pred[i], label="modèle")
ax2.set_ylabel("sortie y")
ax2.set_xlabel("temps (s)")
ax2.legend()

# Boucle fermée LQ : consigne / état / commande
t_c = np.arange(yc.shape[1]) * TS
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig.suptitle("Boucle fermée LQ")
ax1.plot(t_c, yc[0], drawstyle="steps-post", label="consigne")
ax1.plot(t_c, x_lq[0], label="état x_lq")
ax1.set_ylabel("sortie y")
ax1.legend()
ax2.plot(t_c, u_cmd[0])
ax2.axhline(U_MAX, color="k", linestyle="--", linewidth=0.8)
ax2.axhline(-U_MAX, color="k", linestyle="--", linewidth=0.8)
ax2.set_ylabel("commande u")
ax2.set_xlabel("temps (s)")
plt.show()
