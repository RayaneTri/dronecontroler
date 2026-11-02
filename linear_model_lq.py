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

def N(U):
    """Caractéristique statique : y en régime établi pour la commande U."""
    return a3 * U**3 + a2 * U**2 + a1 * U + a0

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
plt.show()
