import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # chemins relatifs au dossier du script
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TE = 0.05      # période d'échantillonnage (s)

# Mesures filtrées et prédictions des deux modèles (une ligne = un essai)
u = np.loadtxt("bdd_in_mat_05_postfiltre.csv", delimiter=",")
y = np.loadtxt("bdd_out_mat_05_postfiltre.csv", delimiter=",")
y_rnn = np.loadtxt("bdd_pred_rnn.csv", delimiter=",")
y_lin = np.loadtxt("bdd_pred_hammerstein.csv", delimiter=",")
n_essais, N = y.shape

def metriques(y, y_pred):
    """MAE et RMSE entre mesure et prédiction."""
    err = y - y_pred
    return np.mean(np.abs(err)), np.sqrt(np.mean(err**2))

# Jeu de test du RNN : 15 % finaux (il a été entraîné sur les 70 % premiers)
debut_test = int(0.7 * n_essais) + int(0.15 * n_essais)

res = {}
for nom, y_pred in [("RNN", y_rnn), ("Linéaire", y_lin)]:
    mae, rmse = metriques(y, y_pred)
    mae_t, rmse_t = metriques(y[debut_test:], y_pred[debut_test:])
    res[nom] = {"MAE": mae, "RMSE": rmse, "MAE test": mae_t, "RMSE test": rmse_t}
print(pd.DataFrame(res).T.round(4))

# Comparaison sur un essai du jeu de test
i = -1
t = np.arange(N) * TE
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(t, y[i], label="mesure")
ax1.plot(t, y_rnn[i], "--", label="RNN")
ax1.plot(t, y_lin[i], ":", label="linéaire")
ax1.set_ylabel("sortie y")
ax1.legend()
ax1.grid(True)
ax2.plot(t, u[i])
ax2.set_ylabel("entrée u")
ax2.set_xlabel("temps (s)")
ax2.grid(True)
plt.show()
