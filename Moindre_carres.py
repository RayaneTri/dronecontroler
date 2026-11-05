import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # chemins relatifs au dossier du script
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (16, 9),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 24,
    "axes.titlesize": 30,
    "axes.labelsize": 26,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 22,
    "axes.linewidth": 2.2,
    "lines.linewidth": 4,
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "xtick.major.size": 8,
    "ytick.major.size": 8,
})

FIGURES_DIR = os.path.join("presentation", "figures")

# 1. Chargement des données
print("Chargement des données...")
u = np.loadtxt("bdd_in_mat_05.csv", delimiter=",")   # Entrée commande (3000, 2000)
y = np.loadtxt("bdd_out_mat_05.csv", delimiter=",")  # Sortie accélération az (3000, 2000)

# Nettoyage simple des valeurs aberrantes (> 50 m/s^2)
mask = np.all(np.abs(y) < 50, axis=1)
u = u[mask]
y = y[mask]

# 2. Construction de la matrice X et du vecteur cible Y
# On veut : y(t+1) = a * y(t) + b * u(t) + c * 1
y_t   = y[:, :-1].ravel()   # Toutes les sorties à l'instant t
u_t   = u[:, :-1].ravel()   # Toutes les entrées à l'instant t
y_tp1 = y[:, 1:].ravel()    # Toutes les sorties à l'instant t+1 (cible)

# Matrice des régresseurs X = [y(t), u(t), 1]
X = np.column_stack([y_t, u_t, np.ones_like(y_t)])
Y = y_tp1

# 3. Résolution par les Moindres Carrés : theta = (X^T X)^(-1) * X^T * Y
# theta contient [a, b, c]
print("Calcul des moindres carrés...")
theta = np.linalg.solve(X.T @ X, X.T @ Y)
a, b, c = theta

print("\n--- MODÈLE IDENTIFIÉ ---")
print(f"a_z(t+1) = {a:.5f} * a_z(t) + {b:.5f} * u(t) + ({c:.5f})")
print(f"Gravité estimée au repos (u=0) : {c / (1 - a):.3f} m/s²")

# 4. Test et comparaison sur une trajectoire (ex : trajectoire n°10)
traj_idx = 10
u_test = u[traj_idx]
y_reel = y[traj_idx]

# Simulation récursive : le modèle calcule la suite de manière autonome
y_modele = np.zeros_like(y_reel)
y_modele[0] = y_reel[0]  # Même point de départ

for t in range(len(u_test) - 1):
    y_modele[t + 1] = a * y_modele[t] + b * u_test[t] + c

# 5. Affichage graphique
Ts = 0.05
temps = np.arange(len(u_test)) * Ts

plt.figure(figsize=(16, 9), dpi=120)
plt.plot(
    temps,
    y_reel,
    label="Measured output (Simulink)",
    color="#1769aa",
    linewidth=4,
)
plt.plot(
    temps,
    y_modele,
    label="Least-squares model prediction",
    color="#c62828",
    linestyle="--",
    linewidth=4,
)
plt.xlabel("Time (s)")
plt.ylabel(r"Vertical acceleration $a_z$ (m/s$^2$)")
plt.title(
    f"Least-Squares Linear Model Validation "
    f"(a={a:.4f}, b={b:.4f}, c={c:.4f})",
    pad=20,
    fontweight="bold",
)
plt.legend(frameon=True, framealpha=0.95)
plt.grid(True, linestyle="--", linewidth=1.5, alpha=0.4)
plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES_DIR, "least_squares_model_validation.png"),
    bbox_inches="tight",
)
plt.show()
