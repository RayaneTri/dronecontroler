import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TE = 0.05      # période d'échantillonnage (s)
T_MIN = 1.5    # durée minimale d'un échelon (s)
DELTA_MIN = 0.5  # variation de sortie minimale pour mesurer tau

# Chaque ligne = un essai ; la ligne k de "in" correspond à la ligne k de "out"
u = np.loadtxt("bdd_in_mat_05.csv", delimiter=",")
y = np.loadtxt("bdd_out_mat_05.csv", delimiter=",")

# Filtrage des essais défectueux : capteur figé ou sortie aberrante
T_FIGE = 2.0   # durée max d'une mesure constante (s)
Y_MAX = 35     # seuil de dépassement de la sortie
N_DEPASS = 5   # nombre de dépassements toléré

def plus_longue_constante(x):
    """Longueur (en échantillons) de la plus longue suite de valeurs identiques."""
    idx = np.flatnonzero(np.diff(x) != 0) + 1
    return np.diff(np.r_[0, idx, len(x)]).max()

fige = np.array([plus_longue_constante(r) * TE > T_FIGE for r in y])
depasse = (y > Y_MAX).sum(axis=1) > N_DEPASS
garde = ~(fige | depasse)
print(f"Filtrage : {fige.sum()} essais figés, {depasse.sum()} avec y > {Y_MAX} plus de {N_DEPASS} fois "
      f"-> {(~garde).sum()} supprimés, {garde.sum()} conservés")

np.savetxt("bdd_in_mat_05_postfiltre.csv", u[garde], delimiter=",")
np.savetxt("bdd_out_mat_05_postfiltre.csv", y[garde], delimiter=",")

# La suite travaille sur les données saines
u = np.loadtxt("bdd_in_mat_05_postfiltre.csv", delimiter=",")
y = np.loadtxt("bdd_out_mat_05_postfiltre.csv", delimiter=",")
n_essais, N = u.shape

rows = []
for k in range(n_essais):
    # Découpage de l'entrée en paliers constants [s, e)
    idx = np.flatnonzero(np.diff(u[k]) != 0) + 1
    starts = np.r_[0, idx]
    ends = np.r_[idx, N]
    for s, e in zip(starts, ends):
        if (e - s) * TE > T_MIN:
            y_fin = y[k, e - 1]
            y_avant = y[k, s - 1] if s > 0 else y[k, 0]
            delta = y_fin - y_avant
            # Premier instant où la sortie a parcouru 63 % du delta
            atteint = np.flatnonzero((y[k, s:e] - y_avant) * np.sign(delta) >= 0.63 * abs(delta))
            # tau seulement pour une commande non nulle et un delta au-dessus du bruit
            mesurable = not np.isclose(u[k, s], 0) and abs(delta) > DELTA_MIN
            tau = atteint[0] * TE if mesurable and atteint.size else np.nan
            rows.append((k, s * TE, u[k, s], y_fin, delta, tau))

df = pd.DataFrame(rows, columns=["essai", "t_debut", "amplitude", "y_fin", "delta_y", "tau"])

print(f"{len(df)} échelons de durée > {T_MIN} s sur {n_essais} essais")
print(df.head(10))

# Constante de temps moyenne (NaN ignorés)
tau_moy = df["tau"].mean()
print(f"tau moyen = {tau_moy:.4g} s")

df.to_csv("echelons_longs.csv", index=False)

# Caractéristique statique y∞ = f(u) : points seuls, très petits vu leur nombre
plt.figure()
plt.scatter(df["amplitude"], df["y_fin"], s=1)
plt.xlabel("u (amplitude de l'échelon)")
plt.ylabel("y∞ (sortie en fin d'échelon)")
plt.title("Caractéristique statique y∞ = f(u)")
plt.grid(True)
plt.show()

# Régression cubique : les moindres carrés minimisent la RMSE
a3, a2, a1, a0 = np.polyfit(df["amplitude"], df["y_fin"], 3)

def Y_inf(U):
    """Sortie statique y∞ prédite pour une entrée U (scalaire ou tableau)."""
    return a3 * U**3 + a2 * U**2 + a1 * U + a0

rmse = np.sqrt(np.mean((df["y_fin"] - Y_inf(df["amplitude"]))**2))
print(f"y∞(u) = {a3:.4g} u³ + {a2:.4g} u² + {a1:.4g} u + {a0:.4g}")
print(f"RMSE = {rmse:.4g}")

# Superposition du modèle au nuage de points
u_lin = np.linspace(df["amplitude"].min(), df["amplitude"].max(), 200)
plt.figure()
plt.scatter(df["amplitude"], df["y_fin"], s=1, label="mesures")
plt.plot(u_lin, Y_inf(u_lin), "r", label=f"cubique (RMSE = {rmse:.3g})")
plt.xlabel("u (amplitude de l'échelon)")
plt.ylabel("y∞ (sortie en fin d'échelon)")
plt.title("Régression cubique y∞ = f(u)")
plt.legend()
plt.grid(True)
plt.show()
