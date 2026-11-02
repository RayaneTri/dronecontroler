import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TE = 0.05      # période d'échantillonnage (s)
T_MIN = 1.5    # durée minimale d'un échelon (s)

# Chaque ligne = un essai ; la ligne k de "in" correspond à la ligne k de "out"
u = np.loadtxt("./bdd_in_mat_05.csv", delimiter=",")
y = np.loadtxt("./bdd_out_mat_05.csv", delimiter=",")

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

np.savetxt("./bdd_in_mat_05_postfiltre.csv", u[garde], delimiter=",")
np.savetxt("./bdd_out_mat_05_postfiltre.csv", y[garde], delimiter=",")

# La suite travaille sur les données saines
u = np.loadtxt("./bdd_in_mat_05_postfiltre.csv", delimiter=",")
y = np.loadtxt("./bdd_out_mat_05_postfiltre.csv", delimiter=",")
n_essais, N = u.shape

rows = []
for k in range(n_essais):
    # Découpage de l'entrée en paliers constants [s, e)
    idx = np.flatnonzero(np.diff(u[k]) != 0) + 1
    starts = np.r_[0, idx]
    ends = np.r_[idx, N]
    for s, e in zip(starts, ends):
        if (e - s) * TE > T_MIN:
            rows.append((k, s * TE, u[k, s], y[k, e - 1]))

df = pd.DataFrame(rows, columns=["essai", "t_debut", "amplitude", "y_fin"])

print(f"{len(df)} échelons de durée > {T_MIN} s sur {n_essais} essais")
print(df.head(10))
print(df.describe())

df.to_csv("./echelons_longs.csv", index=False)

# Caractéristique statique y∞ = f(u) : points seuls, très petits vu leur nombre
plt.figure()
plt.scatter(df["amplitude"], df["y_fin"], s=1)
plt.xlabel("u (amplitude de l'échelon)")
plt.ylabel("y∞ (sortie en fin d'échelon)")
plt.title("Caractéristique statique y∞ = f(u)")
plt.grid(True)
plt.show()
