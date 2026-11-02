"""Trace la réponse du modèle 1er ordre identifié par gain_statique.py
à la consigne bdd_in, et la compare à la sortie mesurée bdd_out.

Réponse à un échelon d'amplitude u :
    y(t) = offset + rapport * u * (1 - exp(-t/tau))
Pour une consigne quelconque (succession de paliers), on utilise la
discrétisation exacte de ce 1er ordre (u constant entre deux échantillons) :
    y[k+1] = a*y[k] + (1 - a)*(offset + rapport*u[k]),   a = exp(-Ts/tau)
"""

import matplotlib.pyplot as plt
import numpy as np

from det_constantes_model_lineaire import constantes_modele  # récupère les constantes

experience = 2 #ligne des CSV tracé

tau, rapport, offset, Ts, u, y = constantes_modele()
print(f"modèle : rapport = {rapport:.4f}, tau = {tau:.4f} s, offset = {offset:.4f}")


def simule(u, rapport, tau, offset):
    """Sortie du 1er ordre pour la consigne u, partant du repos (y = offset)."""
    a = np.exp(-Ts / tau)
    y = np.empty(len(u))
    y[0] = offset
    for k in range(len(u) - 1):
        y[k + 1] = a * y[k] + (1 - a) * (offset + rapport * u[k])
    return y


u = u[experience]
y_mes = y[experience]
y_mod = simule(u, rapport, tau, offset)
time = np.arange(len(u)) * Ts

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(time, u, color="red")
ax1.set_ylabel("consigne u")
ax1.grid()

ax2.plot(time, y_mes, '.', color="blue", label="mesure")
ax2.plot(time, y_mod, color="black", label="modèle")
ax2.set_ylabel("sortie y")
ax2.set_xlabel("time (s)")
ax2.legend()
ax2.grid()

fig.suptitle(f"Expérience {experience}")
plt.tight_layout()
plt.show()
