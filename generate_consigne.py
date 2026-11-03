import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # chemins relatifs au dossier du script
import numpy as np
import pandas as pd

TS = 0.05             # période d'échantillonnage (s), identique aux autres CSV
DUREE_CRENEAU = 2     # durée d'un créneau (s)
DUREE = 10            # durée totale de la consigne (s)

# Niveaux successifs autour du point de repos N(0) ≈ -9.81, dans la plage atteignable N([-1, 1]) ≈ [-29.9, 10.3]
NIVEAUX = [-10, 0, -10, -20, -10]
assert len(NIVEAUX) * DUREE_CRENEAU == DUREE

yc = np.repeat(NIVEAUX, round(DUREE_CRENEAU / TS)).astype(float)

# Une ligne, sans en-tête ni index (relue avec np.loadtxt comme les autres CSV)
pd.DataFrame([yc]).to_csv("consigne.csv", header=False, index=False)
print(f"Consigne {yc.shape} enregistrée ({yc.size * TS:.4g} s)")
