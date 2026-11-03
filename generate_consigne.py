import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # chemins relatifs au dossier du script
import numpy as np
import pandas as pd

TS = 0.05             # période d'échantillonnage (s), identique aux autres CSV
DUREE = 5             # durée totale de la consigne (s)
AMPLITUDE = 1         # amplitude du sinus
FREQUENCE = 1         # fréquence du sinus (Hz), soit une période de 1 s
OFFSET = -10          # valeur moyenne, proche du point de repos N(0) ≈ -9.81

t = np.arange(round(DUREE / TS)) * TS
yc = OFFSET + AMPLITUDE * np.sin(2 * np.pi * FREQUENCE * t)

# Une ligne, sans en-tête ni index (relue avec np.loadtxt comme les autres CSV)
pd.DataFrame([yc]).to_csv("consigne.csv", header=False, index=False)
print(f"Consigne {yc.shape} enregistrée ({yc.size * TS:.4g} s)")
