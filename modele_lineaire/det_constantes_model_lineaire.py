"""Pour chaque palier suffisamment long, on considère que la sortie
a atteint le régime permanent sur ses derniers échantillons :

rapport = (y_permanent - offset) / u_palier
offset est la sortie moyenne avant le début de l'excitation (u = 0).

tau : temps pour atteindre 95 % de la variation finale de y sur le palier,
t95 = 3*tau (1er ordre) donc tau = t95 / 3.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

Ts = 0.05  # période d'échantillonnage (s)
DATA_DIR = Path(__file__).resolve().parent.parent


def segments_constants(u):
    """Renvoie la liste des (début, fin) des paliers constants de u."""
    ruptures = np.flatnonzero(np.diff(u) != 0) + 1
    bornes = np.r_[0, ruptures, len(u)]
    return list(zip(bornes[:-1], bornes[1:]))


def calcul_tau(y_palier, y_init, y_perm):
    """tau = t95 / 3, avec t95 le 1er instant où y atteint 95 % de (y_perm - y_init)."""
    seuil = y_init + 0.95 * (y_perm - y_init)
    # on se ramène à une montée pour gérer aussi les échelons descendants
    signe = np.sign(y_perm - y_init)
    z = signe * np.r_[y_init, y_palier]  # z[0] = valeur juste avant l'échelon (t = 0)
    s = signe * seuil
    k = np.argmax(z >= s)  # 1er échantillon au-dessus du seuil
    # interpolation linéaire entre k-1 et k pour ne pas être limité à Ts près
    t95 = (k - 1 + (s - z[k - 1]) / (z[k] - z[k - 1])) * Ts
    return t95 / 3


def rapports_experience(u, y, min_len, n_tail, u_min, dy_min):
    """Rapports y/u et constantes de temps en régime permanent pour une expérience.

    Renvoie l'offset et une liste de (début, u_palier, y_permanent, rapport, tau).
    """
    segs = segments_constants(u)
    # Offset : moyenne de y sur le premier palier à u = 0 (avant excitation)
    d0, f0 = segs[0]
    offset = y[d0:f0].mean() if np.all(u[d0:f0] == 0) else 0.0 #l'offset est récupérer en prenant la moyenne du début où la commande est nulle

    resultats = []
    for debut, fin in segs:
        u_val = u[debut]
        if fin - debut < min_len or abs(u_val) < u_min: #si regime permanent trop court ou s'il est trop prêt de 0
            continue
        y_perm = y[fin - n_tail:fin].mean() #on prend la moyenne de y_permanent
        rapport = (y_perm - offset) / u_val #on sort le rapport
        y_init = y[debut - 1] if debut > 0 else offset #valeur de y juste avant l'échelon
        #tau seulement si l'échelon en sortie est assez grand devant le bruit
        tau = calcul_tau(y[debut:fin], y_init, y_perm) if abs(y_perm - y_init) >= dy_min else np.nan
        resultats.append((debut, u_val, y_perm, rapport, tau)) #on rajoute les données de se régime permanent aux données
    return offset, resultats


def constantes_modele():
    bdd_in = DATA_DIR / "bdd_in_mat_05_postfiltre.csv"
    bdd_out = DATA_DIR / "bdd_out_mat_05_postfiltre.csv"
    min_len = 25 #longueur min d'un palier (échantillons) pour être en régime permanent
    n_tail = 5 #nb d'échantillons de fin de palier moyennés pour y permanent
    u_min = 0.05 #|u| minimal (évite de diviser par 0)
    dy_min = 2.0 #variation min de y sur l'échelon pour calculer tau (sinon noyé dans le bruit)


    u = np.loadtxt(bdd_in, delimiter=",")
    y = np.loadtxt(bdd_out, delimiter=",")
    assert u.shape == y.shape, "les CSV in et out n'ont pas la même taille"
    lignes = []          # (experience, t_debut, u, y_perm, offset, rapport, tau)
    rapport_moyen = np.full(u.shape[0], np.nan)
    for i in range(u.shape[0]):
        offset, res = rapports_experience(u[i], y[i], min_len, n_tail,
                                          u_min, dy_min)
        for debut, u_val, y_perm, r, tau in res:
            lignes.append((i, debut * Ts, u_val, y_perm, offset, r, tau))
        if res:
            rapport_moyen[i] = np.median([r for *_, r, _ in res])
    lignes = np.array(lignes)
    r = lignes[:, 5]
    tau = lignes[:, 6]
    tau = tau[~np.isnan(tau)]

    # print(f"{len(lignes)} paliers en régime permanent sur {u.shape[0]} expériences "
    #       f"({np.isnan(rapport_moyen).sum()} expériences sans palier exploitable)")
    # print(f"rapport y/u : moyenne = {r.mean():.4f}, médiane = {np.median(r):.4f}, "
    #       f"écart-type = {r.std():.4f}")
    # print(f"offset moyen = {lignes[:, 4].mean():.4f}")
    # print(f"tau ({len(tau)} paliers) : moyenne = {tau.mean():.4f} s, "
    #       f"médiane = {np.median(tau):.4f} s, écart-type = {tau.std():.4f} s")
    
    return tau.mean(), r.mean(), lignes[:, 4].mean(), Ts, u, y


