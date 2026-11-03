"""Filtre de Kalman pour le modèle linéaire du drone.

Le modèle identifié dans :mod:`linear_model_matlab` est :

    x[k + 1] = a*x[k] + (1 - a)*N(u[k]) + w[k]
    y[k]     = x[k] + v[k]

Il est linéaire par rapport à l'état x. La caractéristique statique N(u)
est une entrée connue du modèle et ne nécessite donc pas d'approximation
dans le filtre de Kalman.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import linear_model_matlab


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = DATA_DIR / "bdd_in_mat_05.csv"
DEFAULT_OUTPUT = DATA_DIR / "bdd_out_mat_05.csv"

# Covariances des bruits de processus et de mesure.
# Elles peuvent être ajustées selon le niveau de bruit des données.
Q_K = 1
R_K = 1.3

def kf(
    y_k: float,
    x_hat_k_minus_1: float,
    u_k_minus_1: float,
    p_k_minus_1: float,
    *,
    a: float,
    q: float = Q_K,
    r: float = R_K,
    ) -> tuple[float, float]:
    """Effectue une itération du filtre de Kalman scalaire.

    Args:
        y_k: Mesure de sortie à l'instant k.
        x_hat_k_minus_1: Estimation de l'état à l'instant k-1.
        u_k_minus_1: Commande appliquée entre k-1 et k.
        p_k_minus_1: Variance de l'erreur d'estimation à l'instant k-1.
        a: Pôle discret du modèle du premier ordre.
        q: Variance du bruit de processus.
        r: Variance du bruit de mesure.

    Returns:
        L'estimation corrigée ``(x_hat_k, p_k)``.
    """
    
    # Prediction: l'entrée statique N(u) est connue sans incertitude.
    x_pred = (
        a * x_hat_k_minus_1
        + (1.0 - a) * linear_model_matlab.N(u_k_minus_1)
    )
    p_pred = a * p_k_minus_1 * a + q

    # Correction avec H = 1.
    innovation = y_k - x_pred
    innovation_variance = p_pred + r
    kalman_gain = p_pred / innovation_variance
    x_hat_k = x_pred + kalman_gain * innovation

    # calcul de p_k.
    p_k = (1.0 - kalman_gain) ** 2 * p_pred + kalman_gain**2 * r
    return float(x_hat_k), float(p_k)


def filter_experiment(
    u: np.ndarray,
    y: np.ndarray,
    *,
    a: float,
    q: float = Q_K,
    r: float = R_K,
    x0: float | None = None,
    p0: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Filtre une expérience complète et renvoie l'état estimé et son écart-type."""
    u = np.asarray(u, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if u.shape != y.shape:
        raise ValueError("u et y doivent avoir la même longueur")
    if u.size == 0:
        return np.empty(0), np.empty(0)

    estimates = np.empty_like(y)
    standard_deviations = np.empty_like(y)
    x_hat = float(y[0] if x0 is None else x0)
    p_hat = float(p0)

    # La première mesure initialise l'état; les suivantes utilisent u[k-1].
    estimates[0] = x_hat
    standard_deviations[0] = np.sqrt(p_hat)
    for k in range(1, y.size):
        x_hat, p_hat = kf(
            y[k],
            x_hat,
            u[k - 1],
            p_hat,
            a=a,
            q=q,
            r=r,
        )
        estimates[k] = x_hat
        standard_deviations[k] = np.sqrt(p_hat)

    return estimates, standard_deviations


def main() -> None:
    """Compare la sortie mesurée, le modèle seul et la sortie filtrée."""
    u = np.loadtxt(DEFAULT_INPUT, delimiter=",")
    y = np.loadtxt(DEFAULT_OUTPUT, delimiter=",")
    if u.ndim != 2 or y.ndim != 2 or u.shape != y.shape:
        raise ValueError("Les CSV doivent être deux matrices de même dimension")

    experience = 0
    model = linear_model_matlab.LQR()
    a = model.a
    y_hat, sigma_y_hat = filter_experiment(
        u[experience],
        y[experience],
        a=a,
    )

    y_model = np.empty_like(y[experience])
    y_model[0] = y[experience, 0]
    for k in range(1, y_model.size):
        y_model[k] = (
            a * y_model[k - 1]
            + (1.0 - a) * linear_model_matlab.N(u[experience, k - 1])
        )

    time = np.arange(y_model.size) * 0.05
    plt.figure()
    plt.plot(time, y[experience], label="y mesuré")
    plt.plot(time, y_model, "--", label="modèle linéaire")
    plt.plot(time, y_hat, label="y filtré (KF)")
    plt.fill_between(
        time,
        y_hat - sigma_y_hat,
        y_hat + sigma_y_hat,
        alpha=0.25,
        label="±σ",
    )
    plt.xlabel("Temps (s)")
    plt.ylabel("y")
    plt.title(f"Filtre de Kalman - essai {experience}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
