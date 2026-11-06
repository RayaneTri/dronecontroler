"""
Commande du système en position dans l'espace (x, y, z)
"""

from math import factorial
import numpy as np

DEGRE = 7          # polynomes de degré 7 par segment (minimum snap)
NB_COEFS = DEGRE + 1


class TRAJECTOIRE_ESPACE:
    """
    Trajectoire 3D polynomiale par morceaux (minimum snap) passant par des points de passage
        points : liste de N >= 2 points (x, y, z)
        temps : instants de passage (optionnel, strictement croissants, longueur N)
        v_moy : vitesse moyenne utilisée si temps n'est pas donné (m/s)
        duree : durée totale utilisée si temps n'est pas donné (prioritaire sur v_moy)
        t0 : instant de départ si temps n'est pas donné
    Chaque segment est un polynome de degré 7 par axe. La trajectoire est C6 aux points intermédiaires
    (vitesse libre : le drone ne s'y arrête pas) et vitesse = accélération = jerk = 0 au départ et à l'arrivée
    (drone arrêté, horizontal et sans vitesse de rotation).
    """
    def __init__(self, points, temps=None, v_moy: float = 1.0, duree: float = None, t0: float = 0.0):
        self.points = np.asarray(points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 3 or self.points.shape[0] < 2:
            raise ValueError("points doit être de dimension (N, 3) avec N >= 2")
        self.v_moy = v_moy
        self.duree = duree
        self.t0 = t0
        if temps is None:
            self.temps = self._allouer_temps()
        else:
            self.temps = np.asarray(temps, dtype=float)
            if self.temps.shape != (len(self.points),):
                raise ValueError("temps doit avoir autant d'éléments que points")
            if np.any(np.diff(self.temps) <= 0):
                raise ValueError("temps doit être strictement croissant")
        self.durees = np.diff(self.temps)
        self.coefs = self._coefficients()   # (n_segments, 8, 3), en temps local tau = t - temps[i]

    def _allouer_temps(self):
        """
        Durées des segments proportionnelles aux distances entre points consécutifs
        """
        distances = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        if np.any(distances <= 0):
            raise ValueError("deux points consécutifs identiques : donner temps explicitement")
        total = self.duree if self.duree is not None else distances.sum() / self.v_moy
        return self.t0 + np.concatenate(([0.0], np.cumsum(distances / distances.sum() * total)))

    @staticmethod
    def _ligne_derivee(k: int, tau: float):
        """
        Ligne d^k/dtau^k [1, tau, ..., tau^7] évaluée en tau
        """
        ligne = np.zeros(NB_COEFS)
        for j in range(k, NB_COEFS):
            ligne[j] = factorial(j) / factorial(j - k) * tau**(j - k)
        return ligne

    def _coefficients(self):
        """
        Assemble et résout le système linéaire (8n x 8n), commun aux trois axes.
        Chaque segment i est p_i(tau) = sum_j c_ij tau^j avec tau = t - temps[i] dans [0, T_i].
        """
        n = len(self.durees)
        T = self.durees
        A = np.zeros((NB_COEFS*n, NB_COEFS*n))
        B = np.zeros((NB_COEFS*n, 3))
        ligne = 0

        def ajouter(seg, k, tau, facteur=1.0):
            A[ligne, NB_COEFS*seg:NB_COEFS*(seg+1)] += facteur * self._ligne_derivee(k, tau)

        # positions aux deux extrémités de chaque segment : p_i(0) = P_i, p_i(T_i) = P_i+1
        for i in range(n):
            ajouter(i, 0, 0.0); B[ligne] = self.points[i]; ligne += 1
            ajouter(i, 0, T[i]); B[ligne] = self.points[i+1]; ligne += 1
        # continuité des dérivées 1 à 6 aux points intermédiaires : p_i^(k)(T_i) - p_i+1^(k)(0) = 0
        for i in range(n - 1):
            for k in range(1, DEGRE):
                ajouter(i, k, T[i])
                ajouter(i+1, k, 0.0, -1.0)
                ligne += 1
        # vitesse, accélération et jerk nuls au départ et à l'arrivée
        for k in (1, 2, 3):
            ajouter(0, k, 0.0); ligne += 1
            ajouter(n-1, k, T[n-1]); ligne += 1

        coefs = np.linalg.solve(A, B)
        return coefs.reshape(n, NB_COEFS, 3)

    def _segment(self, t: float):
        """
        Indice du segment actif à l'instant t
        """
        i = int(np.searchsorted(self.temps, t, side='right')) - 1
        return min(max(i, 0), len(self.durees) - 1) #manip pour gérer les cas ou t n'appartient pas au temp de l'experience

    def _derivee(self, t: float, k: int):
        """
        Dérivée k-ième de la position à l'instant t (vecteur 3). Hors de [t0, tf] : drone immobile
        """
        if t <= self.temps[0] or t >= self.temps[-1]:
            return self.points[0 if t <= self.temps[0] else -1].copy() if k == 0 else np.zeros(3)
        i = self._segment(t)
        tau = t - self.temps[i] #tau est le temps local depuis le début du segment
        return self._ligne_derivee(k, tau) @ self.coefs[i]

    def _derivee_segment(self, i: int, tau: float, k: int):
        """
        Dérivée k-ième du segment i en tau (sans saturation), pour les vérifications
        """
        return self._ligne_derivee(k, tau) @ self.coefs[i]

    def position(self, t):
        """
        Renvoie la position (x, y, z) à l'instant t
        t scalaire -> array (3,) ; t tableau -> array (len(t), 3)
        Avant temps[0] / après temps[-1], renvoie le premier / dernier point
        """
        if np.ndim(t) == 0:
            return self._derivee(float(t), 0)
        return np.array([self._derivee(float(ti), 0) for ti in np.ravel(t)])

    def tracer(self, Ts: float = 0.01):
        """
        Trace la trajectoire 3D et x, y, z en fonction du temps
        """
        import matplotlib.pyplot as plt

        t = np.arange(self.temps[0], self.temps[-1] + Ts/2, Ts)
        pos = self.position(t)

        fig = plt.figure(figsize=(12, 5))
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')
        ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2])
        ax3d.plot(*self.points.T, 'o')
        ax3d.set_xlabel("x (m)"); ax3d.set_ylabel("y (m)"); ax3d.set_zlabel("z (m)")
        ax = fig.add_subplot(1, 2, 2)
        for j, nom in enumerate("xyz"):
            ax.plot(t, pos[:, j], label=nom)
            ax.plot(self.temps, self.points[:, j], 'o', color=ax.lines[-1].get_color())
        ax.set_xlabel("temps (s)"); ax.set_ylabel("position (m)")
        ax.legend(); ax.grid()
        fig.suptitle("Trajectoire de référence dans l'espace")
        plt.show()


def verifier(traj: TRAJECTOIRE_ESPACE, tol=1e-6):
    """
    Vérifie passage par les points, conditions d'arrêt aux extrémités et continuité C6
    """
    for ti, pi in zip(traj.temps, traj.points):
        assert np.allclose(traj.position(ti), pi, atol=tol), "point de passage non atteint"
    n = len(traj.durees)
    for k in (1, 2, 3):
        assert np.allclose(traj._derivee_segment(0, 0.0, k), 0, atol=tol), f"dérivée {k} non nulle au départ"
        assert np.allclose(traj._derivee_segment(n-1, traj.durees[-1], k), 0, atol=tol), f"dérivée {k} non nulle à l'arrivée"
    for i in range(n - 1):
        for k in range(DEGRE):
            g = traj._derivee_segment(i, traj.durees[i], k)
            d = traj._derivee_segment(i+1, 0.0, k)
            assert np.allclose(g, d, atol=tol * max(1.0, np.abs(g).max())), f"dérivée {k} discontinue au point {i+1}"


if __name__ == "__main__":
    points = [[0,0,0], [0,0,0.5], [0, 0, -0.2]]
    traj = TRAJECTOIRE_ESPACE(points, v_moy=0.5)
    verifier(traj)
    verifier(TRAJECTOIRE_ESPACE(points, temps=[0, 5, 10]))
    verifier(TRAJECTOIRE_ESPACE(points, duree=10, t0=1))
    verifier(TRAJECTOIRE_ESPACE([[0, 0, 0], [1, 2, 3]], duree=5))
    print("Vérifications OK")
    print("Instants de passage :", np.round(traj.temps, 3))
    print("Position à t=3s :", traj.position(3.0))
    traj.tracer()
