"""
Commande du système en position
"""

import time
import numpy as np

class COMMANDE:
    """
    Trajectoire polynomiale par morceaux en position
    Commandée par :
        temps : instants de passage, positions : positions aux instants de passage
        vitesses : vitesses aux instants de passage (nulles par défaut)
        acc_initiale : accélération en temps[0]
        Ts : période d'échantillonnage
        borner : si True, raccourcit les segments qui font sortir z de [min(positions), max(positions)]
    """
    def __init__(self,temps:list,positions:list,Ts:float,vitesses:list=None,acc_initiale=0.0,borner:bool=False):
        self.temps = list(temps)
        self.positions = positions
        self.Ts = Ts
        self.vitesses = vitesses if vitesses is not None else [0]*len(temps)
        self.acc_initiale = acc_initiale
        self.coefs, self.bornes = self.coefficients()
        if borner:
            self.borner_trajectoire()

    @staticmethod
    def polynome(coef:list,t:float):
        """
        Renvoie la réponse du polynome de coefficients coef (ordre croissant) en t
        """
        return sum(c*t**i for i, c in enumerate(coef))

    @staticmethod
    def conv_pos_acceleration(coef,T):
        """
        Renvoie l'accélération z''(T) du polynome coef
        """
        return 2*coef[2] + 6*coef[3]*T + 12*coef[4]*T**2 + 20*coef[5]*T**3

    @staticmethod
    def coefficients_segment(T,za,zb,va,vb,aa,ab):
        """
        rends les coefs du polynome de degrée 5 avec z, z' et z'' imposés en 0 et en T (jerk libre)
        """
        R = zb - za - va*T - aa*T**2/2
        S = vb - va - aa*T
        U = ab - aa
        A = np.array([[T**3,T**4,T**5],[3*T**2,4*T**3,5*T**4],[6*T,12*T**2,20*T**3]])
        a3, a4, a5 = np.linalg.solve(A,[R,S,U])
        return [za, va, aa/2, a3, a4, a5]

    def depassement(self):
        """
        Renvoie le dépassement maximal de la trajectoire hors de [min(positions), max(positions)]
        """
        z = np.array(self.generation_traj())
        return max(min(self.positions) - z.min(), z.max() - max(self.positions), 0.0)

    def borner_trajectoire(self,tol=1e-3,max_iter=50):
        """
        Raccourcit (facteur 0.9) les segments dépassant les bornes, sans changer les autres instants de passage
        """
        for _ in range(max_iter):
            if self.depassement() <= tol:
                return
            z = np.array(self.generation_traj())
            k = int(np.argmax(np.maximum(min(self.positions) - z, z - max(self.positions))))
            t_k = self.instants()[k]
            i = max(j for j in range(len(self.temps)-1) if self.temps[j] <= t_k)
            nouveau = self.temps[i] + 0.9*(self.temps[i+1] - self.temps[i])
            if i+1 == len(self.temps)-1 or nouveau - self.temps[i] < self.Ts:
                break
            self.temps[i+1] = nouveau
            self.coefs, self.bornes = self.coefficients()

    def coefficients(self):
        """
        rends (coefs, bornes), deux listes de même dimension : coefs[i] est le polynome du segment [temps[i],temps[i+1]]
        (temps local) et bornes[i] = (temps[i], temps[i+1]) ses bornes en temps, tels que :
        z(temps[i]) = positions[i], z'(temps[i]) = vitesses[i], z''(temps[0]) = acc_initiale
        Le premier segment part de acc_initiale, chaque segment arrive avec une accélération nulle
        (donc l'accélération est continue) et un jerk libre (degrée 5). Entre deux waypoints à vitesse nulle,
        un segment après le premier est monotone : pas de dépassement des positions de passage.
        """
        coefs = []
        bornes = []
        acc = self.acc_initiale
        for i in range(len(self.temps)-1):
            T = self.temps[i+1] - self.temps[i]
            coef = self.coefficients_segment(T,self.positions[i],self.positions[i+1],
                                                        self.vitesses[i],self.vitesses[i+1],acc,0.0)
            coefs.append(coef)
            bornes.append((self.temps[i],self.temps[i+1]))
            acc = 0.0
        return coefs, bornes

    def echantillonnage_segment(self,coef,t_debut,k_debut,k_fin):
        """
        Évalue le polynome d'un segment (commençant en t_debut) aux instants Ts*k pour k de k_debut à k_fin inclus
        """
        return [self.polynome(coef,self.Ts*k - t_debut) for k in range(k_debut,k_fin+1)]

    def echantillonnage(self,coefs):
        """
        Échantillonne entièrement chaque segment l'un après l'autre de t=temps[0] à t=temps[-1]
        """
        valeurs = []
        k_debut = round(self.temps[0]/self.Ts)
        for i, coef in enumerate(coefs):
            k_fin = round(self.temps[i+1]/self.Ts)
            valeurs += self.echantillonnage_segment(coef,self.temps[i],k_debut,k_fin)
            k_debut = k_fin + 1
        return valeurs

    def generation_traj(self):
        """
        Génère la trajectoire polynomiale en position échantillonnée de temps[0] à temps[-1]
        """
        return self.echantillonnage(self.coefs)

    def generation_acc(self):
        """
        Donne la commande en acceleration pour suivre la trajectoire definie
        (dérivée seconde des polynomes de position de degré 5 au plus)
        """
        coefs_acc = [[2*c[2],6*c[3],12*c[4],20*c[5]] for c in self.coefs]
        return self.echantillonnage(coefs_acc)

    def coefficients_a(self,t):
        """
        Renvoie (coef, t_debut) : les coefs du segment actif à l'instant t et son instant de début (les coefs sont en temps local)
        Avant le début / après la fin, renvoie le premier / dernier segment
        """
        i = len(self.bornes) - 1
        for j, (_, fin) in enumerate(self.bornes):
            if t < fin:
                i = j
                break
        return self.coefs[i], self.bornes[i][0]
    
    def position(self,t):
        """
        Renvoie la position à l'instant t
        Avant temps[0] / après temps[-1], renvoie la première / dernière position de passage
        """
        if t <= self.temps[0]:
            return self.positions[0]
        if t >= self.temps[-1]:
            return self.positions[-1]
        coef, t_debut = self.coefficients_a(t)
        return self.polynome(coef,t - t_debut)

    def acceleration(self,t):
        """
        Renvoie l'acceleration à l'instant t
        Hors de [temps[0], temps[-1]], la position est tenue : accélération nulle
        """
        if t < self.temps[0] or t >= self.temps[-1]:
            return 0.0
        coef, t_debut = self.coefficients_a(t)
        return self.conv_pos_acceleration(coef,t - t_debut)

    def instants(self):
        """
        Renvoie les instants d'échantillonnage correspondant aux sorties échantillonnées
        """
        k_debut = round(self.temps[0]/self.Ts)
        k_fin = round(self.temps[-1]/self.Ts)
        return [self.Ts*k for k in range(k_debut,k_fin+1)]

    def tracer(self):
        """
        Trace la trajectoire de référence et la commande en accélération
        """
        import matplotlib.pyplot as plt

        pos = self.generation_traj()
        acc = self.generation_acc()
        t = self.instants()

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.plot(t, pos)
        ax1.plot(self.temps, self.positions, 'o')
        ax1.set_ylabel("position z (m)")
        ax1.grid()
        ax2.plot(t, acc)
        ax2.set_ylabel("accélération (m/s²)")
        ax2.set_xlabel("temps (s)")
        ax2.grid()
        fig.suptitle("Trajectoire de référence")
        plt.show()

if __name__ == "__main__":
    commande = COMMANDE(temps=[0, 5, 10], positions=[0, 0.5, -0.2], Ts=0.05, acc_initiale=-9.81, borner=True)
    t0 = time.perf_counter()
    commande.generation_acc()
    print(f"Temps de calcul de la génération d'accélération : {(time.perf_counter() - t0)*1e3:.3f} ms")
    commande.tracer()


_commande_simulink = None
def f(t):
    """Return the acceleration command at time ``t`` for Simulink."""
    global _commande_simulink
    if _commande_simulink is None:
        _commande_simulink = COMMANDE(
            temps=[0, 5, 10],
            positions=[0, 0.5, -0.2],
            Ts=0.05,
            acc_initiale=-9.81,
            borner=True,
        )
    return float(_commande_simulink.acceleration(float(t)))
