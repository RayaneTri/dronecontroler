"""
Commande du système en position
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev as C
from scipy.linalg import null_space
from scipy.optimize import minimize

class COMMANDE:
    """
    Trajectoire polynomiale par morceaux en position
    Commandée par :
        temps : instants de passage, positions : positions aux instants de passage
        vitesses : vitesses aux instants de passage (nulles par défaut)
        acc_initiale : accélération en temps[0]
        Ts : période d'échantillonnage
        borner : si True, z reste dans [min(positions), max(positions)]
        degre_max : degré maximal des polynomes, tol : dépassement toléré (m)
    Chaque segment [temps[i],temps[i+1]] dure tout son intervalle et minimise l'intégrale de z''² :
    le degré augmente tant que les bornes ne sont pas respectées ou que les accélérations diminuent encore.
    """
    def __init__(self,temps:list,positions:list,Ts:float,vitesses:list=None,acc_initiale=0.0,borner:bool=False,
                 degre_max:int=,tol:float=1e-3):
        self.temps = list(temps)
        self.positions = positions
        self.Ts = Ts
        self.vitesses = vitesses if vitesses is not None else [0]*len(temps)
        self.acc_initiale = acc_initiale
        self.borner = borner
        self.degre_max = degre_max
        self.tol = tol
        self.coefs, self.bornes = self.coefficients()

    @staticmethod
    def matrices_cheb(T,n,t):
        """
        Renvoie [V0,V1,V2] : V_m @ c est la dérivée m-ième, aux instants locaux t de [0,T],
        du polynome de degré n de coefficients c dans la base de Chebyshev sur [0,T]
        """
        V0 = C.chebvander(2*np.asarray(t,dtype=float)/T - 1,n)
        D = np.vstack([C.chebder(np.eye(n+1),axis=0)*2/T, np.zeros((1,n+1))])  # dérivation dans la base
        return [V0, V0 @ D, V0 @ D @ D]

    def coefficients_segment(self,T,za,zb,va,vb,aa,ab,n):
        """
        Polynome de degré n minimisant l'intégrale de z''² sur [0,T] avec z, z' et z'' imposés en 0 et en T
        (et z dans les bornes si borner). Renvoie (polynome, coût) ou None si les bornes ne sont pas respectées
        """
        E = np.vstack(self.matrices_cheb(T,n,[0,T]))
        d = np.array([za,zb,va,vb,aa,ab])
        c0 = np.linalg.lstsq(E,d,rcond=None)[0]
        N = null_space(E)
        # coût exact par quadrature de Gauss-Legendre
        x_g, w_g = np.polynomial.legendre.leggauss(n+1)
        A2 = self.matrices_cheb(T,n,(x_g+1)*T/2)[2]
        Q = A2.T @ (A2*(w_g*T/2)[:,None])
        H, g = N.T @ Q @ N, N.T @ Q @ c0
        y = -np.linalg.lstsq(H,g,rcond=None)[0] if N.shape[1] else np.zeros(0)
        z_min, z_max = min(self.positions), max(self.positions)
        if self.borner and N.shape[1]:
            # contraintes sur une grille de Chebyshev (resserrée aux extrémités)
            V = self.matrices_cheb(T,n,T*(1 - np.cos(np.pi*np.linspace(0,1,20*n)))/2)[0]
            Vc0, VN = V @ c0, V @ N
            z = Vc0 + VN @ y
            if z.min() < z_min or z.max() > z_max:  # sinon la solution sans contrainte convient déjà
                y = minimize(lambda y: (y @ H @ y + 2*g @ y), y, jac=lambda y: 2*(H @ y + g), method='SLSQP',
                             constraints=[{'type':'ineq','fun':lambda y: np.concatenate([z_max - Vc0 - VN@y, Vc0 + VN@y - z_min]),
                                           'jac':lambda y: np.vstack([-VN, VN])}],
                             options={'maxiter':500,'ftol':1e-12}).x
        c = c0 + N @ y
        p = C.Chebyshev(c,domain=[0,T])
        if self.borner:
            z = p(np.linspace(0,T,max(2000,round(10*T/self.Ts))))
            if max(z_min - z.min(), z.max() - z_max) > self.tol:
                return None
        return p, c @ Q @ c

    def depassement(self):
        """
        Renvoie le dépassement maximal de la trajectoire hors de [min(positions), max(positions)]
        """
        z = np.array(self.generation_traj())
        return max(min(self.positions) - z.min(), z.max() - max(self.positions), 0.0)

    def coefficients(self):
        """
        rends (coefs, bornes), deux listes de même dimension : coefs[i] est le polynome (Chebyshev, temps local) du segment
        [temps[i],temps[i+1]] et bornes[i] = (temps[i], temps[i+1]), tels que :
        z(temps[i]) = positions[i], z'(temps[i]) = vitesses[i], z''(temps[0]) = acc_initiale
        Le premier segment part de acc_initiale, chaque segment arrive avec une accélération nulle
        (donc l'accélération est continue). Le degré part de 5 et augmente tant que le segment sort des bornes
        ou que le coût (intégrale de z''²) baisse de plus de 1 %.
        """
        coefs = []
        bornes = []
        acc = self.acc_initiale
        for i in range(len(self.temps)-1):
            T = self.temps[i+1] - self.temps[i]
            meilleur = None
            for n in range(5,self.degre_max+1):
                sol = self.coefficients_segment(T,self.positions[i],self.positions[i+1],
                                                self.vitesses[i],self.vitesses[i+1],acc,0.0,n)
                if sol is None:
                    continue
                if meilleur is not None and sol[1] > 0.99*meilleur[1]:
                    break
                meilleur = sol
            if meilleur is None:
                raise ValueError(f"segment {i} : bornes non respectées jusqu'au degré {self.degre_max}")
            coefs.append(meilleur[0])
            bornes.append((self.temps[i],self.temps[i+1]))
            acc = 0.0
        return coefs, bornes

    def echantillonnage_segment(self,coef,t_debut,k_debut,k_fin):
        """
        Évalue le polynome d'un segment (commençant en t_debut) aux instants Ts*k pour k de k_debut à k_fin inclus
        """
        return list(coef(self.Ts*np.arange(k_debut,k_fin+1) - t_debut))

    def echantillonnage(self,coefs):
        """
        Échantillonne entièrement chaque segment l'un après l'autre de t=temps[0] à t=temps[-1]
        """
        valeurs = []
        k_debut = round(self.temps[0]/self.Ts)
        for i, coef in enumerate(coefs):
            t_debut, t_fin = self.bornes[i]
            k_fin = round(t_fin/self.Ts)
            valeurs += self.echantillonnage_segment(coef,t_debut,k_debut,k_fin)
            k_debut = max(k_debut, k_fin + 1)
        return valeurs

    def generation_traj(self):
        """
        Génère la trajectoire polynomiale en position échantillonnée de temps[0] à temps[-1]
        """
        return self.echantillonnage(self.coefs)

    def generation_acc(self):
        """
        Donne la commande en acceleration pour suivre la trajectoire definie
        (dérivée seconde des polynomes de position)
        """
        return self.echantillonnage([c.deriv(2) for c in self.coefs])

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
    
    def acceleration(self,t):
        """
        Renvoie l'acceleration à l'instant t
        """
        coef, t_debut = self.coefficients_a(t)
        return float(coef.deriv(2)(t - t_debut))

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
    commande = COMMANDE(temps=[0, 10, 20], positions=[0, 0.5, -0.2], Ts=0.05, acc_initiale=-9.81, borner=True)
    t0 = time.perf_counter()
    commande.generation_acc()
    print(f"Temps de calcul de la génération d'accélération : {(time.perf_counter() - t0)*1e3:.3f} ms")
    commande.tracer()
