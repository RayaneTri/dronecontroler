#Non utilisé car lent
import numpy as np
import sympy as sp
from pathlib import Path

#Importation des données du NN (chemin relatif au fichier, indépendant du dossier de lancement)
NN_DIR = Path(__file__).resolve().parent.parent / "export_NN"

def load_matlab_array(filename):
    """Lit un fichier au format matrice MATLAB ([a b; c d]) et renvoie un array numpy 2D."""
    text = (NN_DIR / filename).read_text().strip().strip("[]")
    return np.array([[float(x) for x in row.split()] for row in text.split(";")])

bxu = load_matlab_array("bias_h.txt") # bxu : hidden layer bias vector
bf = load_matlab_array("head_bias.txt") # bf : output layer bias
Wf = load_matlab_array("head_weight.txt") # Wf : hidden to output layer weights
Wx = load_matlab_array("weight_hh_l0.txt") # Wx : hidden to hidden layer weights
Wu = load_matlab_array("weight_ih_l0.txt") # Wu : input to hidden layer weights

u_mean = -0.0023 #à importer de rnn
y_mean = -9.7708 #à importer de rnn
u_std = 0.5731 #à importer de rnn
y_std = 12.5953 #à importer de rnn

n = Wx.shape[0]   # 16 états cachés
m = Wu.shape[1]   # 1 entrée

# Variables symboliques
x = sp.Matrix(sp.symbols(f'x0:{n}'))
u = sp.Matrix(sp.symbols(f'u0:{m}'))

# f(x,u) = tanh(Wx x + Wu u + bxu)
f = (sp.Matrix(Wx) * x + sp.Matrix(Wu) * u
     + sp.Matrix(bxu).reshape(n, 1)).applyfunc(sp.tanh)

# Jacobiennes symboliques, converties en fonctions NumPy
F_x = sp.lambdify((x, u), f.jacobian(x), 'numpy')   # 16x16
F_u = sp.lambdify((x, u), f.jacobian(u), 'numpy')   # 16x1
