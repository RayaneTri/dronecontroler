import numpy as np
import matplotlib.pyplot as plt


u = np.loadtxt("bdd_in_mat_05.csv", delimiter=",")
y = np.loadtxt("bdd_out_mat_05.csv", delimiter=",")

Ts = 0.05
t = np.arange(u.shape[1]) * Ts

i = 10
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(t, u[i])
ax1.set_ylabel("entrée u")
ax2.plot(t, y[i])
ax2.set_ylabel("sortie y")
ax2.set_xlabel("temps (s)")
plt.show()
