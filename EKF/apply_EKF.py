#site utilisé pour squelette EKF, author: Addison Sears-Collins: https://automaticaddison.com/extended-kalman-filter-ekf-with-python-code-example/
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

#Importation des données du NN (chemin relatif au fichier, indépendant du dossier de lancement)
NN_DIR = Path(__file__).resolve().parent.parent / "export_NN"

def load_matlab_array(filename):
    """Lit un fichier au format matrice MATLAB ([a b; c d]) et renvoie un array numpy 2D."""
    text = (NN_DIR / filename).read_text().strip().strip("[]")
    return np.array([[float(x) for x in row.split()] for row in text.split(";")])

bxu = load_matlab_array("bias_h.txt").ravel() # bxu : hidden layer bias vector (n_x,)
bf = load_matlab_array("head_bias.txt").ravel() # bf : output layer bias (n_y,)
Wf = load_matlab_array("head_weight.txt") # Wf : hidden to output layer weights
Wx = load_matlab_array("weight_hh_l0.txt") # Wx : hidden to hidden layer weights
Wu = load_matlab_array("weight_ih_l0.txt") # Wu : input to hidden layer weights

n_x = Wx.shape[0] # dimension de l'état caché

u_mean = -0.0023 #à importer de rnn
y_mean = -9.7708 #à importer de rnn
u_std = 0.5731 #à importer de rnn
y_std = 12.5953 #à importer de rnn

# Données d'experience :
DATA_DIR = Path(__file__).resolve().parent.parent
bdd_in = DATA_DIR / "bdd_in_mat_05_postfiltre.csv"
bdd_out = DATA_DIR / "bdd_out_mat_05_postfiltre.csv"

# Supress scientific notation when printing NumPy arrays
np.set_printoptions(precision=3,suppress=True)

# State model noise covariance matrix Q_k
# When Q is large, the Kalman Filter tracks large changes in
# the sensor measurements more closely than for smaller Q.
Q_k = np.eye(n_x)

# Sensor measurement noise covariance matrix R_k
# If we are sure about the measurements, R will be near zero.
R_k = np.eye(bf.size)

# Measurement matrix H_k
H_k = Wf

def getF(x, u):
    """
    Calculates and returns the Jacobian F matrix (Jacobian of F according to x)
    """
    z = Wx @ x + Wu @ np.atleast_1d(u) + bxu
    return np.diag(1 - np.tanh(z)**2) @ Wx
 
def getB(x, u):
    """
    Calculates and returns the Jacobian B matrix (Jacobian of F according to u)
    """
    z = Wx @ x + Wu @ np.atleast_1d(u) + bxu
    return np.diag(1 - np.tanh(z)**2) @ Wu

def ekf(y_k, x_hat_k_minus_1,
        u_k_minus_1, P_k_minus_1):
    """
    Extended Kalman Filter at step k
         
    INPUT
        :param y_k
        :param x_hat_k_minus_1 The state estimate at time k-1
        :param u_k_minus_1 The control vector applied at time k-1
        :param P_k_minus_1 The state covariance matrix estimate at time k-1
        :param dk Time interval in seconds
             
    OUTPUT
        :return x_hat_k near-optimal state estimate at time k 
        :return P_k state covariance_estimate for time k           
    """

    ######################### Predict #############################
    # Predict the state estimate at time k based on the state
    # estimate at time k-1 and the control input applied at time k-1.
    # Noise is zero-mean, so it only appears through Q_k in P_k.
    x_hat_k = np.tanh(
                    Wx @ x_hat_k_minus_1 +
                    Wu @ np.atleast_1d(u_k_minus_1) +
                    bxu)

    print(f'State Estimate Before EKF={x_hat_k}')

    # Jacobian of the motion model, evaluated at the previous estimate
    F_k = getF(x_hat_k_minus_1, u_k_minus_1)

    # Predict the state covariance estimate based on the previous
    # covariance and some noise
    P_k = F_k @ P_k_minus_1 @ F_k.T + Q_k

    ################### Update (Correct) ##########################
    # Calculate the difference between the actual sensor measurements
    # at time k minus what the measurement model predicted
    # the sensor measurements would be for the current timestep k.
    # Noise is zero-mean, so it only appears through R_k in S_k.
    measurement_residual_y_k = y_k - (H_k @ x_hat_k + bf)

    print(f'Observation={y_k}')

    # Calculate the measurement residual covariance
    S_k = H_k @ P_k @ H_k.T + R_k

    # Calculate the near-optimal Kalman gain
    # We use pseudoinverse since some of the matrices might be non-square or singular.
    K_k = P_k @ H_k.T @ np.linalg.pinv(S_k)

    # Calculate an updated state estimate for time k
    x_hat_k = x_hat_k + (K_k @ measurement_residual_y_k)

    # Update the state covariance estimate for time k
    P_k = P_k - (K_k @ H_k @ P_k)

    # Print the best (near-optimal) estimate of the current state of the robot
    print(f'State Estimate After EKF={x_hat_k}')

    # Return the updated state and covariance estimates
    return x_hat_k, P_k

class EKF:
    def __init__(self,model_path,model_nn):
        self.model = model_nn.load_from_checkpoint(model_path, map_location="cpu")
        self.model.eval()
        self.model.requires_grad_(False)

        self.u_m, self.u_s = torch.tensor(-0.0023).item(), torch.tensor(0.5731).item()
        self.y_m, self.y_s = torch.tensor(-9.7708).item(), torch.tensor(12.5953).item()

def main():
    """
    Boucle EKF de test
    """

    # List of input / outputs
    u = np.loadtxt(bdd_in, delimiter=",")
    y = np.loadtxt(bdd_out, delimiter=",")

    # The estimated state vector at time k-1 in the global reference frame.
    x_hat_k_minus_1 = np.zeros(n_x)

    # State covariance matrix P_k_minus_1
    P_k_minus_1 = np.eye(n_x)*0.1

    # Une ligne = un essai, une colonne = un pas de temps
    essai = 0

    # Sortie estimée (unités physiques) et son écart-type
    y_hat = []
    sigma_y_hat = []

    # Simulation du RNN seul (boucle ouverte, sans correction par les mesures)
    x_rnn = np.zeros(n_x)
    y_rnn = []

    # Start at k=1
    for k in range(1, y.shape[1]):

        # Print the current timestep
        # print(f'Timestep k={k}')

        # Run the Extended Kalman Filter and store the near-optimal state and covariance estimates
        optimal_x_hat_k, P_hat_k = ekf(
            (y[essai, k] - y_mean) / y_std, # Most recent sensor measurement (normalisée)
            x_hat_k_minus_1, # Our most recent estimate of the state
            (u[essai, k-1] - u_mean) / u_std, # Control input applied at k-1 (normalisé)
            P_k_minus_1) # State covariance at k-1

        # Dé-normalisation de la sortie estimée et de sa variance
        y_hat_k = ((H_k @ optimal_x_hat_k + bf) * y_std + y_mean).item()
        var_y_hat_k = (H_k @ P_hat_k @ H_k.T).item() * y_std**2
        y_hat.append(y_hat_k)
        sigma_y_hat.append(np.sqrt(var_y_hat_k))
        # print(f'Estimated output={y_hat_k:.3f} (mesure={y[essai, k]:.3f})')

        # Pas du RNN seul : même prédiction que l'EKF, mais sans mise à jour
        x_rnn = np.tanh(Wx @ x_rnn + Wu @ np.atleast_1d((u[essai, k-1] - u_mean) / u_std) + bxu)
        y_rnn.append(((H_k @ x_rnn + bf) * y_std + y_mean).item())

        # Get ready for the next timestep by updating the variable values
        x_hat_k_minus_1 = optimal_x_hat_k
        P_k_minus_1 = P_hat_k

    # Tracé y mesuré vs y estimé
    y_hat = np.array(y_hat)
    sigma_y_hat = np.array(sigma_y_hat)
    t = np.arange(1, y.shape[1])
    plt.figure()
    plt.plot(t, y[essai, 1:], label="y mesuré")
    plt.plot(t, y_hat, label="y estimé (EKF)")
    plt.fill_between(t, y_hat - sigma_y_hat, y_hat + sigma_y_hat, alpha=0.3, label="±σ")
    plt.plot(t, y_rnn, "--", label="y prédit (RNN seul)")
    plt.xlabel("k")
    plt.ylabel("y")
    plt.title(f"EKF - essai {essai}")
    plt.legend()
    plt.show()