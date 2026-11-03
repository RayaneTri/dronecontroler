% 1. Configurer l'environnement Python (.venv dans le dossier parent)
venv_python = fullfile(pwd, '..', '.venv', 'bin', 'python');
pyenv('Version', venv_python);

% 2. Choix du contrôleur
% true  : Modèle linéaire (LQR / LinearModel)
% false : MPC non-linéaire (RNN PyTorch)
use_linear_model = false; 

horizon_N = 10;

Q = 1;
R = 10;

disp('Initialisation du contrôleur pour le modèle linéaire (LQR)...');
linear_system = LinearModel(Q, R);

if use_linear_model
    mpc_system = linear_system; % Pour compatibilité si le bloc Simulink utilise mpc_system
else
    disp('Initialisation du contrôleur MPC (RNN PyTorch)...');
    model_path = 'model.ckpt';
    mpc_system = MPC(model_path, Q, R, horizon_N);
    %linear_system = mpc_system;
end

% 3. Définir une consigne (référence) pour l'horizon N
r_reference = ones(1, horizon_N) * 1.5; 

% 4. Calculer la commande
commande_u = mpc_system.step(r_reference);

disp('Commande calculée :');
disp(commande_u);