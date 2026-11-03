% 1. Configurer l'environnement Python (.venv dans le dossier parent)
venv_python = fullfile(pwd, '..', '.venv', 'bin', 'python');
pyenv('Version', venv_python);

% 2. Instancier le contrôleur depuis MATLAB
model_path = 'model.ckpt';
horizon_N = 20;
mpc_system = MPC(model_path, horizon_N);

% 3. Définir une consigne (référence) pour l'horizon N
r_reference = ones(1, horizon_N) * 1.5; 

% 4. Calculer la commande optimale via PyTorch
commande_u = mpc_system.step(r_reference);

disp('Commande calculée :');
disp(commande_u);