random_init ;

% 1. Configurer l'environnement Python (.venv dans le dossier parent)
script_dir = fileparts(mfilename('fullpath'));
repo_dir = fileparts(script_dir);

if ispc
    python_executable = fullfile(repo_dir, '.venv', 'Scripts', 'python.exe');
elseif ismac || isunix
    python_executable = fullfile(repo_dir, '.venv', 'bin', 'python');
else
    error('main_simulation:UnsupportedOS', ...
        'Système d''exploitation non pris en charge.');
end

if ~isfile(python_executable)
    error('main_simulation:MissingPython', ...
        'Interpréteur Python introuvable : %s', python_executable);
end

pyenv('Version', python_executable);

addpath(script_dir);
python_path = py.sys.path;
if count(python_path, script_dir) == 0
    insert(python_path, int32(0), script_dir);
end
commande_system = Commande( ...
    [0, 5, 10], ...
    [0, 0.5, -0.2], ...
    0.05, ...
    [], ...
    -9.81, ...
    true);

% 2. Choix du contrôleur
% true  : Modèle linéaire (LQR / LinearModel)
% false : MPC non-linéaire (RNN PyTorch)
use_linear_model = false; 

horizon_N = 20;

Q = 10;
R = 1;

V = 1;
W = 1;
ekf = true;

disp('Initialisation du contrôleur pour le modèle linéaire (LQR)...');
linear_system = LinearModel(Q, R);

if use_linear_model
    mpc_system = linear_system; % Pour compatibilité si le bloc Simulink utilise mpc_system
else
    disp('Initialisation du contrôleur MPC (RNN PyTorch)...');
    model_path = fullfile(script_dir, 'model.ckpt');
    if ~isfile(model_path)
        error('main_simulation:MissingModel', ...
            'Modèle RNN introuvable : %s', model_path);
    end
    mpc_system = MPC(model_path, Q, R, ekf, horizon_N, V, W);
    %linear_system = mpc_system;
end

% 3. Définir une consigne (référence) pour l'horizon N
%r_reference = ones(1, horizon_N) * 1.5; 

% 4. Calculer la commande
%commande_u = mpc_system.step(r_reference);

%disp('Commande calculée :');
%disp(commande_u);