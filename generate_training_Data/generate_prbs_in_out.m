% Génère un jeu de données entrée/sortie en excitant le drone par APRBS
%
% experience_nbr simulations, chacune avec un tirage APRBS différent (seed = numéro
% de l'expérience). Tout est sauvé dans data/prbs_data.mat (v7.3) :
%   inputs  : experience_nbr x N x 4  (U1..U4)
%   outputs : experience_nbr x N x 12 (phi..w_dot_z, sans positions ni vitesses)
% une ligne = une expérience, une colonne = un pas de temps Ts (temps non sauvé).
here = fileparts(mfilename('fullpath'));
addpath(here);                                  % system_simulation_training_creator.slx
addpath(fullfile(here, '..', 'Matlab files'));  % random_init.m, UAV_model.slx

random_init;                        % définit Ts = 0.05 et les graines du bruit

experience_nbr = 3000;
batch_size     = 100;               % simulations en mémoire avant écriture
N              = 2000;              % échantillons par expérience (comme bdd_in_mat)
vary_noise     = true;              % nouvelles graines de bruit à chaque expérience
                                    % (false : plus rapide, active le Fast Restart)

T_end   = (N - 1) * Ts;             % t = 0:Ts:T_end donne N échantillons
U_hover = 0.55;
amp     = [0.02 0.01 0.01 0.04];    % [dT dr dp dy], dy plus grand
opts    = struct('seed', 1, 't_start', 2, 'hold_min', 1, 'hold_max', 3, 'ms_amp', [0 0 0 0], 'f_max', 2);

in_names  = {'U1', 'U2', 'U3', 'U4'};
out_names = {'x', 'y', 'z', 'vx', 'vy', 'vz', 'phi', 'theta', 'psi', ...
             'wx', 'wy', 'wz', 'ax', 'ay', 'az', 'w_dot_x', 'w_dot_y', 'w_dot_z'};

% simout = Mux(all_outputs, From Workspace) : 18 sorties puis 4 entrées
col_names = [out_names in_names];
in_cols   = 19:22;
out_cols  = 7:18;                   % on ne garde pas x, y, z, vx, vy, vz

data_dir = fullfile(here, 'data');
if ~exist(data_dir, 'dir'), mkdir(data_dir); end
mat_file = fullfile(data_dir, 'prbs_data.mat');
if exist(mat_file, 'file'), delete(mat_file); end
m = matfile(mat_file, 'Writable', true);
m.in_names  = in_names;
m.out_names = out_names(out_cols);
m.Ts = Ts; m.U_hover = U_hover; m.amp = amp; m.opts = opts;

mdl = 'system_simulation_training_creator';
load_system(mdl);
seed_names = who('seed_*');         % graines des blocs de bruit (random_init)

for b0 = 1:batch_size:experience_nbr
    exp_idx = b0:min(b0 + batch_size - 1, experience_nbr);
    n_exp = numel(exp_idx);

    clear in
    in(1:n_exp) = Simulink.SimulationInput(mdl);
    for j = 1:n_exp
        opts.seed = exp_idx(j);
        [~, ~, d, U_ts] = generate_prbs(U_hover, amp, T_end, Ts, opts);
        if exp_idx(j) == 1, d_first = d; end      % pour l'affichage
        in(j) = in(j).setVariable('U_ts', U_ts);
        in(j) = in(j).setModelParameter('StopTime', num2str(T_end));
        if vary_noise
            for s = 1:numel(seed_names)
                in(j) = in(j).setVariable(seed_names{s}, randi(10000));
            end
        end
    end
    out = sim(in(1:n_exp), 'UseFastRestart', matlab.lang.OnOffSwitchState(~vary_noise), ...
              'ShowProgress', 'off');

    % Bloc (n_exp x N x signal), ajouté à la suite dans le .mat
    rows = zeros(n_exp, N, numel(col_names), 'single');
    for j = 1:n_exp
        if ~isempty(out(j).ErrorMessage)
            error('Expérience %d : %s', exp_idx(j), out(j).ErrorMessage);
        end
        data = out(j).simout.signals.values;      % N x 22
        assert(isequal(size(data), [N numel(col_names)]), ...
               'Expérience %d : simout de taille %s', exp_idx(j), mat2str(size(data)));
        rows(j, :, :) = reshape(data, 1, N, []);
    end
    if b0 == 1
        m.inputs  = rows(:, :, in_cols);
        m.outputs = rows(:, :, out_cols);
    else
        m.inputs(exp_idx, 1:N, 1:numel(in_cols))   = rows(:, :, in_cols);
        m.outputs(exp_idx, 1:N, 1:numel(out_cols)) = rows(:, :, out_cols);
    end

    if b0 == 1
        Y_first = squeeze(rows(1, :, 1:18));      % pour l'affichage
        U_first = squeeze(rows(1, :, 19:22));
    end
    fprintf('%d / %d expériences\n', exp_idx(end), experience_nbr);
end

% Affichage de la première expérience
tY = (0:N-1)' * Ts;
disp(std(Y_first(:, 16:18)))        % ṗ, q̇, ṙ : les trois valeurs doivent être du même ordre
disp(corrcoef(d_first(tY >= opts.t_start, :)))   % hors diagonale proche de 0 : modes décorrélés

% Affichage des modes (c'est sur eux que portent hold_min / hold_max)
figure('Name', 'Modes');
mode_names = {'\delta_T', '\delta_r', '\delta_p', '\delta_y'};
for k = 1:4
    subplot(4, 1, k);
    stairs(tY, d_first(:, k)); grid on; ylabel(mode_names{k});
end
xlabel('t (s)'); sgtitle('Modes (expérience 1)');

figure('Name', 'Entrées');
stairs(tY, U_first); legend(in_names); grid on;
xlabel('t (s)'); title('Entrées U1..U4 (expérience 1)');

groups = {'Position', 'Vitesse', 'Angles', 'Vitesses angulaires', ...
          'Accélérations', 'Accélérations angulaires'};
figure('Name', 'Sorties');
for g = 1:6
    idx = 3 * (g - 1) + (1:3);
    subplot(3, 2, g);
    plot(tY, Y_first(:, idx)); legend(out_names(idx), 'Interpreter', 'none'); grid on;
    xlabel('t (s)'); title(groups{g});
end

function [t, U, delta, U_ts] = generate_prbs(U_hover, amp, T_end, Ts, opts)
% GENERATE_PRBS  Génère une excitation APRBS pour les 4 entrées du drone.
%
%   U1 = U_hover + dT + dr - dp + dy
%   U2 = U_hover + dT - dr - dp - dy
%   U3 = U_hover + dT - dr + dp + dy
%   U4 = U_hover + dT + dr + dp - dy
%
%   Un APRBS indépendant est tiré sur chaque mode [dT dr dp dy] : niveau
%   uniforme dans [-amp(k), amp(k)], maintenu pendant une durée aléatoire
%   dans [hold_min(k), hold_max(k)].
%   Optionnellement, un multisinus est ajouté sur chaque mode. Le mode k
%   utilise les harmoniques k, k+4, k+8, ... de f0 = 1/(T_end - t_start) :
%   les 4 jeux de fréquences sont disjoints, donc les modes sont orthogonaux.
%
% Entrées :
%   U_hover : commande de stationnaire (scalaire, trouvée par dichotomie)
%   amp     : amplitudes [aT ar ap ay] (prendre ay > ar, ap : yaw faible)
%   T_end   : durée totale (s)
%   Ts      : période d'échantillonnage (s)
%   opts    : (optionnel) struct avec les champs
%       .hold_min : durée de maintien min (s), scalaire ou 1x4 (défaut 0.2)
%       .hold_max : durée de maintien max (s), scalaire ou 1x4 (défaut 1.0)
%       .seed     : graine aléatoire (défaut 0)
%       .U_min    : saturation basse des U (défaut -Inf)
%       .U_max    : saturation haute des U (défaut +Inf)
%       .t_start  : durée initiale à U_hover sans excitation (défaut 0)
%       .ms_amp   : amplitude crête du multisinus par mode, 1x4 (défaut 0)
%       .f_max    : fréquence max du multisinus (Hz, défaut 2)
%
% Sorties :
%   t     : vecteur temps (N x 1)
%   U     : commandes (N x 4), colonnes U1..U4
%   delta : modes (N x 4), colonnes [dT dr dp dy]
%   U_ts  : timeseries de U, utilisable dans un bloc "From Workspace"
%
% Exemple :
%   [t, U, d, U_ts] = generate_prbs(0.55, [0.02 0.01 0.01 0.04], 60, 0.01);

    if nargin < 5
        opts = struct();
    end
    hold_min = get_opt(opts, 'hold_min', 0.2);
    hold_max = get_opt(opts, 'hold_max', 1.0);
    seed     = get_opt(opts, 'seed', 0);
    U_min    = get_opt(opts, 'U_min', -Inf);
    U_max    = get_opt(opts, 'U_max', Inf);
    t_start  = get_opt(opts, 't_start', 0);
    ms_amp   = get_opt(opts, 'ms_amp', zeros(1, 4));
    f_max    = get_opt(opts, 'f_max', 2);

    amp      = reshape(amp, 1, 4);
    if isscalar(hold_min), hold_min = repmat(hold_min, 1, 4); end
    if isscalar(hold_max), hold_max = repmat(hold_max, 1, 4); end

    rng(seed);

    t = (0:Ts:T_end)';
    N = numel(t);
    delta = zeros(N, 4);

    % Tirage APRBS indépendant sur chaque mode
    for k = 1:4
        n_min = max(1, round(hold_min(k) / Ts));
        n_max = max(n_min, round(hold_max(k) / Ts));
        i = 1;
        while i <= N
            n_hold = randi([n_min, n_max]);
            i_end = min(N, i + n_hold - 1);
            delta(i:i_end, k) = amp(k) * (2 * rand - 1);
            i = i_end + 1;
        end
    end

    % Multisinus : fréquences entrelacées entre les modes, phases aléatoires
    if isscalar(ms_amp), ms_amp = repmat(ms_amp, 1, 4); end
    f0 = 1 / (T_end - t_start);
    tau = t - t_start;
    for k = 1:4
        if ms_amp(k) == 0, continue; end
        h = k:4:floor(f_max / f0);
        ms = sum(cos(2 * pi * f0 * tau * h + 2 * pi * rand(1, numel(h))), 2);
        delta(:, k) = delta(:, k) + ms_amp(k) * ms / max(abs(ms));
    end

    % Phase initiale sans excitation (le drone se stabilise au hover)
    delta(t < t_start, :) = 0;

    % Matrice de mixage : lignes U1..U4, colonnes [dT dr dp dy]
    M = [1  1 -1  1;
         1 -1 -1 -1;
         1 -1  1  1;
         1  1  1 -1];

    U = U_hover + delta * M';
    U = min(max(U, U_min), U_max);

    U_ts = timeseries(U, t, 'Name', 'U_prbs');
end

function v = get_opt(opts, name, default)
    if isfield(opts, name)
        v = opts.(name);
    else
        v = default;
    end
end
