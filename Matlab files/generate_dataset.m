%% generate_dataset.m
% Génère Nseed simulations avec des consignes APRBS différentes et exporte
% dans un seul fichier dataset_all.mat (v7) :
%   REF (N, T, 4)  : consignes z_ref, phi_ref, theta_ref, psi_ref
%   Y   (N, T, k)  : colonnes de sortie choisies dans keep_names (ax, ay, az, z)
%
% Les simulations sont lancées PAR LOTS : la mémoire reste faible, les
% résultats sont sauvegardés après chaque lot, et un lancement interrompu
% reprend là où il s'était arrêté.
%
% Prérequis dans le modèle :
%   - bloc aprbs_ref dont l'entrée seed vient d'un Constant `aprbs_seed`
%     (variable définie dans le Model Workspace)
%   - To Workspace `ref_log` sur la sortie du bloc aprbs_ref (4 consignes)
%   - To Workspace `simout`  sur all_outputs
%   - Save format = Timeseries, Sample time = 0.05 pour ces deux blocs
%   - Model Settings > Data Import/Export : "Single simulation output" coché
%   - pas de solveur fixe 0.005, scopes et blocs MPC commentés (plus rapide)

% Pas de "clear" général : le modèle a besoin de variables du workspace de base
% (Ts, g_noise, seed_...) définies par random_init.
clear in out REF Y SEED REF_all Y_all valid done rec run t ref y seeds

%% Paramètres
init_script = 'random_init';  % script qui définit Ts, g_noise et les seed_...
mdl    = 'system_simulation_FC';
Nseed      = 3000;
seed0      = 1000;
batch_size = 500;        % seeds utilisées : seed0+1 ... seed0+Nseed (reproductible)
T_sim  = 100;         % durée d'une simulation [s] (< 128 s : pas de répétition de l'APRBS)
T_warm = 3;           % début jeté (démarrage moteurs), = T_warm du bloc aprbs_ref
dt     = 0.05;        % pas d'échantillonnage du dataset [s] (20 Hz, comme le MPCC)
outdir = 'dataset';
% Noms des colonnes (ordre de all_outputs dans UAV_model)
ref_names = {'z_ref','phi_ref','theta_ref','psi_ref'};
y_names   = {'x','y','z','vx','vy','vz','ax','ay','az', ...
             'phi','theta','psi','wx','wy','wz','w_dot_x','w_dot_y','w_dot_z'};
keep_names = {'ax','ay','z'};   % colonnes de y exportées dans Y

% Variables de graine du bruit capteur (fixées à partir de la seed du run)
noise_vars = {'seed_x','seed_y','seed_z','seed_vx','seed_vy','seed_vz', ...
              'seed_ax','seed_ay','seed_az','seed_phi','seed_theta','seed_psi', ...
              'seed_wx','seed_wy','seed_wz','seed_w_dot_x','seed_w_dot_y','seed_w_dot_z'};

angle_max = 1.0;      % [rad] au-delà, le run est rejeté (drone perdu)

% Vitesse de génération
sim_mode     = 'accelerator';  % 'accelerator' (compilé, plus rapide) ou 'normal'
fast_restart = true;           % compile le modèle une seule fois par lot
use_parallel = true;           % parsim sur tous les cœurs si la Parallel Computing Toolbox est là
n_workers    = 4;              % nb de workers parallèles (chacun prend ~1 à 2 Go de RAM)

%% Préparation
warning('off', 'Simulink:Engine:InputNotConnected');
warning('off', 'Simulink:Engine:OutputNotConnected');

% ATTENTION : random_init ne doit pas contenir de "clear" (même workspace que ce script).
evalin('base', init_script);
if ~exist(outdir, 'dir'), mkdir(outdir); end
load_system(mdl);

seeds = seed0 + (1:Nseed);
t     = (T_warm:dt:T_sim).';
T     = numel(t);
i_ang = find(ismember(y_names, {'phi','theta'}));
[~, i_keep] = ismember(keep_names, y_names);
if any(i_keep == 0), error('keep_names contient un nom absent de y_names.'); end

has_pct = use_parallel && license('test', 'Distrib_Computing_Toolbox') && ~isempty(ver('parallel'));
fr = 'off'; if fast_restart, fr = 'on'; end
if has_pct
    pool = gcp('nocreate');
    if isempty(pool) || pool.NumWorkers ~= n_workers
        delete(pool);  parpool(n_workers);
    end
end
fast_ok = true;   % passe à false si le mode rapide est refusé une fois

% Préparation de chaque worker parallèle : mêmes dossiers dans le path que toi
% (fonctions et classes appelées par les blocs MATLAB Function), puis random_init.
proj_dir  = pwd;
setup_fcn = @() evalin('base', sprintf('addpath(genpath(''%s'')); %s', proj_dir, init_script));

%% Reprise d'un lancement interrompu
partial_file = fullfile(outdir, 'dataset_partial.mat');
cfg = struct('Nseed', Nseed, 'seed0', seed0, 'T_sim', T_sim, 'T_warm', T_warm, ...
             'dt', dt, 'keep_names', {keep_names});
resume = false;
if exist(partial_file, 'file')
    P = load(partial_file);
    resume = isequaln(P.cfg, cfg);
end
if resume
    REF_all = P.REF_all;  Y_all = P.Y_all;  valid = P.valid;  done = P.done;
    fprintf('Reprise : %d runs déjà faits sur %d.\n', nnz(done), Nseed);
else
    REF_all = nan(Nseed, T, 4, 'single');
    Y_all   = nan(Nseed, T, numel(i_keep), 'single');
    valid   = false(Nseed, 1);
    done    = false(Nseed, 1);
end
clear P

%% Simulations par lots
t_start = tic;
todo = find(~done).';
for b0 = 1:batch_size:numel(todo)
    idx = todo(b0 : min(b0 + batch_size - 1, numel(todo)));

    % Entrées de simulation du lot
    in = repmat(Simulink.SimulationInput(mdl), 1, numel(idx));
    for k = 1:numel(idx)
        s = seeds(idx(k));
        in(k) = in(k).setModelParameter('StopTime', num2str(T_sim));
        in(k) = in(k).setVariable('aprbs_seed', s, 'Workspace', mdl);
        rs = RandStream('mt19937ar', 'Seed', s);
        for v = 1:numel(noise_vars)
            in(k) = in(k).setVariable(noise_vars{v}, randi(rs, 10000));
        end
        if fast_ok
            in(k) = in(k).setModelParameter('SimulationMode', sim_mode);
        end
    end

    % Lancement (mode rapide, puis repli en mode normal si refusé)
    try
        if ~fast_ok, error('mode normal'); end
        if has_pct
            out = parsim(in, 'ShowProgress', 'off', 'UseFastRestart', fr, ...
                         'SetupFcn', setup_fcn);
        else
            out = sim(in, 'ShowProgress', 'off', 'UseFastRestart', fr);
        end
    catch err
        if fast_ok
            warning('Mode rapide refusé (%s). Passage en mode normal.', err.message);
            fast_ok = false;
        end
        for k = 1:numel(in)
            in(k) = in(k).setModelParameter('SimulationMode', 'normal');
        end
        if has_pct
            out = parsim(in, 'ShowProgress', 'off', 'SetupFcn', setup_fcn);
        else
            out = sim(in, 'ShowProgress', 'off');
        end
    end

    % Extraction immédiate (on ne garde que ce qui sert)
    for k = 1:numel(idx)
        n = idx(k);
        if ~isempty(out(k).ErrorMessage)
            % run NON marqué comme fait : il sera retenté au prochain lancement
            warning('Run %d en erreur : %s', n, out(k).ErrorMessage);
            continue
        end
        done(n) = true;
        ref = getsig(out(k), 'ref_log', t, 'previous');
        y   = getsig(out(k), 'simout',  t, 'linear');
        if size(y, 2) ~= numel(y_names)
            error('simout a %d colonnes, y_names en déclare %d.', size(y, 2), numel(y_names));
        end
        if any(~isfinite([ref y]), 'all') || any(abs(y(:, i_ang)) > angle_max, 'all')
            continue   % run divergent : ignoré
        end
        REF_all(n, :, :) = ref;
        Y_all(n, :, :)   = y(:, i_keep);
        valid(n) = true;
    end
    clear out in

    % Sauvegarde intermédiaire et avancement
    save(partial_file, 'REF_all', 'Y_all', 'valid', 'done', 'cfg', '-v7');
    el = toc(t_start);
    nd = nnz(done);
    fprintf('%4d / %d runs (%d valides) | %.0f s écoulées | reste ~%.0f min\n', ...
            nd, Nseed, nnz(valid), el, el / max(1, b0 + numel(idx) - 1) * (numel(todo) - b0 - numel(idx) + 1) / 60);
end

%% Fichier final
REF  = double(REF_all(valid, :, :));    % (N, T, 4)
Y    = double(Y_all(valid, :, :));      % (N, T, numel(keep_names))
SEED = seeds(valid);
y_names_out = keep_names;
save(fullfile(outdir, 'dataset_all.mat'), 'REF', 'Y', 'SEED', 't', 'dt', ...
     'ref_names', 'y_names_out', '-v7');
delete(partial_file);
fprintf('Terminé : %d runs valides sur %d, fichier %s\n', nnz(valid), Nseed, ...
        fullfile(outdir, 'dataset_all.mat'));

% Contrôle : les consignes doivent changer d'un run à l'autre
if size(REF, 1) >= 2 && isequal(REF(1,:,:), REF(2,:,:))
    warning(['Les runs 1 et 2 ont des consignes identiques : la seed APRBS n''est pas ', ...
             'prise en compte. Relancer avec fast_restart = false.']);
end

%% Fonction locale : lit un signal loggé et le rééchantillonne sur la grille t
function X = getsig(o, name, tq, method)
    if ~any(strcmp(o.who, name))
        error('Signal %s absent de la sortie : vérifier le nom du bloc To Workspace.', name);
    end
    s = o.get(name);
    if isa(s, 'timeseries')
        ts = s.Time;  d = s.Data;
    elseif isstruct(s) && isfield(s, 'signals')
        ts = s.time;  d = s.signals.values;
    else
        error('Format non géré pour %s : mettre Save format = Timeseries.', name);
    end
    d = squeeze(d);
    if size(d, 1) ~= numel(ts), d = d.'; end          % une ligne par instant
    [ts, iu] = unique(ts, 'last');                    % instants dupliqués éventuels
    if strcmp(method, 'previous')
        tq = tq + 1e-9;                               % évite de prendre le palier d'avant
    end
    X = interp1(ts, double(d(iu, :)), tq, method, 'extrap');
end