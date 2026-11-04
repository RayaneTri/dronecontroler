function y = mpc_internal_state(varargin)
% MPC_INTERNAL_STATE  Lit l'état interne du contrôleur mpc_system depuis Simulink.
%
%   y = mpc_internal_state()   renvoie mpc_system.internal_state()
%   y = mpc_internal_state(u)  idem ; l'argument u est utilisé pour forcer l'ordre
%                              d'exécution Simulink (dépendance de données).
%
%   Sortie : y = [az; vz; z] (vecteur colonne 3x1 double, unités physiques)

% 1. Récupération de l'objet depuis le Workspace de base
controller = evalin('base', 'mpc_system');

% 2. Vérification de la présence de la méthode
if ~ismethod(controller, 'internal_state') && ~isprop(controller, 'internal_state')
    error('mpc_internal_state:noMethod', ...
        'mpc_system (classe %s) n''a pas de méthode internal_state. Utilisez use_linear_model = false.', ...
        class(controller));
end

% 3. Appel de la méthode (renvoie un py.numpy.ndarray ou un type MATLAB)
raw_y = internal_state(controller);

% 4. Conversion explicite Python -> MATLAB double
if isa(raw_y, 'py.numpy.ndarray') || startsWith(class(raw_y), 'py.')
    % Convertit le ndarray Python en liste Python, puis en double MATLAB
    y = double(py.list(raw_y.tolist()));
else
    y = double(raw_y);
end

% 5. Force la forme en vecteur colonne 3x1
y = reshape(y, [3, 1]);
end