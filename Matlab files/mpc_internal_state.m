function y = mpc_internal_state(varargin)
% MPC_INTERNAL_STATE  Lit l'état interne du contrôleur mpc_system depuis Simulink.
%
%   y = mpc_internal_state()   renvoie mpc_system.internal_state()
%   y = mpc_internal_state(u)  idem ; l'argument u est ignoré, il sert seulement
%                              à forcer Simulink à exécuter ce bloc APRÈS le bloc
%                              du contrôleur (dépendance de données).
%
%   À appeler depuis un bloc MATLAB Function, en extrinsèque :
%
%       function y = mpc_state_block(u)
%       %#codegen
%       coder.extrinsic('mpc_internal_state');
%       y = 0.0;                     % type et taille de la sortie Simulink
%       y = mpc_internal_state(u);
%       end
%
%   L'objet mpc_system doit exister dans le Workspace de base (main_simulation.m)
%   et être une instance de la classe MPC (pas LinearModel).

controller = evalin('base', 'mpc_system');

if ~ismethod(controller, 'internal_state')
    error('mpc_internal_state:noMethod', ...
        'mpc_system (classe %s) n''a pas de méthode internal_state. Utilisez use_linear_model = false.', ...
        class(controller));
end

y = internal_state(controller);
end
