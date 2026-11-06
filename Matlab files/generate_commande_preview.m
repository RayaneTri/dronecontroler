function r_seq = generate_commande_preview(t)
%GENERATE_COMMANDE_PREVIEW Genere une reference de position 3D sur l'horizon MPC.
%
% Modes : 'polynome' (trajectoire minimum snap entre waypoints, voir
% commande_space_step.m), 'sinus' et 'echelon' (appliques sur z, x = y = 0).
%
% Dans un bloc MATLAB Function Simulink :
%   r_seq = generate_commande_preview(t);
%
% Entree : t  temps courant en secondes
% Sortie : r_seq  vecteur colonne (3N x 1) [x1;y1;z1; x2;y2;z2; ...]
%          (format attendu par MPC.step)

    % --- CHOIX DU SIGNAL ---
    signal_type = 'polynome';

    Ts = 0.05;
    N = 20;

    % Parametres du sinus
    amplitude = 1;
    frequency = 1;

    % Parametres de l'echelon
    step_amplitude = 1;
    step_start = 1.0;

    coder.extrinsic('commande_space_step');

    t_future = t + (0:N-1)' * Ts;
    r = zeros(N, 3);

    if strcmp(signal_type, 'sinus')
        r(:, 3) = amplitude * sin(2 * pi * frequency * t_future);
    elseif strcmp(signal_type, 'echelon')
        r(:, 3) = step_amplitude * double(t_future >= step_start);
    elseif strcmp(signal_type, 'polynome')
        r = commande_space_step(t_future);
    else
        error('generate_commande_preview:UnknownSignal', ...
            'Mode inconnu. Choisir polynome, sinus ou echelon.');
    end

    r_seq = reshape(r.', [], 1);
end
