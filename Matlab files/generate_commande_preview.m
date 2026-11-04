function r_seq = generate_commande_preview(t)
%GENERATE_COMMANDE_PREVIEW Genere une reference sur l'horizon MPC.
%
% Modes disponibles : 'polynome', 'sinus' et 'echelon'.
%
% Pour choisir le signal, modifier la variable signal_type ci-dessous :
%   signal_type = 'polynome' : trajectoire quintique entre les waypoints.
%   signal_type = 'sinus'    : sinus defini par amplitude et frequency.
%   signal_type = 'echelon'  : echelon defini par step_amplitude et step_start.
%
% Dans un bloc MATLAB Function Simulink, appeler simplement :
%   r_seq = generate_commande_preview(t);
%
% Entree :
%   t      temps courant en secondes
%
% Sortie :
%   r_seq  vecteur colonne des N prochaines positions de reference

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

    t_future = t + (0:N-1)' * Ts;

    if strcmp(signal_type, 'sinus')
        r_seq = amplitude * sin(2 * pi * frequency * t_future);
        return;
    end

    if strcmp(signal_type, 'echelon')
        r_seq = step_amplitude * double(t_future >= step_start);
        return;
    end

    if ~strcmp(signal_type, 'polynome')
        error('generate_commande_preview:UnknownSignal', ...
            'Mode inconnu. Choisir polynome, sinus ou echelon.');
    end

    waypoint_times = [0, 5, 10];
    waypoint_positions = [0, 0.5, -0.2];
    waypoint_velocities = [0, 0, 0];
    initial_acceleration = -9.81;
    borner = true;

    if borner
        [waypoint_times, coefficients] = bound_trajectory( ...
            waypoint_times, waypoint_positions, waypoint_velocities, ...
            Ts, initial_acceleration);
    else
        coefficients = build_coefficients( ...
            waypoint_times, waypoint_positions, waypoint_velocities, ...
            initial_acceleration);
    end

    r_seq = zeros(N, 1);
    for k = 1:N
        future_time = t + (k - 1) * Ts;
        r_seq(k) = evaluate_trajectory( ...
            future_time, waypoint_times, waypoint_positions, coefficients);
    end
end

function coefficients = build_coefficients( ...
        waypoint_times, waypoint_positions, waypoint_velocities, initial_acceleration)
    segment_count = numel(waypoint_times) - 1;
    coefficients = zeros(segment_count, 6);
    acceleration = initial_acceleration;

    for i = 1:segment_count
        duration = waypoint_times(i + 1) - waypoint_times(i);
        coefficients(i, :) = quintic_coefficients( ...
            duration, ...
            waypoint_positions(i), ...
            waypoint_positions(i + 1), ...
            waypoint_velocities(i), ...
            waypoint_velocities(i + 1), ...
            acceleration, ...
            0);
        acceleration = 0;
    end
end

function [waypoint_times, coefficients] = bound_trajectory( ...
        waypoint_times, waypoint_positions, waypoint_velocities, Ts, initial_acceleration)
    tolerance = 1e-3;
    max_iterations = 50;
    segment_count = numel(waypoint_times) - 1;
    coefficients = build_coefficients( ...
        waypoint_times, waypoint_positions, waypoint_velocities, initial_acceleration);

    for iteration = 1:max_iterations
        first_sample = round(waypoint_times(1) / Ts);
        last_sample = round(waypoint_times(end) / Ts);
        sample_count = last_sample - first_sample + 1;
        trajectory = zeros(sample_count, 1);

        for k = 1:sample_count
            sample_time = Ts * (first_sample + k - 1);
            trajectory(k) = evaluate_trajectory( ...
                sample_time, waypoint_times, waypoint_positions, coefficients);
        end

        lower_excess = min(waypoint_positions) - min(trajectory);
        upper_excess = max(trajectory) - max(waypoint_positions);
        if max(lower_excess, upper_excess) <= tolerance
            return;
        end

        excess = max( ...
            min(waypoint_positions) - trajectory, ...
            trajectory - max(waypoint_positions));
        [~, sample_index] = max(excess);
        sample_time = Ts * (first_sample + sample_index - 1);

        active_segment = segment_count;
        for i = 1:segment_count
            if sample_time < waypoint_times(i + 1)
                active_segment = i;
                break;
            end
        end

        if active_segment == segment_count
            return;
        end

        shortened_time = waypoint_times(active_segment) + ...
            0.9 * (waypoint_times(active_segment + 1) - ...
            waypoint_times(active_segment));
        if shortened_time - waypoint_times(active_segment) < Ts
            return;
        end

        waypoint_times(active_segment + 1) = shortened_time;
        coefficients = build_coefficients( ...
            waypoint_times, waypoint_positions, waypoint_velocities, initial_acceleration);
    end
end

function coefficient = quintic_coefficients(T, za, zb, va, vb, aa, ab)
    remainder_position = zb - za - va * T - aa * T^2 / 2;
    remainder_velocity = vb - va - aa * T;
    remainder_acceleration = ab - aa;

    matrix = [ ...
        T^3,     T^4,      T^5; ...
        3*T^2,   4*T^3,    5*T^4; ...
        6*T,    12*T^2,   20*T^3];
    high_order = matrix \ [ ...
        remainder_position; ...
        remainder_velocity; ...
        remainder_acceleration];

    coefficient = [za, va, aa / 2, high_order'];
end

function value = evaluate_trajectory(t, waypoint_times, waypoint_positions, coefficients)
    if t <= waypoint_times(1)
        value = waypoint_positions(1);
        return;
    end

    if t >= waypoint_times(end)
        value = waypoint_positions(end);
        return;
    end

    segment = 1;
    for i = 1:size(coefficients, 1)
        if t < waypoint_times(i + 1)
            segment = i;
            break;
        end
    end

    local_time = t - waypoint_times(segment);
    coefficient = coefficients(segment, :);
    value = 0;
    for i = 1:numel(coefficient)
        value = value + coefficient(i) * local_time^(i - 1);
    end
end
