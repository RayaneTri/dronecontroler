function p = commande_space_step(t)
%COMMANDE_SPACE_STEP Positions [x y z] de la trajectoire spatiale aux instants t.
%   t : scalaire ou vecteur de temps -> p : numel(t) x 3
%   La trajectoire est construite une seule fois (persistent).
%   Apres modification des waypoints : clear commande_space_step
    persistent cmd
    if isempty(cmd)
        waypoints = [0 0 0; 0 0 0.5; 0 0 -0.2];   % points de passage (x y z)
        waypoint_times = [0, 5, 10];
        cmd = Commande(waypoints, waypoint_times);
    end
    t = t(:);
    p = zeros(numel(t), 3);
    for k = 1:numel(t)
        p(k, :) = cmd.position(t(k));
    end
end
