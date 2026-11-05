classdef Commande < handle
    properties
        py_commande
    end

    methods
        function obj = Commande(points, temps, v_moy, duree, t0)
            % points : matrice N x 3 de points de passage (x, y, z)
            % temps  : instants de passage (optionnel, [] pour allocation automatique)
            % v_moy  : vitesse moyenne si temps non donne (m/s)
            % duree  : duree totale si temps non donne (prioritaire sur v_moy, [] pour ignorer)
            % t0     : instant de depart si temps non donne
            if nargin < 2
                temps = [];
            end
            if nargin < 3 || isempty(v_moy)
                v_moy = 1.0;
            end
            if nargin < 4
                duree = [];
            end
            if nargin < 5 || isempty(t0)
                t0 = 0.0;
            end

            module_folder = fileparts(mfilename('fullpath'));
            python_path = py.sys.path;
            if count(python_path, module_folder) == 0
                insert(python_path, int32(0), module_folder);
            end

            py_module = py.importlib.import_module('command_space');

            points = double(points);
            if size(points, 2) ~= 3
                error('Commande:BadPoints', 'points doit etre de dimension N x 3');
            end
            n = size(points, 1);
            % points.' (3 x N) en ordre colonne = ordre ligne de la matrice N x 3
            py_points = py.numpy.reshape( ...
                py.numpy.array(num2cell(reshape(points.', 1, []))), ...
                py.tuple({int32(n), int32(3)}));

            if isempty(temps)
                py_temps = py.None;
            else
                py_temps = py.numpy.array(num2cell(reshape(double(temps), 1, [])));
            end

            if isempty(duree)
                py_duree = py.None;
            else
                py_duree = double(duree);
            end

            obj.py_commande = py_module.TRAJECTOIRE_ESPACE( ...
                py_points, ...
                pyargs('temps', py_temps, ...
                       'v_moy', double(v_moy), ...
                       'duree', py_duree, ...
                       't0', double(t0)));
        end

        function p = position(obj, t)
            % Position [x y z] a l'instant t (scalaire) -> vecteur ligne 1 x 3
            py_p = obj.py_commande.position(double(t));
            p = reshape(double(py_p.tolist()), 1, []);
        end
    end
end
