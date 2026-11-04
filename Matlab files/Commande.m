classdef Commande < handle
    properties
        py_commande
    end

    methods
        function obj = Commande(temps, positions, Ts, vitesses, acc_initiale, borner)
            if nargin < 4
                vitesses = [];
            end
            if nargin < 5
                acc_initiale = 0.0;
            end
            if nargin < 6
                borner = false;
            end

            module_folder = fileparts(mfilename('fullpath'));
            python_path = py.sys.path;
            if count(python_path, module_folder) == 0
                insert(python_path, int32(0), module_folder);
            end

            py_module = py.importlib.import_module('commande');
            py_temps = py.list(num2cell(reshape(double(temps), 1, [])));
            py_positions = py.list(num2cell(reshape(double(positions), 1, [])));

            if isempty(vitesses)
                py_vitesses = py.None;
            else
                py_vitesses = py.list(num2cell(reshape(double(vitesses), 1, [])));
            end

            obj.py_commande = py_module.COMMANDE( ...
                py_temps, ...
                py_positions, ...
                double(Ts), ...
                py_vitesses, ...
                double(acc_initiale), ...
                py.bool(logical(borner)));
        end

        function y = acceleration(obj, t)
            py_y = obj.py_commande.acceleration(double(t));
            y = double(py_y);
        end
    end
end
