classdef MPC < handle
    properties
        py_mpc % Stocke l'instance de la classe Python
    end
    
    methods
        function obj = MPC(model_path, N)
            % Constructeur de la classe MATLAB
            if nargin < 2
                N = 20;
            end
            
            % S'assurer que le dossier courant est dans le path Python
            path_list = py.sys.path;
            if count(path_list, pwd) == 0
                insert(path_list, int32(0), pwd);
            end
            
            % Instancier la classe Python mpc_controller.MPC
            % model_path doit être un char ou string MATLAB
            obj.py_mpc = py.mpc_controller.MPC(model_path, int32(N));
        end
        
        function u = step(obj, r)
            % Méthode pour appeler l'optimisation MPC
            % Entrée : r (vecteur de consigne MATLAB de longueur N)
            
            % Convertir le tableau MATLAB en liste Python pour la compatibilité
            r_row = reshape(double(r), 1, []);
            py_r = py.list(num2cell(r_row));
            
            % Appeler la méthode step() du script Python
            py_u = obj.py_mpc.step(py_r);
            
            % Convertir le résultat Python en double MATLAB
            u = double(py_u);
        end
    end
end