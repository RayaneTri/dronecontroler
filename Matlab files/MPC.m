classdef MPC < handle
    properties
        py_mpc % Stocke l'instance de la classe Python
    end
    
    methods
        function obj = MPC(model_path, Q, R, N)
            % Constructeur de la classe MATLAB
            if nargin < 4
                N = 20;
            end
            
            % S'assurer que le dossier courant est dans le path Python
            path_list = py.sys.path;
            if count(path_list, pwd) == 0
                insert(path_list, int32(0), pwd);
            end
            
            % Instancier la classe Python mpc_controller.MPC
            % model_path doit être un char ou string MATLAB
            obj.py_mpc = py.mpc_controller.MPC(model_path, Q, R, int32(N));
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

        function y = internal_state(obj)
            % Méthode pour lire l'état interne (sortie prédite y) du MPC Python
            % Sortie : y (double MATLAB, sortie normalisée du RNN)

            % Appeler la méthode internal_state() du script Python
            py_y = obj.py_mpc.internal_state();

            % self.y vaut 0 (int) avant le 1er step, puis un tenseur torch 1x1 :
            % py.float() gère les deux cas avant la conversion en double MATLAB
            y = double(py.float(py_y));
        end
    end
end