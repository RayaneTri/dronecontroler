classdef MPC < handle
    properties
        py_mpc % Stocke l'instance de la classe Python
    end
    
    methods
        function obj = MPC(model_path, Q, R, ekf, N, V, W)
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
            obj.py_mpc = py.mpc_controller.MPC(model_path, Q, R, ekf, int32(N), V, W);
        end
        
        function u = step(obj, r, az, vz, z)
            % Méthode pour appeler l'optimisation MPC
            % Entrée : r (vecteur de consigne MATLAB de longueur N)
            
            % Convertir le tableau MATLAB en liste Python pour la compatibilité
            r_row = reshape(double(r), 1, []);
            py_r = py.list(num2cell(r_row));
            py_az = py.float(double(az));
            py_vz = py.float(double(vz));
            py_z = py.float(double(z));
            
            % Appeler la méthode step() du script Python
            py_u = obj.py_mpc.step(py_r, py_az, py_vz, py_z);
            
            % Convertir le résultat Python en double MATLAB
            u = double(py_u);
        end

        function y = internal_state(obj)
            % Méthode pour lire l'état interne [az, vz, z] du MPC Python
            % Sortie : y = [az; vz; z] (double MATLAB, vecteur colonne 3x1)
            
            % Appeler la méthode internal_state() du script Python (renvoie un np.ndarray)
            py_y = obj.py_mpc.internal_state();
            
            % Conversion : ndarray -> liste Python -> double MATLAB -> vecteur 3x1
            y = double(py.list(py_y.tolist()));
            y = reshape(y, [3, 1]);
        end
    end
end