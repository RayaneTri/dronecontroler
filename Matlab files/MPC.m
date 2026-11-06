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
            if count(path_list, fileparts(mfilename('fullpath'))) == 0
                insert(path_list, int32(0), fileparts(mfilename('fullpath')));
            end
            
            % Instancier la classe Python mpc_controller.MPC
            % model_path doit être un char ou string MATLAB
            obj.py_mpc = py.mpc_controller.MPC(model_path, Q, R, ekf, int32(N), V, W);
        end
        
        function u = step(obj, r, a, v, p)
            % Methode pour appeler l'optimisation MPC (position 3D)
            % r : consigne de position sur l'horizon, matrice N x 3 [x y z]
            %     ou vecteur plat [x1 y1 z1 x2 y2 z2 ...] (3N elements)
            % a, v, p : mesures [ax ay az], [vx vy vz], [x y z] (3 elements chacun)
            % u : consignes [z_ref phi_ref theta_ref psi_ref] (4x1)

            if ~isvector(r) && size(r, 2) == 3
                r = r.';          % N x 3 -> ordre ligne x1 y1 z1 x2 ...
            end
            py_r = py.list(num2cell(reshape(double(r), 1, [])));
            py_a = py.list(num2cell(reshape(double(a), 1, [])));
            py_v = py.list(num2cell(reshape(double(v), 1, [])));
            py_p = py.list(num2cell(reshape(double(p), 1, [])));

            py_u = obj.py_mpc.step(py_r, py_a, py_v, py_p);

            u = reshape(double(py.list(py_u.tolist())), [], 1);
        end

        function y = internal_state(obj)
            % Méthode pour lire l'état interne [ax ay az vx vy vz x y z] du MPC Python
            % Sortie : y = 9x1 (double MATLAB, vecteur colonne)
            
            % Appeler la méthode internal_state() du script Python (renvoie un np.ndarray)
            py_y = obj.py_mpc.internal_state();
            
            % Conversion : ndarray -> liste Python -> double MATLAB -> vecteur 9x1
            y = double(py.list(py_y.tolist()));
            y = reshape(y, [9, 1]);
        end
    end
end