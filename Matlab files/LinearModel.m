classdef LinearModel < handle
    properties
        py_linear % Stocke l'instance de la classe Python (modèle linéaire / LQR)
    end
    
    methods
        function obj = LinearModel(varargin)
            % Constructeur de la classe MATLAB pour le modèle linéaire
            
            % S'assurer que le dossier courant est dans le path Python
            path_list = py.sys.path;
            if count(path_list, pwd) == 0
                insert(path_list, int32(0), pwd);
            end
            
            % Instancier la classe Python linear_model_matlab.LQR
            obj.py_linear = py.linear_model_matlab.LQR();
        end
        
        function u = step(obj, r)
            % Méthode pour appeler le calcul de commande du modèle linéaire
            % Entrée : r (scalaire ou vecteur de consigne MATLAB)
            
            % Convertir le tableau MATLAB en liste Python pour la compatibilité
            r_row = reshape(double(r), 1, []);
            py_r = py.list(num2cell(r_row));
            
            % Appeler la méthode step() du script Python
            py_u = obj.py_linear.step(py_r);
            
            % Convertir le résultat Python en double MATLAB
            u = double(py_u);
        end
        
        function reset(obj)
            % Réinitialiser l'état du contrôleur
            obj.py_linear = py.linear_model_matlab.LQR();
        end
    end
end

