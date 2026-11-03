classdef LQR < LinearModel
    % LQR : Alias de la classe LinearModel pour le contrôle LQR du modèle linéaire
    methods
        function obj = LQR(varargin)
            obj@LinearModel(varargin{:});
        end
    end
end

