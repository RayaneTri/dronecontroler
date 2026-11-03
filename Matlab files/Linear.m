classdef Linear < LinearModel
    % Linear : Alias de LinearModel
    methods
        function obj = Linear(varargin)
            obj@LinearModel(varargin{:});
        end
    end
end

