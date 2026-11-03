function y = EKF_StateTransitionFcn_coder_ext( x , u )

    % Wu: Input-to-hidden layer weights (weight_ih_l0)
    Wu  = evalin('base','Wu');

    % Wx: Hidden-to-hidden layer weights (weight_hh_l0)
    Wx  = evalin('base','Wx');

    % bxu: Hidden layer bias vector
    bxu = evalin('base','bxu');

    coef_input = evalin('base','coef_input');
    % State equation
    y = tanh(Wx*x(:) + Wu(:)*(u/coef_input) + bxu(:));% To be completed
    
end

