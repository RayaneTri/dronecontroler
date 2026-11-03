function y = EKF_MeasurementFcn_coder_ext( x )
    
    % Wf: Hidden-to-output layer weights (head_weight)
    Wf  = evalin('base','Wf');
    
    % bf: Output layer bias (head_bias)
    bf  = evalin('base','bf');

    y_span = evalin('base','y_span');
    T0     = evalin('base','T0');
    % Output equation
    y = y_span*(Wf*x(:) + bf) + T0; % To be completed

end

