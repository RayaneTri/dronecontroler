function y = EKF_StateTransitionFcn( x , u )

    % Coder information
    coder.extrinsic('EKF_StateTransitionFcn_coder_ext')

    % Output init
    y = zeros(length(x),1);

    % Function
    y = EKF_StateTransitionFcn_coder_ext( x , u );

end

