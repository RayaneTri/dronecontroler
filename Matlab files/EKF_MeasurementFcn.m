function y = EKF_MeasurementFcn( x )
    
    % Coder information
    coder.extrinsic('EKF_MeasurementFcn_coder_ext')

    % Output init
    y = 0;

    % Function
    y = EKF_MeasurementFcn_coder_ext( x );

end

