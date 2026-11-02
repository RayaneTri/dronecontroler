% NN - MPC Script

%% Previous script (NN definition)

RNN_syst;                                % Load neural network parameters from previous script
close_system('RNN_simulation_comparaison.slx')    % Close the previous Simulink model

%% Parameters
N = 20; % To be completed                           % Prediction horizon
Nu = 20; % To be completed                           % Control horizon

nb_iterations_optim = 50; % To be completed         % Number of optimization iterations

W_MPC_y = 20;% To be completed                    % State weighting
W_MPC_u = 1;% To be completed                      % Output weighting

%% Define MPC using the neural network as the model

% Initialize MPC
nlmpc_nn = nlmpc(nx,ny,nu);% To be completed      % Initialize the MPC object

nlmpc_nn.Ts = 190;% To be completed                % Set sampling time
nlmpc_nn.PredictionHorizon = N; % To be completed  % Set prediction horizon
nlmpc_nn.ControlHorizon = Nu; % To be completed    % Set control horizon

% Model for MPC

nlmpc_nn.Model.StateFcn = @(x,u) tanh(Wx*x(:) + Wu(:)*(u/coef_input) + bxu(:)); % Normalizing the input        % State update function (neural network dynamics) /!\ If you have size problem, use for instance x(:) to be sure that x is a column vector
nlmpc_nn.Model.OutputFcn = @(x,u) y_span*(Wf*x + bf) + T0; % Denormalize the output                          % Output function (neural network output)

nlmpc_nn.Model.IsContinuousTime = false;% To be completed                                % Discrete-time model

% Initialization
h_eq = zeros(nx,1);
for k = 1:3000, h_eq = tanh(Wx*h_eq + bxu(:)); end
x0 = h_eq;                                                     % Initial state (all zeros)
u0 = 0;                                                                 % Initial control input
validateFcns(nlmpc_nn, x0, u0, [], []);                                 % Validate model functions

% Weightings
nlmpc_nn.W.OutputVariables = W_MPC_y;           % State weighting
nlmpc_nn.W.ManipulatedVariables = W_MPC_u;      % Output weighting

% Constraints
nlmpc_nn.MV.Min = -20;% To be completed              % Minimum control input (minimum value of the input)
nlmpc_nn.MV.Max = 20;% To be completed             % Maximum control input
nlmpc_nn.MV.RateMin = -4;         % Minimum rate of change of control input
nlmpc_nn.MV.RateMax = 4;          % Maximum rate of change of control input

% Optimization algorithm
nlmpc_nn.Optimization.SolverOptions.MaxIterations = nb_iterations_optim;  % Set maximum number of optimization iterations


%% Open Simulink
open('nlMPC_simulation_NN_obs.slx');  % Open the Simulink model for MPC simulation with both NN and state-space models