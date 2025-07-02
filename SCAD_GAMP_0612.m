% scad_gamp_with_counts_and_A.m
% GAMP for SCAD in MATLAB
% –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
rng(42);
startTime = tic;

%% 1) Parameters
M          = 900;        % number of measurements
N          = 1000;        % signal dimension
rho        = 0.4;          % sparsity of X
sigma      = 1.0;          % std of nonzero entries
bar_x      = 0.0;          % mean of nonzero entries
Delta      = 0.0;          % noise variance
maxIter = 400;          % max AMP iterations
epsilon    = 1e-5;         % convergence threshold
%cF         = 1.0 / N;      % approx. sum F_ij^2 / (M*N)
lambda     = 0.483;            % SCAD parameter λ
a     = 3.7;          % SCAD parameter a

%% 2) Generate true sparse signal X
X = zeros(N,1);
mask = rand(N,1) < rho;
X(mask) = bar_x + sigma*randn(sum(mask),1);

%% 3) Measurement and observations
F = randn(M,N) / sqrt(N);
Y = F * X;
cF   = sum(F(:).^2)/(M*N);
%% 4) AMP Initialization (spectral)
hatx = F' * Y;
hatx = hatx / norm(hatx); %Initialization for estimator hatx
delta = 0.35 * ones(N,1); %Initialize delta (V is initialized similarly since V = mean(delta))
g     = zeros(M,1); %Initialization for g
%% 5) Preallocate history (Record the data)
hatx_hist = zeros(maxIter, N);
V_hist    = zeros(maxIter,1);
A_hist    = zeros(maxIter,1);
mse_hist=zeros(maxIter,1);
overlap_hist=zeros(maxIter,1);
epsilon_hist=zeros(maxIter,1);
%% 6) Main loop
time_loop_hist = zeros(maxIter,1);
for t = 1:maxIter
    % (1) Output linear
    V     = mean(delta); %eq. 20
    omega = F * hatx - g * V; %eq. 13
    % (2) Output nonlin
    g     = (Y - omega) / (Delta + V); %eq. 14
    Gamma = 1 / (Delta + V); %eq. 15
    % (3) Input linear
    A = cF * M * Gamma; %eq. 16
    B = F' * g + A * hatx; %eq. 17

    % (4) Input nonlin
    hatx_old = hatx;
    %tic;
    hatx     = f_hatx_SCAD(A, B, lambda, a); %eq. 18
    delta    = f_delta_SCAD(A, B, lambda, a); %eq. 19
    %time_loop = toc;
    %time_loop_hist(t) = time_loop;
    %fprintf('Time to compute input channel terms at iteration %d: %.6f seconds\n', t, time_loop);    
    % record histories
    hatx_hist(t,:) = hatx.';
    V_hist(t)      = V;
    A_hist(t)      = A;
    mse_hist(t)=mean((hatx-X).^2);
    overlap_hist(t) = (hatx' * X) / (norm(hatx) * norm(X));
    epsilon_hist(t) = (norm(hatx - hatx_old) / norm(hatx))^2;
    % convergence stopping criterion
    if (norm(hatx - hatx_old) / norm(hatx))^2 < epsilon
        fprintf('Converged at iteration %d\n', t);
        break;
    end
    V_old = V;
end

%avg_time_loop = mean(time_loop_hist(1:t));
%fprintf('Average time to compute input channel terms per iteration: %.6f seconds\n', avg_time_loop);


t_final   = t;
V_hist    = V_hist(1:t_final);
A_hist    = A_hist(1:t_final);

%—— Plot results —— 
    it=1:t;
    figure;
    plot(it, epsilon_hist(it), 'o-', 'LineWidth', 1.5);
    legend({'$\| \hat{\mathbf{x}}^{(t)} - \hat{\mathbf{x}}^{(t-1)} \|_2^2/ \| \hat{\mathbf{x}}^{(t)} \|_2^2$'}, 'Interpreter', 'latex', 'FontSize', 20);
    xlabel('Iteration', 'FontSize', 20, 'Interpreter', 'latex');
    ylabel('$\| \hat{\mathbf{x}}^{(t)} - \hat{\mathbf{x}}^{(t-1)} \|_2^2/ \| \hat{\mathbf{x}}^{(t)} \|_2^2$', 'FontSize', 20, 'Interpreter', 'latex');
    title(['\textbf{SCAD-AMP:} $N=', num2str(N), ...
           '$, $\alpha=', num2str(M/N, '%.2f'), ...
           '$, $\rho=', num2str(rho, '%.2f'), ...
           '$, $\sigma^2=', num2str(sigma^2, '%.2f'), ...
           '$, $a=', num2str(a, '%.2f'), ...
           '$, $\lambda=', num2str(lambda, '%.3f'), '$'], ...
           'FontSize', 20, 'Interpreter', 'latex');
    grid on;
    set(gca, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on');

    figure;
    plot(it, A_hist(it), 'o-', 'LineWidth', 1.5);
    legend({'$A$'}, 'Interpreter', 'latex', 'FontSize', 20);
    xlabel('Iteration', 'FontSize', 20, 'Interpreter', 'latex');
    ylabel('$A$', 'FontSize', 20, 'Interpreter', 'latex');
    title(['\textbf{SCAD-AMP:} $N=', num2str(N), ...
           '$, $\alpha=', num2str(M/N, '%.2f'), ...
           '$, $\rho=', num2str(rho, '%.2f'), ...
           '$, $\sigma^2=', num2str(sigma^2, '%.2f'), ...
           '$, $a=', num2str(a, '%.2f'), ...
           '$, $\lambda=', num2str(lambda, '%.3f'), '$'], ...
           'FontSize', 20, 'Interpreter', 'latex');
    grid on;
    set(gca, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on');

    figure;
    plot(it, V_hist(it), 'o-', 'LineWidth', 1.5);
    legend({'$V$'}, 'Interpreter', 'latex', 'FontSize', 20);
    xlabel('Iteration', 'FontSize', 20, 'Interpreter', 'latex');
    ylabel('$V$', 'FontSize', 20, 'Interpreter', 'latex');
    title(['\textbf{SCAD-AMP:} $N=', num2str(N), ...
           '$, $\alpha=', num2str(M/N, '%.2f'), ...
           '$, $\rho=', num2str(rho, '%.2f'), ...
           '$, $\sigma^2=', num2str(sigma^2, '%.2f'), ...
           '$, $a=', num2str(a, '%.2f'), ...
           '$, $\lambda=', num2str(lambda, '%.3f'), '$'], ...
           'FontSize', 20, 'Interpreter', 'latex');
    grid on;
    set(gca, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on');
    
    figure;
    plot(it, mse_hist(it), '-m', 'LineWidth', 1.5); hold on;
    plot(it, overlap_hist(it), '-b', 'LineWidth', 1.5);
    legend({'MSE', 'Overlap'}, 'FontSize', 20, 'Interpreter', 'latex');
    xlabel('Iteration', 'FontSize', 20, 'Interpreter', 'latex');
    ylabel('MSE, Overlap', 'FontSize', 20, 'Interpreter', 'latex');
    title(['\textbf{SCAD-AMP:} $N=', num2str(N), ...
           '$, $\alpha=', num2str(M/N, '%.2f'), ...
           '$, $\rho=', num2str(rho, '%.2f'), ...
           '$, $\sigma^2=', num2str(sigma^2, '%.2f'), ...
           '$, $a=', num2str(a, '%.2f'), ...
           '$, $\lambda=', num2str(lambda, '%.3f'), '$'], ...
           'FontSize', 20, 'Interpreter', 'latex');
    grid on;
    set(gca, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on');
    ylim([0 1]);

%% SCAD denoiser subfunctions
function x = f_hatx_SCAD(A, B, lambda, a)
    absB = abs(B);
    x = zeros(size(B));
    I   = absB > lambda                   & absB <= lambda*(1+A);
    II  = absB > lambda*(1+A)             & absB <= a*lambda*A;
    III = absB > a*lambda*A;
    x( I) = (B(I) - lambda*sign(B(I))) / A;
    denom = A*(a-1) - 1;
    x(II) = ((a-1)*B(II) - a*lambda*sign(B(II))) / denom;
    x(III)= B(III) / A;
end

function d = f_delta_SCAD(A, B, lambda, a)
    absB = abs(B);
    d = zeros(size(B));
    I   = absB > lambda                   & absB <= lambda*(1+A);
    II  = absB > lambda*(1+A)             & absB <= a*lambda*A;
    III = absB > a*lambda*A;
    d( I) = 1 / A;
    denom = A*(a-1) - 1;
    d(II) = (a-1) / denom;
    d(III)= 1 / A;
end
elapsedTime = toc(startTime);
fprintf('running time：%.4f s\n', elapsedTime);

fprintf('Iter\tA\t\tV\t\tMSE\t\tOverlap\n');
for t = 1:t_final
    fprintf('%3d\t%.4e\t%.4e\t%.4e\t%.4e\n', ...
        t, A_hist(t), V_hist(t), mse_hist(t), overlap_hist(t));
end
