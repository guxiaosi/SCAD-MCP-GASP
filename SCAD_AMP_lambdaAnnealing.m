% scad_gamp_lambda_annealing.m
function SCAD_AMP_lambdaAnnealing()
    rng(42);
    %% Parameters
    N          = 10000;
    alpha      = 0.630;
    M          = round(alpha * N);
    rho        = 0.4;
    sigma      = 1.0;
    bar_x      = 0.0;
    Delta      = 0.0;
    maxIter    = 400;
    epsilon    = 1e-5;
    lambda_init = 1.0;
    lambda_min  = 0.05;
    lambda_step = 0.1;
    a           = 3.7;
    %% Generate data
    X = zeros(N,1);
    mask = rand(N,1) < rho;
    X(mask) = bar_x + sigma * randn(sum(mask),1);

    F = randn(M,N) / sqrt(N);
    Y = F * X;
    cF = sum(F(:).^2) / (M*N);
    %% Initial hatx
    hatx = F' * Y;
    hatx = hatx / norm(hatx);
    delta = 0.35 * ones(N,1);
    g = zeros(M,1);
    %% Allocate storage for all iterations across all lambdas
    all_mse = [];
    all_overlap = [];
    lambda_boundaries = [];
    %% Lambda annealing loop
    lambda = lambda_init;
    fprintf('lambda\tIter\tMSE\t\tOverlap\n');

    while lambda >= lambda_min
        for t = 1:maxIter
            V = mean(delta);
            omega = F * hatx - g * V;
            g = (Y - omega) / (Delta + V);
            Gamma = 1 / (Delta + V);
            A = cF * M * Gamma;
            B = F' * g + A * hatx;

            hatx_old = hatx;
            hatx = f_hatx_SCAD(A, B, lambda, a);
            delta = f_delta_SCAD(A, B, lambda, a);

            mse = mean((hatx - X).^2);
            overlap = (hatx' * X) / (norm(hatx) * norm(X));
            
            all_mse = [all_mse; mse];
            all_overlap = [all_overlap; overlap];

            err = norm(hatx - hatx_old)^2 / norm(hatx)^2;
            if err < epsilon
                break;
            end
        end

        mse = mean((hatx - X).^2);
        overlap = (hatx' * X) / (norm(hatx) * norm(X));
        fprintf('%.3f\t%d\t%.4e\t%.4f\n', lambda, t, mse, overlap);
        lambda_boundaries = [lambda_boundaries; length(all_mse)];
        lambda = lambda - lambda_step;
    end
    %% Plot
    figure;
    subplot(2,1,1); hold on;
    plot(all_mse, 'b-', 'LineWidth', 1.2);
    for i = 1:length(lambda_boundaries)
        xline(lambda_boundaries(i), 'k--');
    end
    title(sprintf('SCAD-AMP under $\\lambda$-annealing (N=%d, $\\alpha=%.3f$, $\\rho=%.1f$, $\\sigma^2=%.1f$, $a=%.1f$)', ...
        N, alpha, rho, sigma, a), 'Interpreter', 'latex','FontSize', 18);
    xlabel('Iteration', 'Interpreter', 'latex','FontSize', 18);
    ylabel('MSE', 'Interpreter', 'latex','FontSize', 18);


    subplot(2,1,2); hold on;
    plot(all_overlap, 'r-', 'LineWidth', 1.2);
    for i = 1:length(lambda_boundaries)
        xline(lambda_boundaries(i), 'k--');
    end
    title(sprintf('SCAD-AMP under $\\lambda$-annealing (N=%d, $\\alpha=%.3f$, $\\rho=%.1f$, $\\sigma^2=%.1f$, $a=%.1f$)', ...
        N, alpha, rho, sigma, a), 'Interpreter', 'latex','FontSize', 18);
    xlabel('Iteration', 'Interpreter', 'latex','FontSize', 18);
    ylabel('Overlap', 'Interpreter', 'latex','FontSize', 18);
end

function x = f_hatx_SCAD(A, B, lambda, a)
    absB = abs(B);
    x = zeros(size(B));
    I   = absB > lambda                   & absB <= lambda*(1+A);
    II  = absB > lambda*(1+A)            & absB <= a*lambda*A;
    III = absB > a*lambda*A;

    x(I)    = (B(I) - lambda * sign(B(I))) / A;
    denom   = A * (a - 1) - 1;
    x(II)   = ((a - 1) * B(II) - a * lambda * sign(B(II))) / denom;
    x(III)  = B(III) / A;
end

function d = f_delta_SCAD(A, B, lambda, a)
    absB = abs(B);
    d = zeros(size(B));
    I   = absB > lambda                   & absB <= lambda*(1+A);
    II  = absB > lambda*(1+A)            & absB <= a*lambda*A;
    III = absB > a*lambda*A;

    d(I)    = 1 / A;
    denom   = A * (a - 1) - 1;
    d(II)   = (a - 1) / denom;
    d(III)  = 1 / A;
end
