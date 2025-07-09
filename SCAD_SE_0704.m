%% SCAD-AMP SE from 100 Initializations (Monte Carlo)
clear; clc;

% Parameters
rho     = 0.4;
sigma   = 1.0;
bar_x   = 0.0;
alpha   = 0.690;
lambda  = 0.483;
a_SCAD  = 3.7;
maxIter = 400;
eps_tol = 1e-6;
N_mc    = 10000;  % Monte Carlo samples
num_init = 400;

% Monte Carlo: fixed samples for all runs
rng(42);
z_samples = randn(1, N_mc);
mask = rand(1, N_mc) < rho;
x0_samples = zeros(1, N_mc);
x0_samples(mask) = bar_x + sigma * randn(1, sum(mask));

figure; hold on;

for k = 1:num_init
    chi = rand()*0.4;
    eps = rand()*0.4;
    chi_path = chi;
    eps_path = eps;

    for t = 1:maxIter
        A = alpha / chi;
        B = A * x0_samples + (1/chi) * sqrt(alpha * eps) * z_samples;

        % Evaluate denoiser and its derivative
        x_hat = scad_denoise_vec(A, B, lambda, a_SCAD);
        d_hat = scad_derivative_vec(A, B, lambda, a_SCAD);

        chi_new = mean(d_hat);
        eps_new = mean((x_hat - x0_samples).^2);

        chi_path(end+1) = chi_new;
        eps_path(end+1) = eps_new;

        if abs(chi_new - chi) < eps_tol && abs(eps_new - eps) < eps_tol
            break;
        end
        chi = chi_new;
        eps = eps_new;
    end

    % Draw trajectory
    quiver(chi_path(1:end-1), eps_path(1:end-1), ...
           diff(chi_path), diff(eps_path), ...
           'Color', [0.5 0.5 0.5 0.3], ...   % gray with transparency
           'LineWidth', 1.2, ...
           'MaxHeadSize', 0.5, ...
           'AutoScale', 'on', ...
           'AutoScaleFactor', 1);         % increase arrow length
    plot(chi_path(end), eps_path(end), 'r*', ...
     'MarkerSize', 10, ...
     'LineWidth', 2);  % red star at endpoint
end

xlabel('$\tilde{\chi}$', 'Interpreter', 'latex', 'FontSize', 20);
ylabel('$\epsilon$', 'Interpreter', 'latex', 'FontSize', 20);
title(sprintf('SCAD-SE $\\rho=%.2f,\\ \\alpha=%.2f,\\ \\bar{x}=%.2f,\\ \\sigma^2=%.2f,\\ a=%.2f,\\ \\lambda=%.2f$', ...
      rho, alpha, bar_x, sigma, a_SCAD, lambda), 'Interpreter', 'latex', 'FontSize', 20);
xlim([0, 0.4]); ylim([0, 0.4]);
grid on; set(gca, 'FontSize', 18);
hold off;

%% SCAD Denoiser
function x = scad_denoise_vec(A, B, lambda, a)
    absB = abs(B);
    x = zeros(size(B));

    II = absB > lambda & absB <= lambda * (1 + A);
    III = absB > lambda * (1 + A) & absB <= a * lambda * A;
    IV = absB > a * lambda * A;

    x(II) = (B(II) - lambda * sign(B(II))) / A;

    denom = A * (a - 1) - 1;
    if denom ~= 0
        x(III) = ((a - 1) * B(III) - a * lambda * sign(B(III))) / denom;
    end

    x(IV) = B(IV) / A;
end

function d = scad_derivative_vec(A, B, lambda, a)
    absB = abs(B);
    d = zeros(size(B));

    II = absB > lambda & absB <= lambda * (1 + A);
    III = absB > lambda * (1 + A) & absB <= a * lambda * A;
    IV = absB > a * lambda * A;

    d(II) = 1 / A;

    denom = A * (a - 1) - 1;
    if denom ~= 0
        d(III) = (a - 1) / denom;
    end

    d(IV) = 1 / A;
end
