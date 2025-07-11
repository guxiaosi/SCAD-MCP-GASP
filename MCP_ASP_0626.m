function MCP_ASP_0626()
    clc; rng(42);
    overall_tic = tic; 
    %% 1) Parameters
    M       = 8500; N       = 10000; %M: sample size, N: dimension.
    rho     = 0.4;  sigma   = 1.0; %rho is sparsity of the true signal.
    bar_x   = 0.0;  m       = 1; %bar_x is the mean, and sigma is the standard deviation of the true signal X_0.
    maxIter = 400;  eps_tol = 1e-7;  %Maximum number of iterations and convergence threshold
    lam     = 1;  a       = 3.7; %a and lambda are MCP penalty parameters.
    psi     = 0; %damping parameter
    %% 2) Generate synthetic data
    X    = zeros(N,1);
    mask = rand(N,1)<rho;
    X(mask)=bar_x+sigma*randn(sum(mask),1);
    F    = randn(M,N)/sqrt(N); %known sensing matrix.
    Y    = F*X; %measurement
    cF   = sum(F(:).^2)/(M*N);
    %% 3) Initialization
    hatx = F' * Y;
    hatx = hatx / norm(hatx); %Initialization for estimator hatx
    delta0 = 0.35*ones(N,1); 
    delta1 = delta0;
    g      = zeros(M,1);
    omega  = zeros(M,1);
    V0_old = 2; 
    V1_old = 1;
    A0 = 1;
    A1 = 1;
    %% 4) Preallocate history (Record the data)
    V0_hist=zeros(maxIter,1);
    V1_hist=zeros(maxIter,1);
    A0_hist=zeros(maxIter,1);
    A1_hist=zeros(maxIter,1);
    mse_hist=zeros(maxIter,1);
    overlap_hist=zeros(maxIter,1);
    %% 5) Iterative updates
    %time_loop_hist = zeros(maxIter,1);
    for t=1:maxIter
        % 1) Damped update for V
        V0_new=mean(delta0);  %eq. 52
        V1_new=mean(delta1); %eq. 53
        V0_old=(1-psi)*V0_new+psi*V0_old;
        V1_old=(1-psi)*V1_new+psi*V1_old;

        % 2) Output channel
        omega_new=F*hatx - g.*(m*V0_old+V1_old); %eq. 42
        omega    =(1-psi)*omega_new+psi*omega;
        g        =(Y-omega)./(V1_old+m*V0_old); %eq. 43

        % 3) Damped update for A
        Gamma0 = V0_old/((V1_old+m*V0_old)*V1_old); %eq. 44
        Gamma1 = 1/V1_old; %eq. 45
        A0_new = cF*M*Gamma0; %eq. 46
        A1_new = cF*M*Gamma1; %eq. 47
        A0      =(1-psi)*A0_new+psi*A0;
        A1      =(1-psi)*A1_new+psi*A1;

        % 4) Linear input
        B = F'*g + (A1-m*A0).*hatx; %eq. 48

        % 5) Compute input channel terms elementwise
        phi = zeros(N,1);
        dphi_dB = zeros(N,1);
        delta0 = zeros(N,1);
        delta1 = zeros(N,1);
        validZ = true;
        %tic;
        for i = 1:N %eqs. 49-51
            [phi(i), dphi_dB(i), delta0(i), delta1(i), tmp_validZ] = compute_phi_mcp(A1, B(i), a, A0, m, lam);
            if ~tmp_validZ
                validZ = false;
                break;
            end
        end
        %time_loop = toc;
        %time_loop_hist(t) = time_loop;
        %fprintf('Time to compute input channel terms at iteration %d: %.6f seconds\n', t, time_loop);
        % 6) Update hatx
        hatx_old=hatx;
        hatx=dphi_dB;

        % 7) Record and print logs
        V0_hist(t)=V0_old; V1_hist(t)=V1_old;
        A0_hist(t)=A0;     A1_hist(t)=A1;
        mse_hist(t)=mean((hatx-X).^2);
        overlap_hist(t) = (hatx' * X) / (norm(hatx) * norm(X));

        if (norm(hatx - hatx_old) / norm(hatx))^2 < eps_tol || ~validZ
            fprintf('Stopped at iteration %d: eps=%.2e\n', t, (norm(hatx - hatx_old) / norm(hatx))^2);
            break;
        end
        fprintf("Overlap=%.3e, MSE=%.3e, A0=%.3e, A1=%.3e, V0=%.3e, V1=%.3e, epsilon=%.3e,\n", ...
                overlap_hist(t),mse_hist(t),A0,A1,V0_old,V1_old,(norm(hatx - hatx_old) / norm(hatx))^2);
    end
    %avg_time_loop = mean(time_loop_hist(1:t));
    %fprintf('Average time to compute input channel terms per iteration: %.6f seconds\n', avg_time_loop);

    total_time = toc(overall_tic);
    fprintf('Total running time: %.6f seconds\n', total_time);
    %—— Plot results —— 
    it=1:t;
    figure;
    plot(it, A0_hist(it), 'o-', 'LineWidth', 1.5); hold on;
    plot(it, A1_hist(it), 'x-', 'LineWidth', 1.5);
    legend({'$A_0$', '$A_1$'}, 'Interpreter', 'latex', 'FontSize', 20);
    xlabel('Iteration', 'FontSize', 20, 'Interpreter', 'latex');
    ylabel('$A_0$, $A_1$', 'FontSize', 20, 'Interpreter', 'latex');
    title(['\textbf{SCAD-ASP:} $N=', num2str(N), ...
           '$, $\alpha=', num2str(M/N, '%.2f'), ...
           '$, $\rho=', num2str(rho, '%.2f'), ...
           '$, $\sigma^2=', num2str(sigma^2, '%.2f'), ...
           '$, $m=', num2str(m, '%.2f'), ...
           '$, $a=', num2str(a, '%.2f'), ...
           '$, $\lambda=', num2str(lam, '%.2f'), '$'], ...
           'FontSize', 20, 'Interpreter', 'latex');
    grid on;
    set(gca, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on');

    figure;
    plot(it, V0_hist(it), 'o-', 'LineWidth', 1.5); hold on;
    plot(it, V1_hist(it), 'x-', 'LineWidth', 1.5);
    legend({'$V_0$', '$V_1$'}, 'Interpreter', 'latex', 'FontSize', 20);
    xlabel('Iteration', 'FontSize', 20, 'Interpreter', 'latex');
    ylabel('$V_0$, $V_1$', 'FontSize', 20, 'Interpreter', 'latex');
    title(['\textbf{SCAD-ASP:} $N=', num2str(N), ...
           '$, $\alpha=', num2str(M/N, '%.2f'), ...
           '$, $\rho=', num2str(rho, '%.2f'), ...
           '$, $\sigma^2=', num2str(sigma^2, '%.2f'), ...
           '$, $m=', num2str(m, '%.2f'), ...
           '$, $a=', num2str(a, '%.2f'), ...
           '$, $\lambda=', num2str(lam, '%.2f'), '$'], ...
           'FontSize', 20, 'Interpreter', 'latex');
    grid on;
    set(gca, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on');
    
    figure;
    plot(it, mse_hist(it), '-m', 'LineWidth', 1.5); hold on;
    plot(it, overlap_hist(it), '-b', 'LineWidth', 1.5);
    legend({'MSE', 'Overlap'}, 'FontSize', 20, 'Interpreter', 'latex');
    xlabel('Iteration', 'FontSize', 20, 'Interpreter', 'latex');
    ylabel('MSE, Overlap', 'FontSize', 20, 'Interpreter', 'latex');
    title(['\textbf{SCAD-ASP:} $N=', num2str(N), ...
           '$, $\alpha=', num2str(M/N, '%.2f'), ...
           '$, $\rho=', num2str(rho, '%.2f'), ...
           '$, $\sigma^2=', num2str(sigma^2, '%.2f'), ...
           '$, $m=', num2str(m, '%.2f'), ...
           '$, $a=', num2str(a, '%.2f'), ...
           '$, $\lambda=', num2str(lam, '%.2f'), '$'], ...
           'FontSize', 20, 'Interpreter', 'latex');
    grid on;
    set(gca, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on');
    
    t_final = t;
    fprintf('Iter\tA0\t\tA1\t\tV0\t\tV1\t\tMSE\t\tOverlap\n');
    for t = 1:t_final
        fprintf('%3d\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\n', ...
            t, A0_hist(t),A1_hist(t), V0_hist(t), V1_hist(t), mse_hist(t), overlap_hist(t));
    end
end
%% ----------------------------- LOCAL FUNCTION -----------------------------
function [phi, dphi_dB, delta0, delta1, validZ] = compute_phi_mcp(A1, B, a, A0, m, lam)
    % Numerical protection
    A0 = max(A0, 1e-12);
    m = max(m, 1e-12);

    [phi, validZ] = compute_phi_only(A1, B, a, A0, m, lam);

    % Derivative w.r.t. A1 using finite difference
    epsA1 = 1e-6;
    [phi_p,~] = compute_phi_only(A1 + epsA1, B, a, A0, m, lam);
    [phi_m,~] = compute_phi_only(A1 - epsA1, B, a, A0, m, lam);
    dphi_dA1 = (phi_p - phi_m) / (2 * epsA1);

    % First derivative w.r.t. B using finite difference
    dB = 1e-6;
    [phi_plus,~] = compute_phi_only(A1, B + dB, a, A0, m, lam);
    [phi_minus,~] = compute_phi_only(A1, B - dB, a, A0, m, lam);
    dphi_dB = (phi_plus - phi_minus) / (2 * dB);

    % Second derivative w.r.t. B
    [phi_center,~] = compute_phi_only(A1, B, a, A0, m, lam);
    d2phi_dB2 = (phi_plus - 2*phi_center + phi_minus) / (dB^2);

    % Compute delta0 and delta1
    delta0 = -2 * dphi_dA1 - dphi_dB^2;
    delta0 = max(delta0, 0);
    delta1 = d2phi_dB2 - m * delta0;
    delta1 = max(delta1, 0);
end

function [phi,validZ] = compute_phi_only(A1, B, a, A0, m, lam)
%% PROTECT A0
eps_val = 1e-12;
if abs(A0) < eps_val
    A0 = eps_val;
end
%% GAUSSIAN TAIL INTEGRAL
H = @(x) 0.5 * erfc(x ./ sqrt(2));
%% PRECOMPUTE CONSTANTS
D1 = max(a*A1 - 1 - m*A0*a,1e-12);
D2 = max((a*A1 - 1)/D1,1e-12);
D3 = max(A1 - m*A0,1e-12);
D4 = max(A1/D3,1e-12);
u1 = (a*m*(B-lam)*sqrt(A0))/D1;
u2 = (a*m*(B+lam)*sqrt(A0))/D1;
u3 = m*B*sqrt(A0)/D3;
%% compute L_k
L1p = sqrt(D2) .* (H(((lam - B) ./ sqrt(A0) - u1) ./ sqrt(D2)) - H(((a * lam * A1 - B) ./ sqrt(A0) - u1) ./ sqrt(D2))) ...
      .* exp(a * m * (B - lam).^2 / (2 * D1));

L1m = sqrt(D2) .* (H(((-a * lam * A1 - B) ./ sqrt(A0) - u2) ./ sqrt(D2)) - H(((-lam - B) ./ sqrt(A0) - u2) ./ sqrt(D2))) ...
      .* exp(a * m * (B + lam).^2 / (2 * D1));

L2p = sqrt(D4) .* H(((a * lam * A1 - B) ./ sqrt(A0) - u3) ./ sqrt(D4)) ...
       .* exp(m * (B.^2 - a * lam.^2 * D3) / (2 * D3));
L2m = sqrt(D4) .* (1 - H(((-a * lam * A1 - B) ./ sqrt(A0) - u3) ./ sqrt(D4))) ...
       .* exp(m * (B.^2 - a * lam.^2 * D3) / (2 * D3));

L3 = H((-lam-B)/sqrt(A0))  - H((lam-B)/sqrt(A0));
Z = L1p + L1m + L2p + L2m + L3;
validZ = (Z > 0);
phi = 1/m * log(Z);
end
