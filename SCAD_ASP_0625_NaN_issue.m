function SCAD_ASP_0625_NaN_issue()
    clc; rng(42);
    overall_tic = tic; 
    %% 1) Parameters
    M       = 6800; N       = 10000; %M: sample size, N: dimension.
    rho     = 0.4;  sigma   = 1.0; %rho is sparsity of the true signal.
    bar_x   = 0.0;  m       = 1; %bar_x is the mean, and sigma is the standard deviation of the true signal X_0.
    maxIter = 400;  eps_tol = 1e-6;  %Maximum number of iterations and convergence threshold
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
        %tic;
        for i = 1:N %eqs. 49-51
            [phi(i), dphi_dB(i), delta0(i), delta1(i)] = compute_phi_scad(A1, B(i), a, A0, m, lam);
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

        if (norm(hatx - hatx_old) / norm(hatx))^2 < eps_tol, break; end
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
function [phi, dphi_dB, delta0, delta1] = compute_phi_scad(A1, B, a, A0, m, lam)
    % Numerical protection
    A0 = max(A0, 1e-12);
    m = max(m, 1e-12);

    phi = compute_phi_only(A1, B, a, A0, m, lam);

    % Derivative w.r.t. A1 using finite difference
    epsA1 = 1e-6;
    phi_p = compute_phi_only(A1 + epsA1, B, a, A0, m, lam);
    phi_m = compute_phi_only(A1 - epsA1, B, a, A0, m, lam);
    dphi_dA1 = (phi_p - phi_m) / (2 * epsA1);

    % First derivative w.r.t. B using finite difference
    dB = 1e-6;
    phi_plus = compute_phi_only(A1, B + dB, a, A0, m, lam);
    phi_minus = compute_phi_only(A1, B - dB, a, A0, m, lam);
    dphi_dB = (phi_plus - phi_minus) / (2 * dB);

    % Second derivative w.r.t. B
    phi_center = compute_phi_only(A1, B, a, A0, m, lam);
    d2phi_dB2 = (phi_plus - 2*phi_center + phi_minus) / (dB^2);

    % Compute delta0 and delta1
    delta0 = -2 * dphi_dA1 - dphi_dB^2;
    delta0 = max(delta0, 0);
    delta1 = d2phi_dB2 - m * delta0;
    delta1 = max(delta1, 0);
end

function phi = compute_phi_only(A1, B, a, A0, m, lam)
% Compute φ^{in}_i(B_i) for SCAD input channel
%% PROTECT A0
eps_val = 1e-12;
if abs(A0) < eps_val
    A0 = eps_val;
end

%% GAUSSIAN TAIL FUNCTION
H = @(x) 0.5 * erfc(x ./ sqrt(2));
%H = @(x) 0.5 * erfc(real(min(max(x, -1e300), 1e300)) ./ sqrt(2));

%% CONSTANTS
%D1 = a*A1 - 1 - m*A0*a;
%D2 = (a*A1 - 1)/D1;
D3 = A1 - m*A0;
D4 = A1/D3;                        % Used in L1, L3
D5 = (a-1)*(A1*(a-1)-1);          % Used in L2
p  = D5 - m*(a-1)^2*A0;

%% SHIFT TERMS
u4 = m*(B - lam)*sqrt(A0)/D3;
u5 = m*(B + lam)*sqrt(A0)/D3;
u8 = m*B*sqrt(A0)/D3;

qip = @(B) m * ((a-1).*B - a*lam) .* (a-1) * sqrt(A0);
qim = @(B) m * ((a-1).*B + a*lam) .* (a-1) * sqrt(A0);
kip = @(B) -m * ((a-1).*B - a*lam).^2 - m * lam.^2 * (A1*(a-1) - 1);
kim = @(B) -m * ((a-1).*B + a*lam).^2 - m * lam.^2 * (A1*(a-1) - 1);

qip_val = qip(B);
qim_val = qim(B);
kip_val = kip(B);
kim_val = kim(B);
u6 = qip_val ./ p;
%u6 = qip_val ./ (p + 1e-12);
u7 = qim_val ./ p;
%u7 = qim_val ./ (p + 1e-12);

%% L_k COMPONENTS
L1p = sqrt(D4) .* ...
    (H(((lam - B) ./ sqrt(A0) - u4) ./ sqrt(D4)) - ...
     H(((lam*(A1+1) - B) ./ sqrt(A0) - u4) ./ sqrt(D4))) ...
    .* exp(m * (B - lam).^2 / (2 * D3));

L1m = sqrt(D4) .* ...
    (H(((-lam*(A1+1) - B) ./ sqrt(A0) - u5) ./ sqrt(D4)) - ...
     H(((-lam - B) ./ sqrt(A0) - u5) ./ sqrt(D4))) ...
    .* exp(m * (B + lam).^2 / (2 * D3));

L2p = sqrt(D5/p) .* ...
    (H(((lam*(A1+1) - B) ./ sqrt(A0) - u6) ./ sqrt(D5/p)) - ...
     H(((a*lam*A1     - B) ./ sqrt(A0) - u6) ./ sqrt(D5/p))) ...
    .* exp((qip_val.^2 - kip_val * p) / (2 * D5 * p));

L2m = sqrt(D5/p) .* ...
    (H(((-a*lam*A1     - B) ./ sqrt(A0) - u7) ./ sqrt(D5/p)) - ...
     H(((-lam*(A1+1)   - B) ./ sqrt(A0) - u7) ./ sqrt(D5/p))) ...
    .* exp((qim_val.^2 - kim_val * p) / (2 * D5 * p));

L3p = sqrt(D4) .* ...
    H(((a*lam*A1 - B) ./ sqrt(A0) - u8) ./ sqrt(D4)) ...
    .* exp(m * (B.^2 - (a+1)*lam.^2*D3) / (2 * D3));

L3m = sqrt(D4) .* ...
    (1 - H(((-a*lam*A1 - B) ./ sqrt(A0) - u8) ./ sqrt(D4))) ...
    .* exp(m * (B.^2 - (a+1)*lam.^2*D3) / (2 * D3));

L4 = H((-lam - B) ./ sqrt(A0)) - H((lam - B) ./ sqrt(A0));

%% FINAL φ^{in}
phi = (1 / m) * log(L1p + L1m + L2p + L2m + L3p + L3m + L4);
end
