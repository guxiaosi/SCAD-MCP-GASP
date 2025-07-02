function SCAD_ASP_Gu_0701()
    clc; rng(42);
    overall_tic = tic; 
    %% 1) Parameters
    M       = 800; N       = 1000; %M: sample size, N: dimension.
    rho     = 0.4;  sigma   = 1.0; %rho is sparsity of the true signal.
    bar_x   = 0.0;  m       = 1; %bar_x is the mean, and sigma is the standard deviation of the true signal X_0.
    maxIter = 400;  eps_tol = 1e-5;  %Maximum number of iterations and convergence threshold
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
    epsilon_hist=zeros(maxIter,1);
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
        for i = 1:N
            [phi(i), dphi_dB(i), delta0(i), delta1(i)] = compute_phi_scad(A1, B(i), a, A0, m, lam);
            if any(isnan([phi(i), dphi_dB(i), delta0(i), delta1(i)]))
                fprintf('[ERROR] NaN at iter %d, index %d. Terminating.\n', t, i);
                t = t - 1;
                break_outer = true;
                break;
            end
        end
        if exist('break_outer', 'var') && break_outer
            break;
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
        epsilon_hist(t) = (norm(hatx - hatx_old) / norm(hatx))^2;

        if (norm(hatx - hatx_old) / norm(hatx))^2 < eps_tol, break; end
        fprintf("Overlap=%.3e, MSE=%.3e, A0=%.3e, A1=%.3e, V0=%.3e, V1=%.3e, epsilon=%.3e,\n", ...
                overlap_hist(t),mse_hist(t),A0,A1,V0_old,V1_old,epsilon_hist(t));
    end
    %avg_time_loop = mean(time_loop_hist(1:t));
    %fprintf('Average time to compute input channel terms per iteration: %.6f seconds\n', avg_time_loop);

    total_time = toc(overall_tic);
    fprintf('Total running time: %.6f seconds\n', total_time);
    %—— Plot results —— 
    it=1:t;
    figure;
    plot(it, epsilon_hist(it), 'b-', 'LineWidth', 1.5); hold on;
    legend({'$\| \hat{\mathbf{x}}^{(t)} - \hat{\mathbf{x}}^{(t-1)} \|_2^2/ \| \hat{\mathbf{x}}^{(t)} \|_2^2$'}, ...
           'FontSize', 20, 'Interpreter', 'latex');
    xlabel('Iteration', 'FontSize', 20, 'Interpreter', 'latex');
    ylabel('$\| \hat{\mathbf{x}}^{(t)} - \hat{\mathbf{x}}^{(t-1)} \|_2^2/ \| \hat{\mathbf{x}}^{(t)} \|_2^2$', 'FontSize', 20, 'Interpreter', 'latex');
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
    plot(it, A0_hist(it), 'o-', 'LineWidth', 1.5); hold on;
    plot(it, A1_hist(it), 'x-', 'LineWidth', 1.5);
    legend({'$A_0$', '$A_1$'}, 'Interpreter', 'latex', 'FontSize', 20);
    xlabel('Iteration', 'FontSize', 20, 'Interpreter', 'latex');
    ylabel('$A_0$, $A_1$', 'FontSize', 20, 'Interpreter', 'latex');
    title(['\textbf{SCAD-ASP:} $N=', num2str(N), ...
           '$, $\alpha=', num2str(M/N, '%.3f'), ...
           '$, $\rho=', num2str(rho, '%.2f'), ...
           '$, $\sigma^2=', num2str(sigma^2, '%.2f'), ...
           '$, $m=', num2str(m, '%.2f'), ...
           '$, $a=', num2str(a, '%.2f'), ...
           '$, $\lambda=', num2str(lam, '%.3f'), '$'], ...
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
    if any(isnan([phi, dphi_dB, delta0, delta1]))
        warning('[WARNING] NaN detected in compute_phi_scad!');
    end
end

function phi = compute_phi_only(A1, B, a, A0, m, lam) %eq. 81
    H = @(x) 0.5*erfc(x./sqrt(2));
    % precompute constants
    %D1 = max(a*A1 - 1,1e-12);                    
    %D2 = max(D1 - m*A0*a,1e-12);                  
    D3 = max(A1 - m*A0,1e-12); 
    D4 = max(A1/D3,1e-12);
    D5 = max((a-1)*(A1*(a-1)-1),1e-12);
    p = max((a-1)*(A1*(a-1)-1)-m*(a-1).^2*A0,1e-12);
    sigma1 = sqrt(D4);
    %sigma2 = sqrt(D5/p);
    %% COMPUTE C1-C5
    g1 = lam.*sign(B)*m*sqrt(A0) ./ sqrt(D4);
    g2 = (a-1)*lam*sign(B)*m*sqrt(A0)/sqrt(D5/p);
    C1 = sqrt(A0)./sqrt(2*pi*A1*D3).*(1-exp(-g1.^2/2))./(H(0)-H(g1));
    C2 = -C1+lam*sign(B);
    C3 = (a-1).^2*sqrt(A0)./sqrt(2*pi*D5*p).*(1-exp(-g2.^2/2))./(H(0)-H(g2))+lam*sign(B);
    C4 = -C3+(a+1)*lam*sign(B);
    C5 = (2*sqrt(A0)./sqrt(2*pi*A1*D3)+a*lam)*sign(B);
    %% Define φ_{1}
    phi1_no_correction = @(B) (1/m)*log( ...
        max(( H( (-m*(B-lam*sign(B))*sqrt(A0)/D3) ./ sigma1 ) ...
        - H( ( lam*sign(B)*m*sqrt(A0) - m*(B-lam*sign(B))*sqrt(A0)/D3 ) ./ sigma1 ) ),1e-12) ...
        ./  max((H(0) - H(lam*sign(B)*m*sqrt(A0)/sigma1)),1e-12) ...
      .* exp( m*(B-lam*sign(B)).^2 ./ (2*D3) ) );
    phi1_vals = phi1_no_correction(B) - (B - lam*sign(B)) .* C1;
    %% Define φ_{2}
    qi = @(B) m * ((a-1).*B - a*sign(B)*lam) .* (a-1) * sqrt(A0);
    ki = @(B) -m * ((a-1).*B - a*sign(B)*lam).^2 ...
               - m * lam.^2 * (A1*(a-1) - 1);
    
    phi2_no_correction = @(B) ...
        (1/m) * log( ...
            max(( H(( lam*sign(B)*m*sqrt(A0) - qi(B)/p) ...
                ./ sqrt((a-1)*(A1*(a-1)-1)/p)) ...
            - H((a*lam*sign(B)*m*sqrt(A0) - qi(B)/p) ...
                ./ sqrt((a-1)*(A1*(a-1)-1)/p)) ),1e-12) ...
          ./ max(( H(( lam*sign(B)*m*sqrt(A0) - a*lam*sign(B)*m*sqrt(A0)) ...
                 ./ sqrt((a-1)*(A1*(a-1)-1)/p)) ...
             - H(0) ),1e-12) ...
          .* exp( ( qi(B).^2 - ki(B).*p ) ...
                 ./ (2*(a-1)*(A1*(a-1)-1)*p) ) ...
        );
    B_boundary1 = lam*(A1 +1 - m*A0);
    phi2_vals = phi2_no_correction(B) ...
        - (B - B_boundary1.*sign(B)) .* (C3 - C2) ...
        - (B - lam.*sign(B))       .*  C1;
    %% Define φ_{3}
    phi3_no_correction = @(B) (1/m)*log( ...
        ( H( (a*lam*m*sqrt(A0)-(m*B.*sign(B)*sqrt(A0))/(D3)) ./ sigma1 )/H(0) ) ...
      .* exp( (m*(B.^2-(a+1)*lam.^2*(A1-m*A0)))/ (2*(A1-m*A0)) ) );
    B_boundary2 = a*lam*(A1- m*A0);
    phi3_vals = phi3_no_correction(B) ...
        - (B - B_boundary2.*sign(B)) .* (C5 - C4) ...
        - (B - B_boundary1.*sign(B)) .* (C3 - C2) ...
        - (B - lam.*sign(B))       .*  C1;

    % Piecewise assignment
    if abs(B) <= lam
        phi = 0;
    elseif B > lam && B <= lam*(A1 +1- m*A0)
        phi = phi1_vals;
    elseif B > lam*(A1 +1- m*A0) && B <= a*lam*(A1 - m*A0)
        phi = phi2_vals;
    elseif B > a*lam*(A1 - m*A0)
        phi = phi3_vals;
    elseif B < -lam && B >= -lam*(A1 +1- m*A0)
        phi = phi1_vals;
    elseif B < -lam*(A1 +1- m*A0) && B >= -a*lam*(A1 - m*A0)
        phi = phi2_vals;
    elseif B < -a*lam*(A1 - m*A0)
        phi = phi3_vals;
    else
        phi = 0;
    end
end
