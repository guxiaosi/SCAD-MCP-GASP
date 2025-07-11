function mcp_amp_asp_0610()
    clc; rng(42);
    overall_tic = tic; 
    %% 1) Parameters
    M       = 6580; N       = 10000;
    rho     = 0.4;  sigma   = 1.0;
    bar_x   = 0.0;  m       = 1;
    maxIter = 400;  eps_tol = 1e-5;
    lam     = 0.5;  a       = 3.7;
    init_eps= 1e-2; psi     = 0;

    %% 2) Generate synthetic data
    X    = zeros(N,1);
    mask = rand(N,1)<rho;
    X(mask)=bar_x+sigma*randn(sum(mask),1);
    F    = randn(M,N)/sqrt(N);
    Y    = F*X;
    cF   = sum(F(:).^2)/(M*N);

    %% 3) Initialization
    var0   = rho*(bar_x^2+sigma^2)-(rho*bar_x)^2;
    v0     = F'*Y; v0=v0/norm(v0);
    hatx   = v0*sqrt(N*(var0+init_eps));
    delta0 = (var0+init_eps)*ones(N,1);
    delta1 = delta0;
    g      = zeros(M,1);
    omega  = zeros(M,1);
    V0_old = 2; 
    V1_old = 1;
    A0 = 1;
    A1 = 1;

    V0_hist=zeros(maxIter,1);
    V1_hist=zeros(maxIter,1);
    A0_hist=zeros(maxIter,1);
    A1_hist=zeros(maxIter,1);
    mse_hist=zeros(maxIter,1);
    overlap_hist=zeros(maxIter,1);

    %% 4) Iterative updates
    time_loop_hist = zeros(maxIter,1);
    for t=1:maxIter
        % 1) Damped update for V
        V0_new=mean(delta0); 
        V1_new=mean(delta1);
        V0_old=(1-psi)*V0_new+psi*V0_old;
        V1_old=(1-psi)*V1_new+psi*V1_old;

        % 2) Output channel
        omega_new=F*hatx - g.*(m*V0_old+V1_old);
        omega    =(1-psi)*omega_new+psi*omega;
        g        =(Y-omega)./(V1_old+m*V0_old);

        % 3) Damped update for A
        Gamma0 = V0_old/((V1_old+m*V0_old)*V1_old);
        Gamma1 = 1/V1_old;
        A0_new = cF*M*Gamma0;
        A1_new = cF*M*Gamma1;
        A0      =(1-psi)*A0_new+psi*A0;
        A1      =(1-psi)*A1_new+psi*A1;

        % 4) Linear input
        B = F'*g + (A1-m*A0).*hatx;

        % 5) Compute input channel terms elementwise
        phi = zeros(N,1);
        dphi_dB = zeros(N,1);
        delta0 = zeros(N,1);
        delta1 = zeros(N,1);
        tic;
        for i = 1:N
            [phi(i), dphi_dB(i), delta0(i), delta1(i)] = compute_phi_MCP(A1, B(i), a, A0, m, lam);
        end
        time_loop = toc;
        time_loop_hist(t) = time_loop;
        fprintf('Time to compute input channel terms at iteration %d: %.6f seconds\n', t, time_loop);
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
    avg_time_loop = mean(time_loop_hist(1:t));
    fprintf('Average time to compute input channel terms per iteration: %.6f seconds\n', avg_time_loop);

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
    title(['\textbf{MCP-ASP:} $N=', num2str(N), ...
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
    title(['\textbf{MCP-ASP:} $N=', num2str(N), ...
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
    plot(it, mse_hist(it), '-m', 'LineWidth', 1.5); hold on;
    plot(it, overlap_hist(it), '-b', 'LineWidth', 1.5);
    legend({'MSE', 'Overlap'}, 'FontSize', 20, 'Interpreter', 'latex');
    xlabel('Iteration', 'FontSize', 20, 'Interpreter', 'latex');
    ylabel('MSE, Overlap', 'FontSize', 20, 'Interpreter', 'latex');
    title(['\textbf{MCP-ASP:} $N=', num2str(N), ...
           '$, $\alpha=', num2str(M/N, '%.3f'), ...
           '$, $\rho=', num2str(rho, '%.2f'), ...
           '$, $\sigma^2=', num2str(sigma^2, '%.2f'), ...
           '$, $m=', num2str(m, '%.2f'), ...
           '$, $a=', num2str(a, '%.2f'), ...
           '$, $\lambda=', num2str(lam, '%.3f'), '$'], ...
           'FontSize', 20, 'Interpreter', 'latex');
    grid on;
    set(gca, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'on');

end


%% ----------------------------- LOCAL FUNCTION -----------------------------
function [phi, dphi_dB, delta0, delta1] = compute_phi_MCP(A1, B, a, A0, m, lam)
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
    H = @(x) 0.5*erfc(x./sqrt(2));
    % precompute constants
    D1 = max(a*A1 - 1 - m*A0*a,1e-12);                    
    D2 = max((a*A1 - 1)/D1,1e-12);                  
    D3 = max(A1 - m*A0,1e-12); 
    D4 = max(A1/D3,1e-12);
    %% COMPUTE C6(B) AND C7
    t0 = a*lam.*sign(B)*m*sqrt(A0) ./ sqrt(D2);
    C6 = a*sqrt(A0) ./ sqrt(2*pi*(a*A1-1)*D1) ...
         .* (1 - exp(-t0.^2/2)) ./ (H(0) - H(t0));
    C7 = a*lam*sign(B)-C6;
    C8 = (2*sqrt(A0)/sqrt(2*pi*A1*D3)+a*lam)*sign(B);
    %% Define φ_{1}
    phi1_no_correction = @(B) (1/m)*log( ...
        max(( H( (-a*m*(B-lam*sign(B))*sqrt(A0)/D1) ./ sqrt(D2) ) ...
        - H( ( a*lam*sign(B)*m*sqrt(A0) - a*m*(B-lam*sign(B))*sqrt(A0)/D1 ) ./ sqrt(D2)) ),1e-12) ...
        ./ max((H(0) - H(a*lam*sign(B)*m*sqrt(A0)/sqrt(D2))),1e-12) ...
      .* exp( a*m*(B-lam*sign(B)).^2 ./ (2*D1) ) );
    phi1_vals = phi1_no_correction(B) - (B - lam*sign(B)) .*C6;
    %% Define φ_{2}
    phi2_no_correction = @(B) (1/m)*log( ...
        ( H( ( a*lam*m*sqrt(A0) - m*B.*sign(B)*sqrt(A0)/D3 ) ./ sqrt(D4) ) ...
          ./ (H(0)) ) ...
      .* exp( m*(B.^2 - a*lam^2*D3) ./ (2*D3) ) );
    B_boundary = a*lam*(A1-m*A0);
    phi2_vals = phi2_no_correction(B) - (B - B_boundary*sign(B)) .* (C8 - C7) - (B-lam*sign(B)) .* C6;

    % Piecewise assignment
    if abs(B) <= lam
        phi = 0;
    elseif B > lam && B <= a*lam*(A1 - m*A0)
        phi = phi1_vals;
    elseif B > a*lam*(A1 - m*A0)
        phi = phi2_vals;
    elseif B < -lam && B >= -a*lam*(A1 - m*A0)
        phi = phi1_vals;
    elseif B < -a*lam*(A1 - m*A0)
        phi = phi2_vals;
    else
        phi = 0;
    end
end