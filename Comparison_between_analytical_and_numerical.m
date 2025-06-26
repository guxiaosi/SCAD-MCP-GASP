clear; clc; close all;

%% PARAMETERS
a   = 3.7;
A0  = 0.3;
A1  = 1;
m   = 0.3;
lam = 1;

%% PROTECTION
eps = 1e-12;
if abs(A0) < eps
    A0 = eps;
end

%% GAUSSIAN TAIL FUNCTION
H = @(x) 0.5 * erfc(x ./ sqrt(2));

%% PRECOMPUTE CONSTANTS
D1 = a*A1 - 1 - m*A0*a;
D3 = A1 - m*A0;
D4 = A1/D3;
D5 = (a-1)*(A1*(a-1)-1);
p  = (a-1)*(A1*(a-1)-1)-m*(a-1)^2*A0;

%% B grid
B = linspace(-10,10,2001);

% Terms for analytic φ^in
u4 = m*(B-lam)*sqrt(A0)/D3;
u5 = m*(B+lam)*sqrt(A0)/D3;
u8 = m*B*sqrt(A0)/D3;
qip_val = m*((a-1)*B - a*lam)*(a-1)*sqrt(A0);
qim_val = m*((a-1)*B + a*lam)*(a-1)*sqrt(A0);
kip_val = -m*((a-1)*B - a*lam).^2 - m*lam^2*(A1*(a-1) - 1);
kim_val = -m*((a-1)*B + a*lam).^2 - m*lam^2*(A1*(a-1) - 1);
u6 = qip_val ./ p;
u7 = qim_val ./ p;

% Compute L_k
L1p = sqrt(D4) .* (H(((lam - B)./sqrt(A0) - u4) ./ sqrt(D4)) - H(((lam*(A1+1) - B)./sqrt(A0) - u4) ./ sqrt(D4))) ...
      .* exp( m * (B - lam).^2 / (2 * D3));
L1m = sqrt(D4) .* (H(((-lam*(A1+1) - B)./sqrt(A0) - u5) ./ sqrt(D4)) - H(((-lam - B)./sqrt(A0) - u5) ./ sqrt(D4))) ...
      .* exp( m * (B + lam).^2 / (2 * D3));
L2p = sqrt(D5/p) .* (H(((lam*(A1+1) - B)./sqrt(A0) - u6) ./ sqrt(D5/p)) - H(((a*lam*A1 - B)./sqrt(A0) - u6) ./ sqrt(D5/p))) ...
      .* exp((qip_val.^2 - kip_val * p) / (2*D5*p));
L2m = sqrt(D5/p) .* (H(((-a*lam*A1 - B)./sqrt(A0) - u7) ./ sqrt(D5/p)) - H(((-lam*(A1+1) - B)./sqrt(A0) - u7) ./ sqrt(D5/p))) ...
      .* exp((qim_val.^2 - kim_val * p) / (2*D5*p));
L3p = sqrt(D4) .* H(((a*lam*A1 - B)./sqrt(A0) - u8) ./ sqrt(D4)) ...
       .* exp(m * (B.^2 - (a+1)*lam^2*D3) / (2*D3));
L3m = sqrt(D4) .* (1 - H(((-a*lam*A1 - B)./sqrt(A0) - u8) ./ sqrt(D4))) ...
       .* exp(m * (B.^2 - (a+1)*lam^2*D3) / (2*D3));
L4 = H((-lam-B)/sqrt(A0))  - H((lam-B)/sqrt(A0));

phi_analytic = 1/m * log(L1p + L1m + L2p + L2m + L3p + L3m + L4);

%% Numerical φ^in
z_grid = linspace(-10, 10, 10000);
dz = z_grid(2) - z_grid(1);
Dz = (1/sqrt(2*pi)) * exp(-0.5 * z_grid.^2) * dz;

B_list = linspace(-10, 10, 2001);
phi_numeric = zeros(size(B_list));

for k = 1:length(B_list)
    B_i = B_list(k);
    sqrtA0 = sqrt(A0);

    Z1p = [(lam - B_i)/sqrtA0, (lam*(A1+1) - B_i)/sqrtA0];
    Z1m = [(-lam*(A1+1) - B_i)/sqrtA0, (-lam - B_i)/sqrtA0];
    Z2p = [(lam*(A1+1) - B_i)/sqrtA0, (a*lam*A1 - B_i)/sqrtA0];
    Z2m = [(-a*lam*A1 - B_i)/sqrtA0, (-lam*(A1+1) - B_i)/sqrtA0];
    Z3p = [(a*lam*A1 - B_i)/sqrtA0, Inf];
    Z3m = [-Inf, (-a*lam*A1 - B_i)/sqrtA0];
    Z4  = [(-lam - B_i)/sqrtA0, (lam - B_i)/sqrtA0];

    f_L1 = @(z, sgn) exp(m * (B_i + sqrtA0 * z - sgn * lam).^2 / (2 * A1));
    L1 = sum(f_L1(z_grid(z_grid>=Z1p(1)&z_grid<=Z1p(2)),+1) .* Dz(z_grid>=Z1p(1)&z_grid<=Z1p(2))) + ...
         sum(f_L1(z_grid(z_grid>=Z1m(1)&z_grid<=Z1m(2)),-1) .* Dz(z_grid>=Z1m(1)&z_grid<=Z1m(2)));

    denom = (a - 1) * ((a - 1) * A1 - 1);
    f_L2 = @(z, sgn) exp(m * ((a - 1) * (B_i + sqrtA0 * z) - sgn * a * lam).^2 / (2 * denom));
    L2 = exp(m * lam^2 / (2 * (a - 1))) * ...
        (sum(f_L2(z_grid(z_grid>=Z2p(1)&z_grid<=Z2p(2)),+1) .* Dz(z_grid>=Z2p(1)&z_grid<=Z2p(2))) + ...
         sum(f_L2(z_grid(z_grid>=Z2m(1)&z_grid<=Z2m(2)),-1) .* Dz(z_grid>=Z2m(1)&z_grid<=Z2m(2))));

    f_L3 = @(z) exp(m * (B_i + sqrtA0 * z).^2 / (2 * A1));
    L3 = exp(-m * (a + 1) * lam^2 / 2) * ...
        (sum(f_L3(z_grid(z_grid>=Z3p(1))) .* Dz(z_grid>=Z3p(1))) + ...
         sum(f_L3(z_grid(z_grid<=Z3m(2))) .* Dz(z_grid<=Z3m(2))));

    L4 = sum(Dz(z_grid>=Z4(1)&z_grid<=Z4(2)));

    phi_numeric(k) = (1 / m) * log(L1 + L2 + L3 + L4);
end

%% Plot comparison
figure;
plot(B, phi_analytic, 'b-', 'LineWidth', 2, 'DisplayName', 'Analytic $\phi^{\mathrm{in}}$');
hold on;
plot(B_list, phi_numeric, 'k--', 'LineWidth', 2, 'DisplayName', 'Numerical $\phi^{\mathrm{in}}$');
xlabel('$B_i$', 'Interpreter','latex','FontSize',18);
ylabel('$\phi^{\mathrm{in}}(B_i)$', 'Interpreter','latex','FontSize',18);
title('SCAD $\phi^{\mathrm{in}}$: Analytic vs Numerical', 'Interpreter','latex','FontSize',20);
legend('Interpreter','latex','FontSize',14,'Location','best');
grid on;

phi_all = [B(:), phi_analytic(:), phi_numeric(:)];
writematrix(phi_all, 'SCAD_phi_comparison.txt', 'Delimiter', 'tab');