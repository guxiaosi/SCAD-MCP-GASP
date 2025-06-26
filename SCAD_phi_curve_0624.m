% -------------------------------------------------------------------------
% SCAd φ^{in}_{i}(B_i,A0,A1,m), ∂φ/∂B, and \hat x_i on one figure
% with protection for A0=0 or tiny denominators.
% Example parameters: a=3.7, A0 can be 0, A1=1, m=1, λ=1
% By Xiaosi, Gu; Date: 2025-06-25
% -------------------------------------------------------------------------
clear; clc; close all;

%% PARAMETERS (Set A0 = 0 to check stability behavior)
a   = 3.7;        % SCAD parameter a
A0  = 0.3;      % Parameter A0 (can be set to 0)
A1  = 1;        % A1
m   = 0.3;        % Parisi parameter
lam = 1;        % Threshold λ

%% PROTECTION EPS
eps = 1e-12;
% Protect A0 if too small
if abs(A0) < eps
    A0 = eps;
end
%% B GRID
B = linspace(-5,5,2001);
%% FINITE‐DIFFERENCE STEP FOR A1
epsA1 = 1e-6;
%% compute φ at A1, A1+epsA1, A1-epsA1
phi0  = compute_phi_scad(A1,  B, a, A0, m, lam);
phi_p = compute_phi_scad(A1+epsA1, B, a, A0, m, lam);
phi_m = compute_phi_scad(A1-epsA1, B, a, A0, m, lam);
%% ∂φ/∂A1
dphi_dA1 = (phi_p - phi_m) / (2*epsA1);
%% ∂φ/∂B by finite difference on φ0
dB = B(2)-B(1);
dphi_dB = zeros(size(phi0));
dphi_dB(2:end-1) = (phi0(3:end) - phi0(1:end-2)) / (2*dB);
dphi_dB(1)       = dphi_dB(2);
dphi_dB(end)     = dphi_dB(end-1);
%% ∂²φ/∂B²
d2phi_dB2 = zeros(size(phi0));
d2phi_dB2(2:end-1) = (phi0(3:end) - 2*phi0(2:end-1) + phi0(1:end-2)) / (dB^2);
d2phi_dB2(1)   = d2phi_dB2(2);
d2phi_dB2(end) = d2phi_dB2(end-1);
%% Compute Δ0 = -2·∂φ/∂A1 - (∂φ/∂B)^2
Delta0 = -2*dphi_dA1 - dphi_dB.^2;
% —— Truncate negatives —— 
%Delta0(Delta0 < 0) = 0;
%% Compute Δ1 = ∂²φ/∂B² - m·Δ0
Delta1 = d2phi_dB2 - m * Delta0;
% —— Truncate negatives —— 
%Delta1(Delta1 < 0) = 0;
%% Compute SCAD estimator \hat x
xhat = zeros(size(B));
maskI  = (abs(B) > lam) & (abs(B) <= lam*(A1+1));
maskII = (abs(B) > lam*(A1+1)) & (abs(B) <=a*lam*(A1));
maskIII= abs(B) > a*lam*(A1);
xhat(maskI)  = ( B(maskI) - lam.*sign(B(maskI)) ) ./ max(A1, eps);
xhat(maskII) =   (B(maskII)*(a-1)-a*lam.*sign(B(maskII))) ./ max(A1*(a-1)-1, eps);
xhat(maskIII) =  B(maskIII)./max(A1, eps);
%% Plot φ, ∂φ/∂B, and \hat x on one figure
figure; hold on;
hPhi = plot(B,    phi0,     'b-',   'LineWidth',2, 'DisplayName','\phi^{in}_i'); 
dbhPhi = plot(B,    dphi_dB,  'r-',   'LineWidth',2, 'DisplayName','\partial_{B}\phi');
%plot(B,    dphi_dA1, 'g--',  'LineWidth',2, 'DisplayName','\partial_{A_1}\phi');
d0hphi = plot(B,    Delta0,   'k:',   'LineWidth',2, 'DisplayName','\Delta_0');
d1hphi = plot(B,    Delta1, '-.','Color', [0.5 0.5 0.5],  'LineWidth',2, 'DisplayName','\Delta_1');
hatx_amp = plot(B,    xhat,     'm-',  'LineWidth',2, 'DisplayName','\hat x_i^{AMP}');
% Draw region boundaries
xline(-lam,  '--k');
xline( lam,  '--k');
xline(-lam*(A1 +1), '--k');
xline( lam*(A1 +1), '--k');
xline(-a*lam*(A1), '--k');
xline( a*lam*(A1), '--k');
%% Label regions I+, I–, II+, II–, III
yl   = ylim;  
ypos = yl(2) - 0.05*(yl(2) - yl(1));
midIplus   = mean([lam,               lam*(A1+1)]);
midIminus  = mean([-lam*(A1 +1), -lam]);
midIIplus   = mean([lam*(A1+1),     a*lam*(A1)]);
midIIminus  = mean([-a*lam*(A1), -lam*(A1+1)]);
midIIIplus  = mean([a*lam*(A1),   max(B)]);
midIIIminus = mean([min(B),            -a*lam*(A1)]);
midIV     = 0;
text(midIplus,   ypos, '$\mathrm{I}^+$',  'Interpreter','latex','FontSize',16,'HorizontalAlignment','center');
text(midIminus,  ypos, '$\mathrm{I}^-$',  'Interpreter','latex','FontSize',16,'HorizontalAlignment','center');
text(midIIplus,  ypos, '$\mathrm{II}^+$', 'Interpreter','latex','FontSize',16,'HorizontalAlignment','center');
text(midIIminus, ypos, '$\mathrm{II}^-$', 'Interpreter','latex','FontSize',16,'HorizontalAlignment','center');
text(midIIIplus,  ypos, '$\mathrm{III}^+$', 'Interpreter','latex','FontSize',16,'HorizontalAlignment','center');
text(midIIIminus, ypos, '$\mathrm{III}^-$', 'Interpreter','latex','FontSize',16,'HorizontalAlignment','center');
text(midIV,     ypos, '$\mathrm{IV}$',  'Interpreter','latex','FontSize',16,'HorizontalAlignment','center');
%% Combined legend and labels
legend([hPhi,dbhPhi,hatx_amp,d0hphi,d1hphi],...
  { ...
    '$\phi^{\mathrm{in}}_{i,\mathrm{SCAD}}(B_i,A_0,A_1,m)$', ...
    '$\partial_{B_i}\phi^{\mathrm{in}}_i(B_i)=\hat x_{i}^{\mathrm{ASP}}$', ...
    '$\hat x_{i}^{\mathrm{AMP}}$', ...
    '$\Delta_{0,i}$', ...
    '$\Delta_{1,i}$', ...
  }, ...
  'Interpreter','latex','FontSize',18,'Location','best');

xlabel('$B_i$','Interpreter','latex','FontSize',20);
ylabel('$\phi^{\mathrm{in}}_{i,\rm{SCAD}},\ \hat x_{i}^{\mathrm{ASP}},\ \hat x_{i}^{\mathrm{AMP}},\ \Delta_{0,i},\ \Delta_{1,i}$', ...
    'Interpreter','latex','FontSize',20);
title(['$\mathrm{SCAD\ input\ channel\ }(', ...
    'a=',   num2str(a),     ',\ ', ...
    '\lambda=', num2str(lam), ',\ ', ...
    'm=',   num2str(m),     ',\ ', ...
    'A_0=', num2str(A0),    ',\ ', ...
    'A_1=', num2str(A1),    ')$'], ...
  'Interpreter','latex','FontSize',20);
grid on;
hold off;
%ylim([-11, 11]);
%xlim([-10, 10]);
%% ----------------------------- LOCAL FUNCTION -----------------------------
function phi = compute_phi_scad(A1, B, a, A0, m, lam)
% Compute φ^{in}_i(B_i) for SCAD input channel
%% PROTECT A0
eps_val = 1e-12;
if abs(A0) < eps_val
    A0 = eps_val;
end

%% GAUSSIAN TAIL FUNCTION
H = @(x) 0.5 * erfc(x ./ sqrt(2));

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
u7 = qim_val ./ p;

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
