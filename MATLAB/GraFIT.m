function [G, G_var, Gry, Gru, Gry_var, Gru_var, Y_contr] = GraFIT(r, y, u, W, freqIdentBand, Fs, L, plotResults)

% Computes the frequency response of a dynamical system G based on 
% (filtered) white noise perturbations using the Local Rational Model (LRM) 
% method for SISO/MIMO and open-loop/closed-loop systems.
% For open-loop identification, the transfer function between input r and
% output y is computed. For closed-loop identification, the 3-point method
% is used by computing G = Gry/Gru. 
% Inputs:
% r             : 1 x nr x N input matrix (typically external noise signal).
% y             : ny x nr x N output matrix 
% u             : nu x nr x N output matrix (leave empty for open-loop identification or take plant correction signal for closed-loop identification)
% W             : local frequency window size (number of neighbouring frequency bins)
% FreqIdentBand : 1 x 2 vector defining the frequency band in which to perform the identification
% Fs            : Sampling frequency
% L             : 1 x 3 vector with L = [Ln Lm Ld] with
%   La          : Plant numerator order (default = 2)
%   Lb          : Transient numerator order (default = 2)
%   Ld          : Plant and Transient denominator order (default = 2)
% PlotResults   : [0|1] to visualize results
% Outputs:
% G             : Estimate of G in ny x nr/nu FRD object
% G_var         : Variance of G in ny x nr/nu FRD object
% Gry           : Estimate of Gry in ny x nr FRD object (only for CL identification)
% Gru           : Estimate of Gru in nu x nr FRD object (only for CL identification)
% Gry_var       : Variance of Gry in ny x nr FRD object (only for CL identification)
% Gru_var       : Variance of Gru in nu x nr FRD object (only for CL identification)
% Y_contr       : Output DFT Y and DFT of input contributions G*R, T and V as struct for G, Gru, Gry with
%   G           : DFT of Y, G*R, T and V for open-loop case y/r
%   Gry         : DFT of Y, G*R, T and V for closed-loop case y/r
%   Gru         : DFT of U, G*R, T and V for closed-loop case u/r
% (c) Mathyn van Dael, Eindhoven University of Technology.
% For any questions/bugs, mail m.r.v.dael@tue.nl

% Detect if SISO inputs and wrong input size (to simplify SISO use), otherwise check if MIMO formatting is correct
if size(r,3) == 1 && (size(r,1) == 1 || size(r,2) == 1)
    r = reshape(r, [1, 1, length(r)]);
elseif size(r,1) > size(r,3) || size(r,2) > size(r,3)
    error('Input matrix r is not the correct size. It should have size 1 x nr x Q with Q the number of datapoints')
end

if size(y,3) == 1 && (size(y,1) == 1 || size(y,2) == 1)
    y = reshape(y, [1, 1, length(y)]);
elseif size(y,1) > size(y,3) || size(y,2) > size(y,3)
    error('Output matrix y is not the correct size. It should have size ny x nr x Q with Q the number of datapoints')
end

if size(u,3) == 1 && (size(u,1) == 1 || size(u,2) == 1)
    u = reshape(u, [1, 1, length(u)]);
elseif size(u,1) > size(u,3) || size(u,2) > size(u,3)
    error('Output matrix u is not the correct size. It should have size nu x nr x Q with Q the number of datapoints')
end

% Initialize variables
N = size(r,3);              % Number of datapoints
freqVec = Fs/N*(0:(N/2));   % Define frequency vector
nr = size(r,2);             % Number of inputs (noise signals)
ny = size(y,1);             % Number of outputs (error signals)
nu = size(u,1);             % Number of outputs (correction signals)


% Check input arguments
if nargin < 6 || isempty(freqIdentBand) % Check if frequency band to identify is given, if not then use whole range
    freqIdentBand = [freqVec(1) freqVec(end)];
end

if nargin < 7 || isempty(L) % Check if coefficients are given, if not then set default
    L = [2, 2, 2]; % Default values for [La, Lb, Ld]
end
[La, Lb, Ld] = deal(L(1), L(2), L(3));

if nargin < 8 && nargout > 0 % Check if plotResults value is set
    plotResults = 0;
elseif nargin < 8 && nargout == 0
    plotResults = 1;
end

% Check if frequency window is sufficiently large
if W <= ceil(((La+1)+Lb+ny*Ld)/2)
    W = ceil((La+1+Lb+ny*Ld)/2)+1;
    disp(['Frequency window has been changed to W = ', num2str(W)]);
end


% Get indices of starting and end frequency
[~, fStartIdx] = min(abs(freqVec-freqIdentBand(1)));
[~, fEndIdx] = min(abs(freqVec-freqIdentBand(2)));

% Check if starting frequency bin + window size is not smaller than lowest frequency bin
if fStartIdx-W < 1
    fStartIdx = W + 1; 
end

if fEndIdx+W > (N/2+1)
    fEndIdx = N/2+1-W; 
end

% Define identified frequency range vector
nIdentFreqPoints = length(fStartIdx:fEndIdx);

% Define vector containing window numbers
winVec = (-W:W);

% Precompute static components of K
Ka = winVec.^((0:La)'); % Input-related static terms
Kb = winVec.^((0:Lb)'); % Transient-related static terms

if isempty(u) % Open-loop identification

    Kd = -repmat(winVec, Ld*ny, 1).^reshape(repmat((1:Ld)', 1, ny)', [], 1); % Output-related static terms

    for iIn = 1:nr
        [G(:, iIn), G_var(:, iIn), Y_contr.G(:, iIn)] = LRM_SIMO(squeeze(r(1,iIn,:))', squeeze(y(:,iIn,:)), W, ny, freqVec, [fStartIdx fEndIdx], La, Ld, N, winVec, Ka, Kb, Kd);
    end

    Gry = []; Gru = []; Gry_var = []; Gru_var = [];

else % Closed-loop identification

    % Initialize matrices
    Grz_cov = zeros((ny^2+nu^2), (ny^2+nu^2), nIdentFreqPoints); 
    G_cov = zeros(ny^2, nu^2, nIdentFreqPoints);
    G_var = zeros(ny, nu, nIdentFreqPoints);

    Kd = -repmat(winVec, Ld*ny*2, 1).^reshape(repmat((1:Ld)', 1, ny*2)', [], 1); % Output-related static terms

    for iIn = 1:nr
        z = [reshape(y(:,iIn,:), [ny, N]); reshape(u(:,iIn,:), [nu, N])]; % Create input vector z = [y; u]
        [Grz(:, iIn), Grz_var(:, iIn), Grz_Y_contr(:, iIn), Grz_cov(((iIn-1)*ny*2+1):iIn*ny*2, ((iIn-1)*ny*2+1):iIn*ny*2, :)] = LRM_SIMO(squeeze(r(1,iIn,:))', z, W, ny*2, freqVec, [fStartIdx fEndIdx], La, Ld, N, winVec, Ka, Kb, Kd);
    end
    
    % Extract y and u outputs from Z vector
    Gry = Grz(1:ny,:); Gry_var = Grz_var(1:ny,:); Y_contr.Gry = Grz_Y_contr(1:ny,:);
    Gru = Grz((ny+1):end,:); Gru_var = Grz_var((ny+1):end,:); Y_contr.Gru = Grz_Y_contr((ny+1):end,:);
    
    G = Gry*inv(Gru);

    % Compute variance of G based on covariance of closed-loop system
    for iFreq = 1:length(Gru.Frequency)
        C = (kron(inv(Gru.ResponseData(:,:,iFreq)).',[eye(ny), -G.ResponseData(:,:,iFreq)]));
        G_cov(:, :, iFreq) =  C*Grz_cov(:,:,iFreq)*C';
        G_var(:,:,iFreq) = reshape(diag(G_cov(:, :, iFreq)), ny, ny);
    end
    G_var = frd(G_var, Gru.Frequency, 'FrequencyUnit', 'Hz');

end

% Plot results
if plotResults

    iAxes = 1;
    figure;
    for iOut = 1:ny
        for iIn = 1:nr
            hax(iAxes) = subplot(ny, nr, iAxes);
    
            % Extract G and variance for this input-output pair
            G_Mag = abs(squeeze(G(iOut, iIn).ResponseData))';
    
            % Compute confidence bounds
            R = sqrt(3) .* sqrt(squeeze(G_var.ResponseData(iOut, iIn,:))');
            G_Upper = G_Mag + R;
            G_Lower = G_Mag - R;
            G_Lower(G_Lower < 1e-12) = 1e-12; % Avoid negative lower bounds
    
            % Plot bounds and transfer function
            confFreq = [G(iOut, iIn).Frequency', fliplr(G(iOut, iIn).Frequency')];
            confRadius = [G_Upper, fliplr(G_Lower)];
        
            hold on;
            plot(G(iOut, iIn).Frequency, G_Mag, 'LineWidth', 2, 'Color', [0.7 0.0 0.0]);
            p = fill(confFreq, confRadius, 'r');
            p.FaceColor = [0.7, 0.0, 0.0];
            p.FaceAlpha = 0.3;
            p.EdgeColor = 'none';        
            ylim([0.1*10^floor(log10(min(G_Mag))), 10*10^ceil(log10(max(G_Mag)))]);
            set(gca, 'XScale', 'log')
            set(gca, 'YScale', 'log')
            set(hax(iAxes).XLabel, 'FontSize', 17)
            grid on;
            iAxes = iAxes + 1;
        end
    end
    for iIn = 1:nu
        set(hax(iIn).Title, 'String', ['\textbf{Input} ' num2str(iIn)], 'Interpreter', 'Latex', 'FontSize', 14)
        set(hax((iIn-1)*ny+1).YLabel, 'String', {['\textbf{Output} ' num2str(iIn)]; 'Magnitude [dB]'}, 'Interpreter', 'Latex')
        set(hax((iIn-1)*ny+1).YLabel, 'FontSize', 14)
        set(hax(iIn+nu*(ny-1)).XLabel, 'String', 'Frequency [Hz]', 'FontSize', 16)
    end
    sgtitle('Frequency response magnitude with uncertainty for G', 'FontSize', 18)
    linkaxes(hax, 'x'); clear hax;

    if ~isempty(u)

        iAxes = 1;        
        figure;
        for iOut = 1:nu
            for iIn = 1:nr
                hax(iAxes) = subplot(nu, nr, iAxes);
        
                % Extract G and variance for this input-output pair
                Gru_Mag = abs(squeeze(Gru(iOut, iIn).ResponseData))';
        
                % Compute confidence bounds
                R = sqrt(3) .* sqrt(squeeze(Gru_var(iOut, iIn).ResponseData)');
                Gru_Upper = Gru_Mag + R;
                Gru_Lower = Gru_Mag - R;
                Gru_Lower(Gru_Lower < 1e-12) = 1e-12; % Avoid negative lower bounds
        
                % Plot bounds and transfer function
                confFreq = [Gru(iOut, iIn).Frequency', fliplr(Gru(iOut, iIn).Frequency')];
                confRadius = [Gru_Upper, fliplr(Gru_Lower)];
            
                hold on;
                plot(Gru(iOut, iIn).Frequency, Gru_Mag, 'LineWidth', 2, 'Color', [0.7 0.0 0.0]);
                p = fill(confFreq, confRadius, 'r');
                p.FaceColor = [0.7, 0.0, 0.0];
                p.FaceAlpha = 0.3;
                p.EdgeColor = 'none';        
                ylim([0.1*10^floor(log10(min(Gru_Mag))), 10*10^ceil(log10(max(Gru_Mag)))]);
                set(gca, 'XScale', 'log')
                set(gca, 'YScale', 'log')
                set(hax(iAxes).XLabel, 'FontSize', 17)
                grid on;
                iAxes = iAxes + 1;
            end
        end      
        for iIn = 1:nu
            set(hax(iIn).Title, 'String', ['\textbf{Input} ' num2str(iIn)], 'Interpreter', 'Latex', 'FontSize', 14)
            set(hax((iIn-1)*ny+1).YLabel, 'String', {['\textbf{Output} ' num2str(iIn)]; 'Magnitude [dB]'}, 'Interpreter', 'Latex')
            set(hax((iIn-1)*ny+1).YLabel, 'FontSize', 14)
            set(hax(iIn+nu*(ny-1)).XLabel, 'String', 'Frequency [Hz]', 'FontSize', 16, 'Interpreter', 'Latex')
        end
        sgtitle('Frequency response magnitude with uncertainty for Gru', 'FontSize', 18, 'Interpreter', 'Latex')
        linkaxes(hax, 'x'); clear hax;

        iAxes = 1;
        figure;
        for iOut = 1:ny
            for iIn = 1:nr
                hax(iAxes) = subplot(ny, nr, iAxes);
        
                % Extract G and variance for this input-output pair
                Gry_Mag = abs(squeeze(Gry(iOut, iIn).ResponseData))';
        
                % Compute confidence bounds
                R = sqrt(3) .* sqrt(squeeze(Gry_var(iOut, iIn).ResponseData)');
                Gry_Upper = Gry_Mag + R;
                Gry_Lower = Gry_Mag - R;
                Gry_Lower(Gry_Lower < 1e-12) = 1e-12; % Avoid negative lower bounds
        
                % Plot bounds and transfer function
                confFreq = [Gry(iOut, iIn).Frequency', fliplr(Gry(iOut, iIn).Frequency')];
                confRadius = [Gry_Upper, fliplr(Gry_Lower)];
            
                hold on;
                plot(Gry(iOut, iIn).Frequency, Gry_Mag, 'LineWidth', 2, 'Color', [0.7 0.0 0.0]);
                p = fill(confFreq, confRadius, 'r');
                p.FaceColor = [0.7, 0.0, 0.0];
                p.FaceAlpha = 0.3;
                p.EdgeColor = 'none';        
                ylim([0.1*10^floor(log10(min(Gry_Mag))), 10*10^ceil(log10(max(Gry_Mag)))]);
                set(gca, 'XScale', 'log')
                set(gca, 'YScale', 'log')
                set(hax(iAxes).XLabel, 'FontSize', 17)
                grid on;
                iAxes = iAxes + 1;
            end
        end
        for iIn = 1:nu
            set(hax(iIn).Title, 'String', ['\textbf{Input} ' num2str(iIn)], 'Interpreter', 'Latex', 'FontSize', 14)
            set(hax((iIn-1)*ny+1).YLabel, 'String', {['\textbf{Output} ' num2str(iIn)]; 'Magnitude [dB]'}, 'Interpreter', 'Latex')
            set(hax((iIn-1)*ny+1).YLabel, 'FontSize', 14)
            set(hax(iIn+nu*(ny-1)).XLabel, 'String', 'Frequency [Hz]', 'FontSize', 16, 'Interpreter', 'Latex')
        end
        sgtitle('Frequency response magnitude with uncertainty for Gry', 'FontSize', 18, 'Interpreter', 'Latex')
        linkaxes(hax, 'x'); clear hax;

        iAxes = 1;
        figure;
        for iOut = 1:nu
            for iIn = 1:nr
                hax(iAxes) = subplot(nu, nr, iAxes);
                hold on;
                scatter(Gru(iOut, iIn).Frequency, abs(Y_contr.Gru(iOut,iIn).Y), 30, [0.0 0.5 0.0], 'filled', 'o', 'DisplayName', '$U$');
                scatter(Gru(iOut, iIn).Frequency, abs(Y_contr.Gru(iOut,iIn).GR), 20, [0.0 0.6 0.6], 'o', 'DisplayName', '$G_{ru} \cdot R$');
                scatter(Gru(iOut, iIn).Frequency, abs(Y_contr.Gru(iOut,iIn).T), 20, [0.7 0.7 0.0], 'o', 'DisplayName', '$T$');
                scatter(Gru(iOut, iIn).Frequency, abs(Y_contr.Gru(iOut,iIn).V), 20, [0.5 0.0 0.5], 'o', 'DisplayName', '$V$');
                set(gca, 'XScale', 'log')
                set(gca, 'YScale', 'log')
                set(gca, 'FontSize', 16)
                grid on;
                iAxes = iAxes + 1;
            end
        end
        legend('Interpreter', 'Latex', 'Location', 'Best');
        for iIn = 1:nu
            set(hax(iIn).Title, 'String', ['\textbf{Input} ' num2str(iIn)], 'Interpreter', 'Latex', 'FontSize', 14)
            set(hax((iIn-1)*ny+1).YLabel, 'String', {['\textbf{Output} ' num2str(iIn)]; 'Magnitude [dB]'}, 'Interpreter', 'Latex')
            set(hax((iIn-1)*ny+1).YLabel, 'FontSize', 14)
            set(hax(iIn+nu*(ny-1)).XLabel, 'String', 'Frequency [Hz]', 'FontSize', 16, 'Interpreter', 'Latex')
        end
        sgtitle('Output contributions for Gru', 'FontSize', 18, 'Interpreter', 'Latex')
        linkaxes(hax, 'x'); clear hax;

        iAxes = 1;
        figure;
        for iOut = 1:ny
            for iIn = 1:nr
                hax(iAxes) = subplot(ny, nr, iAxes);
                hold on;
                scatter(Gry(iOut, iIn).Frequency, abs(Y_contr.Gry(iOut,iIn).Y), 30, [0.0 0.5 0.0], 'filled', 'o', 'DisplayName', '$Y$');
                scatter(Gry(iOut, iIn).Frequency, abs(Y_contr.Gry(iOut,iIn).GR), 20, [0.0 0.6 0.6], 'o', 'DisplayName', '$G_{ry} \cdot R$');
                scatter(Gry(iOut, iIn).Frequency, abs(Y_contr.Gry(iOut,iIn).T), 20, [0.7 0.7 0.0], 'o', 'DisplayName', '$T$');
                scatter(Gry(iOut, iIn).Frequency, abs(Y_contr.Gry(iOut,iIn).V), 20, [0.5 0.0 0.5], 'o', 'DisplayName', '$V$');
                set(gca, 'XScale', 'log')
                set(gca, 'YScale', 'log')
                set(gca, 'FontSize', 16)
                grid on;
                iAxes = iAxes + 1;
            end
        end
        legend('Interpreter', 'Latex', 'Location', 'Best');
        for iIn = 1:nu
            set(hax(iIn).Title, 'String', ['\textbf{Input} ' num2str(iIn)], 'Interpreter', 'Latex', 'FontSize', 14)
            set(hax((iIn-1)*ny+1).YLabel, 'String', {['\textbf{Output} ' num2str(iIn)]; 'Magnitude [dB]'}, 'Interpreter', 'Latex')
            set(hax((iIn-1)*ny+1).YLabel, 'FontSize', 14)
            set(hax(iIn+nu*(ny-1)).XLabel, 'String', 'Frequency [Hz]', 'FontSize', 16, 'Interpreter', 'Latex')
        end
        sgtitle('Output contributions for Gry', 'FontSize', 18, 'Interpreter', 'Latex')
        linkaxes(hax, 'x'); clear hax;

    else

        iAxes = 1;
        figure;
        for iOut = 1:ny
            for iIn = 1:nr
                hax(iAxes) = subplot(ny, nr, iAxes);
                hold on;
                scatter(G(iOut, iIn).Frequency, abs(Y_contr.G(iOut,iIn).Y), 30, [0.0 0.5 0.0], 'filled', 'o', 'DisplayName', '$Y$');
                scatter(G(iOut, iIn).Frequency, abs(Y_contr.G(iOut,iIn).GR), 20, [0.0 0.6 0.6], 'o', 'DisplayName', '$G \cdot R$');
                scatter(G(iOut, iIn).Frequency, abs(Y_contr.G(iOut,iIn).T), 20, [0.7 0.7 0.0], 'o', 'DisplayName', '$T$');
                scatter(G(iOut, iIn).Frequency, abs(Y_contr.G(iOut,iIn).V), 20, [0.5 0.0 0.5], 'o', 'DisplayName', '$V$');
                set(gca, 'XScale', 'log')
                set(gca, 'YScale', 'log')
                set(gca, 'FontSize', 16)
                grid on;
                iAxes = iAxes + 1;
            end
        end
        legend('Interpreter', 'Latex', 'Location', 'Best');
        for iIn = 1:nu
            set(hax(iIn).Title, 'String', ['\textbf{Input} ' num2str(iIn)], 'Interpreter', 'Latex', 'FontSize', 14)
            set(hax((iIn-1)*ny+1).YLabel, 'String', {['\textbf{Output} ' num2str(iIn)]; 'Magnitude [dB]'}, 'Interpreter', 'Latex')
            set(hax((iIn-1)*ny+1).YLabel, 'FontSize', 14)
            set(hax(iIn+nu*(ny-1)).XLabel, 'String', 'Frequency [Hz]', 'FontSize', 16, 'Interpreter', 'Latex')
        end
        sgtitle('Output contributions for G', 'FontSize', 18, 'Interpreter', 'Latex')
        linkaxes(hax, 'x'); clear hax;

    end

end

end


function [G_FRD, G_var_FRD, Y_contr, G_cov] = LRM_SIMO(r, y, W, ny, freqVec, freqIdx, Ln, Ld, N, winVec, Ka, Kb, Kd)

% Ensure u is a row vector
if size(r, 1) ~= 1
    r = r';
end

% Ensure output channels are row vectors
if size(y,1) > size(y,2)
    y = y';
end

[fStartIdx, fEndIdx] = deal(freqIdx(1), freqIdx(2));

% Create FFT of signals
R_twoSided = fft(r)/sqrt(2*N);
Y_twoSided = fft(y, [], 2)/sqrt(2*N);

R = R_twoSided(1, 1:(N/2+1));
Y = Y_twoSided(:, 1:(N/2+1));

% Define identified frequency range vector
freqVecIdent = freqVec(fStartIdx:fEndIdx);
nIdentFreqPoints = length(fStartIdx:fEndIdx);

% Initialize empty output matrices
G = zeros(ny, nIdentFreqPoints);
T = zeros(ny, nIdentFreqPoints);
V = zeros(ny, nIdentFreqPoints);
G_var = zeros(ny, nIdentFreqPoints);
G_cov = zeros(ny, ny, nIdentFreqPoints);


for kIdx = 1:nIdentFreqPoints
    k = kIdx + fStartIdx - 1;

    % Define the local frequency window
    localFreqWin = k+winVec;
    
    % Obtain local Input/Output data
    Rw = R(:,localFreqWin);
    Yw = Y(:,localFreqWin);

    % Create Kw matrix
    Kw = [Ka .* Rw;                   % Input U
          Kb;                         % Transient T
          Kd .* repmat(Yw, [Ld, 1])]; % Output Y

    %  Apply scaling Pintelon 2012 (7-25)
    Dscale = diag(vecnorm(Kw, 2, 2));
    Kw = Dscale\Kw;

    % Compute least squares solution
    [Uk,Sk,Vk] = svd(Kw'); % better computational feasability Pintelon 2012 (7-24)
    Theta = Yw*Uk/Sk'*Vk';

    % Compute residuals before scaling Theta back
    Vn = Yw-Theta*Kw; 

    % Rescale Theta and Kw
    Theta = Theta/Dscale;
    Kw = Dscale*Kw; 

    % Store estimated G, T and V
    G(:,kIdx) = Theta(1:ny,1);
    T(:,kIdx) = Theta(1:ny,Ln+2);
    V(:,kIdx) = Vn(1:ny,W+1);

    % Compute (co-)variances
    q = 2*W+1-rank(Kw); % Compute number of dofs

    V_cov = Vn*Vn'/q; % Covariance matrix of noise
    S_cov = (Kw'/(Kw*Kw'))*[eye(1);zeros(size(Kw,1)-1,1)];

    G_cov(:,:,kIdx) = kron(S_cov'*S_cov,V_cov); % Covariance matrix of G

    G_var(:,kIdx) = diag(squeeze(G_cov(:,:,kIdx))); % Take diagonal elements to find variances
    
end

% Reduce DFT vectors to only the identified frequencies
R = R(:,fStartIdx:fEndIdx);
Y = Y(:,fStartIdx:fEndIdx);

% Loop through outputs to create FRD objects
for iOut = 1:ny
    G_FRD(iOut,1) = frd(G(iOut,:), freqVecIdent, 'FrequencyUnit', 'Hz');
    G_var_FRD(iOut,1) = frd(G_var(iOut,:), freqVecIdent, 'FrequencyUnit', 'Hz');
    Y_contr(iOut,1).Y = Y(iOut,:);
    Y_contr(iOut,1).GR = G(iOut, :).*R;
    Y_contr(iOut,1).T = T(iOut,:);
    Y_contr(iOut,1).V = V(iOut,:);
end

end