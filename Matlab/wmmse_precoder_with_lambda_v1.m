function W = wmmse_precoder_with_lambda_v1(H, Pt, sigma2, maxIter, W0)
%WMMSE_PRECODER_WITH_LAMBDA  MU‐MIMO WMMSE precoder with robust λ-search
%
%   W = wmmse_precoder_with_lambda(H, Pt, sigma2, maxIter)
%   W = wmmse_precoder_with_lambda(H, Pt, sigma2, maxIter, W0)
%
% This normalization is only for the WMMSE algo. Its not reflected outside
H=H/norm(H); 
[K, M] = size(H);

% 1) Initialize W
if nargin>=5 && ~isempty(W0)
    W = W0;% * sqrt(Pt)/norm(W0,'fro');
else
    W = (randn(M,K)+1j*randn(M,K));
    W = W * sqrt(Pt)/norm(W,'fro');
end

for it = 1:maxIter
    %% 2) MMSE combiners U_i
    U = zeros(1,K);
    for i = 1:K
        hi = H(i,:);
        C  = sum(abs(hi*W).^2) + sigma2;
        U(i) = conj(hi*W(:,i)) / C;
    end

    %% 3) MSE weights V_i
    V = zeros(1,K);
    for i = 1:K
        hi = H(i,:);
        e = 1 ...
          - U(i)*(hi*W(:,i)) ...
          - conj(U(i)*(hi*W(:,i))) ...
          + abs(U(i))^2*(sum(abs(hi*W).^2) + sigma2);
        V(i) = 1 / e;
    end

    %% 4) Build Q and B
    Q = zeros(M,M);
    for j = 1:K
        hj = H(j,:);
        Q = Q + V(j)*abs(U(j))^2 * (hj' * hj);
    end
    B = zeros(M,K);
    for i = 1:K
        B(:,i) = conj(U(i))*V(i)*H(i,:)';
    end

    %% 5) λ-search with lam_low > 0
    powerErr = @(lam) norm((Q + lam*eye(M))\B,'fro')^2 - Pt;

    lam_low  = 1e-6;              % small positive to keep Q+lam*I well-conditioned
    f_low    = powerErr(lam_low);

    % if at lam_low we already under-shoot power, just use lam_low
    if f_low <= 0
        lam = lam_low;
    else
        lam_high = lam_low;
        f_high   = f_low;
        % expand until f_high < 0
        while f_high > 0
            lam_high = lam_high * 10;
            f_high   = powerErr(lam_high);
            if lam_high > 1e12
                error('Could not bracket lambda in wmmse_precoder_with_lambda');
            end
        end
        lam = fzero(powerErr, [lam_low, lam_high]);
    end

    %% 6) Update W
    W = (Q + lam*eye(M)) \ B;
end
end
