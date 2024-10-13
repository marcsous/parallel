function [x lambda errvec] = fistaN(AA,Ab,sparsity,tol,maxit,Q,svals)
% [x lambda errvec] = fista(AA,Ab,sparsity,tol,maxit,Q)
%
% Solves the following problem via FISTA (normal eq):
%
%   minimize (1/2)||AAx-Ab||_2^2 + lambda*||Qx||_1
%
% -AA is an [n x n] matrix (or function handle)
% -sparsity is the fraction of zeros (0.1=10% zeros)
% -tol/maxit are tolerance and max. no. iterations
% -Q is a wavelet transform (Q*x and Q'*x - see HWT)
% -svals is [largest smallest] singular values of AA
%
% -lambda that yields the required sparsity (scalar)
% -errvec is the rel. change at each iteration (vector)
%
%% check arguments
if nargin<3 || nargin>7
    error('Wrong number of input arguments');
end
if nargin<4 || isempty(tol)
    tol = 1e-3;
end
if nargin<5 || isempty(maxit)
    maxit = 100;
end
if nargin<6 || isempty(Q)
    Q = 1;
end
if nargin<7 || isempty(svals)
    svals = [];
else
    normAA = svals(1);
end
if isnumeric(AA)
    AA = @(arg) AA*arg;
end
if ~iscolumn(Ab)
    error('Ab must be a column vector');
end
if ~isscalar(sparsity) || sparsity<0 || sparsity>1
    error('sparsity must be a scalar between 0 and 1.');
end

%% solve by FISTA
time = tic();

z = Ab;
t = 1;

for iter = 1:maxit
    
    Az = AA(z);
    
    % steepest descent along Ab
    if iter==1
        alpha = (z'*z) / (Az'*Az);
        z = alpha * z;
        x = z;
    end

    % power iteration to get norm(AA)   
    if ~exist('normAA','var')      
        tmp = Az/norm(Az);
        for k = 1:20
            if k>1
                tmp = tmp/normAA(k-1);
            end
            tmp = AA(tmp);
            normAA(k) = norm(tmp);   
        end
        normAA = norm(tmp);
    end
    
    z = z + (Ab - Az) / normAA;
    
    xold = x;
    
    x = Q*z;
    [x lambda(iter)] = shrinkage(x,sparsity);
    x = Q'*x;

    errvec(iter) = norm(x-xold)/norm(x);

    if errvec(iter) < tol
        break;
    end
    
    % FISTA-ADA
    %if numel(svals)==2
    %    r = 4 * (1-sqrt(prod(svals)))^2 / abs(1-prod(svals));
    %else
        r = 4;
    %end
    t0 = t;
    t = (1+sqrt(1+r*t^2))/2;
    z = x + ((t0-1)/t) * (x-xold);
    
end

% report convergence
if iter < maxit
    fprintf('%s converged at iteration %i to a solution with relative error %.1e. ',mfilename,iter,errvec(iter));
else
    fprintf('%s stopped at iteration %i with relative error %.1e without converging to the desired tolerance %.1e. ',mfilename,iter,errvec(iter),tol);  
end
toc(time);

%% shrink based on sparsity => return lambda
function [z lambda] = shrinkage(z,sparsity)

absz = abs(z);
v = sort(absz,'ascend');
lambda = interp1(v,sparsity*numel(z),'linear',0);

z = sign(z) .* max(absz-lambda, 0); % complex ok
