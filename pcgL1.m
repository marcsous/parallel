function x = pcgL1(A, b, lambda, maxit, varargin)
% x = pcgL1(A, b, lambda, maxit, varargin)
%
% Solves the following problem via ADMM:
%
%   minimize (1/2)*||Ax-b||_2^2 + λ*||x||_1,
%
% where A is symmetric positive definite (same as pcg).
% Best used with anonymous functions, A = @(x)myfunc(x)
% where myfunc(x) returns (A*x).
%
% rho is the augmented Lagrangian parameter.
%
% alpha is the over-relaxation parameter (typical values for alpha are
% between 1.0 and 1.8).
%
% Derived from lasso_lsqr.m at
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
%
%% default options

opts.tol = 1e-6;
opts.rho = 1;
opts.alpha = 1;

% varargin handling (must be option/value pairs)
for k = 1:2:numel(varargin)
    if k==numel(varargin) || ~ischar(varargin{k})
        error('''varargin'' must be option/value pairs.');
    end
    if ~isfield(opts,varargin{k})
        warning('''%s'' is not a valid option.',varargin{k});
    end
    opts.(varargin{k}) = varargin{k+1};
end

%% check arguments

if nargin<2
    error('Not enough input arguments: b missing');
end
if nargin<3 || isempty(lambda)
    lambda = 0;
    warning('Not enough input arguments. Setting lambda=%f',lambda);
end
if nargin<4 || isempty(maxit)
    maxit = 20;
    warning('Not enough input arguments. Setting maxit=%f',maxit);
end
if ~iscolumn(b)
    error('b argument must be a column vector');
else
    n = numel(b);
end

% not intended for matrix inputs but they are supported
if isnumeric(A)
    A = @(arg) A * arg;
end

% check A is square [n x n]
try
    tmp = A(b);
catch
    error('A(x) failed when passed a vector of length %i',n);
end   
if ~isequal(size(tmp),[n 1])
    error('A(x) did not return a vector of length %i',n);
end

% check positive definiteness (50% chance of catching error)
tmp = b'*tmp; % x'Ax > 0
if abs(imag(tmp)) > n*eps(tmp) || real(tmp) < -n*eps(tmp)
    warning('Matrix operator A may not be positive definite.');
end
clear tmp;

%% ADMM solver

x = zeros(n,1,'like',b);
z = zeros(n,1,'like',b);
u = zeros(n,1,'like',b);

normb = norm(b);

for k = 1:maxit

    % x-update with pcg - doesn't need to be very accurate 
    Ak = @(x)A(x) + opts.rho*x;
    bk = b + opts.rho*(z-u);
    [x flag relres iters resvec] = pcg(Ak,bk,[],[],[],[],x);

    if flag~=0
        warning('PCG problem (iter=%i flag=%i)',k,flag);
    end
    
    % z-update with relaxation
    zold = z;
    x_hat = opts.alpha*x + (1 - opts.alpha)*zold;
    z = shrinkage(x_hat + u, lambda/opts.rho);

    % catch bad cases and overwrite lambda
    if k==1
        target = median(abs(x_hat+u))*opts.rho; % 50% sparsity
        if all(z(:)==0)
            fprintf('Too sparse (all zero), reducing lambda to %.1e.\n',target);
        end
        if all(z(:)~=0)
            lambda = median(abs(x_hat+u))*opts.rho;
            fprintf('Not sparse (all nonzero), increasing lambda to %.1e.\n',target);
        end
        lambda = target;
    end
    
    u = u + (x_hat - z);

    % check convergence
    if norm(x-z) <  opts.tol*normb
        break;
    end

end

% report sparsity
fprintf('%s: nonzeros %.1f%% (lambda=%.1e)\n',mfilename,100*nnz(z)/numel(z),lambda);

% check convergence
if norm(x-z) <  opts.tol*norm(b)
    fprintf('%s: tolerance reached in %i iterations\n',mfilename,k);
else
    fprintf('%s: tolerance not reached in %i iterations\n',mfilename,k);  
end

x = z;


%% Helper function

function z = shrinkage(x, kappa)

z = sign(x) .* max( abs(x) - kappa, 0); % complex ok

