function [x,flag,relres,iter,resvec] = pcgpc(A,b,tol,maxit,M1,M2,x0)
% Preconditioned conjugate gradient (pcg) with modifications to
% support the use of a penalty on Im(x) aka a phase constraint.
%
% Meant to be used with anonymous functions, A = @(x)myfunc(x),
% where myfunc(x) should return:
% - A*x           : to minimize ||b-Ax||^2
% - A*x+λ*x       : to minimize ||b-Ax||^2 + λ^2||x||^2
% - A*x+λ*i*Im(x) : to minimize ||b-Ax||^2 + λ^2||Im(x)||^2
%
% References:
% - An Introduction to the CG Method Without the Agonizing Pain
%    Jonathan Richard Shewchuk (1994)
% - Partial Fourier Partially Parallel Imaging
%    Mark Bydder and Matthew D. Robson (MRM 2005;53:1393)
%
% Modifications:
% - uses the real part only of the dot products
% - allows multiple RHS vectors (b = [b1 b2 ...])
% - mostly compatible with pcg (except M2 and flag)
%
% Usage: see Matlab's pcg function (help pcg)

% check arguments
if nargin<2; error('Not enough input arguments.'); end
if ~exist('tol') || isempty(tol); tol = 1e-6; end
if ~exist('maxit') || isempty(maxit); maxit = 20; end
if ~exist('M1') || isempty(M1); M1 = @(arg) arg; end
if exist('M2') && ~isempty(M2); error('M2 argument not supported'); end
if ~ismatrix(b); error('b argument must be a column vector or 2d array'); end
validateattributes(tol,{'numeric'},{'scalar','nonnegative','finite'},'','tol');
validateattributes(maxit,{'numeric'},{'scalar','nonnegative','integer'},'','maxit');

% not intended for matrix inputs but they can be supported
if isnumeric(A); A = @(arg) A * arg; end
if isnumeric(M1); M1 = @(arg) M1 \ arg; end

% initialize
t = tic;
iter = 1;
flag = 1;
if ~exist('x0') || isempty(x0)
    r = b;
    x = zeros(size(b),'like',b);
else
    if ~isequal(size(x0),size(b))
        error('x0 must be a column vector of length %i to match the problem size.',numel(b));
    end
    x = x0;
    r = A(x);   
    if ~isequal(size(r),size(b))
        error('A(x) must return a column vector of length %i to match the problem size.',numel(b));
    end  
    r = b - r;
end
d = M1(r);
if ~isequal(size(d),size(b))
    error('M1(x) must return a column vector of length %i to match the problem size.',numel(b));
end
delta0 = vecnorm(b);
delta_new = real(dot(r,d));
resvec(iter,:) = vecnorm(r);
solvec(iter,:) = vecnorm(x);

% min norm solution
xmin = x;
imin = zeros(1,size(b,2));

% main loop
while maxit
    
    iter = iter+1;
	clear q; q = A(d);
	alpha = delta_new./real(dot(d,q));
    
    % unsuccessful termination
    if ~all(isfinite(alpha)); flag = 4; break; end
    
	x = x + alpha.*d;
    
    % recalculate residual occasionally
    if mod(iter,20)==0
        r = b - A(x);
    else
        r = r - alpha.*q;
    end

    % residual and solution vectors
    resvec(iter,:) = vecnorm(r);
    solvec(iter,:) = vecnorm(x);

    % keep best solution
    ok = resvec(iter,:) < min(resvec(1:iter-1,:));
    if any(ok)
        xmin(:,ok) = x(:,ok);
        imin(ok) = iter;
    end
    
    % successful termination
    if all(resvec(iter,:)<tol*delta0); flag = 0; break; end

    % unsuccessful termination
    if iter>maxit; flag = 1; break; end

    clear q; q = M1(r);
	delta_old = delta_new;
    delta_new = real(dot(r,q));
   
    % unsuccessful termination
    if all(delta_new<=0); flag = 4; break; end

    beta = delta_new./delta_old;
    d = q + beta.*d;

end

% min norm solution
ok = imin==iter;
if any(~ok)
    flag = 3;
    x(:,~ok) = xmin(:,~ok);
end

% remeasure final residual
if nargout>2
    resvec(end,:) = vecnorm(b-A(x));
end
relres = resvec(end,:)./delta0;

% only display if flag not supplied
if nargout<2
    for k = 1:size(b,2)
        fprintf('pcg terminated at iteration %i (flag %i): relres = %e.\n',imin(k),flag,relres(k)); 
    end
    toc(t);
end
