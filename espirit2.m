function im = espirit2(data,varargin)
%im = espirit2(data,varargin)
%
% Implementation of ESPIRIT 2D.
%
% Inputs:
% - data is kspace [nx ny nc] with zeros in empty points
% - varargin options pairs (e.g. 'width',4)
%
% Output:
% - im is the coil-combined image(s) [nx ny ni]
%
% Example:
if nargin==0
    disp('Running example...')
    load head
    data=fftshift(ifft2(data));
    mask = false(1,256);
    mask(1:3:end) = 1; mask(125:132) = 1;
    data = bsxfun(@times,data,mask);
    varargin = {'std',2.5e-5,'beta',0.01,'sparsity',0.25};
    clearvars -except data varargin
end

%% options

opts.width = 5; % kernel width
opts.radial = 1; % use radial kernel
opts.ni = 2; % no. image components
opts.tol = 1e-6; % pcg tolerance
opts.maxit = 1000; % pcg max iterations
opts.std = []; % noise std dev, if available
opts.sparsity = 0; % L1 sparsity (0.2=20% zeros)
opts.beta = 0; % L2 Tikhonov regulariztaion
opts.gpu = 1; % use gpu, if available

% varargin handling (must be option/value pairs)
for k = 1:2:numel(varargin)
    if k==numel(varargin) || ~ischar(varargin{k})
        if isempty(varargin{k}); continue; end
        error('''varargin'' must be option/value pairs.');
    end
    if ~isfield(opts,varargin{k})
        error('''%s'' is not a valid option.',varargin{k});
    end
    opts.(varargin{k}) = varargin{k+1};
end

%% initialize
[nx ny nc] = size(data);

% sampling mask [nx ny]
mask = any(data,3); 

% estimate noise std (heuristic)
if isempty(opts.std)
    tmp = data(repmat(mask,[1 1 nc]));
    tmp = sort([real(tmp); imag(tmp)]);
    k = ceil(numel(tmp)/10); tmp = tmp(k:end-k+1); % trim 20%
    opts.std = 1.4826 * median(abs(tmp-median(tmp))) * sqrt(2);
end
noise_floor = opts.std * sqrt(nnz(data)/nc);

fprintf('ESPIRIT noise std = %.1e\n',opts.std)

%% ESPIRIT setup

% convolution kernel indicies
[x y] = ndgrid(-fix(opts.width/2):fix(opts.width/2));
if opts.radial
    k = sqrt(x.^2+y.^2)<=opts.width/2;
else
    k = abs(x)<=opts.width/2 & abs(y)<=opts.width/2;
end
nk = nnz(k);
kernel.x = x(k);
kernel.y = y(k);
kernel.mask = k;

fprintf('ESPIRIT kernel width = %i\n',opts.width)
fprintf('ESPIRIT radial kernel = %i\n',opts.radial)
fprintf('ESPIRIT kernel points = %i\n',nk)

R = numel(mask)/nnz(mask);
fprintf('ESPIRIT acceleration = %.2f\n',R)

%% detect autocalibration samples

% points that satisfy acs (cconvn => wrap)
acs = cconvn(mask,kernel.mask)==nk;
na = nnz(acs);

% expand coils (save looping)
acs = repmat(acs,[1 1 nc]);

fprintf('ESPIRIT ACS lines = %i\n',round(na/nx));

%% calibration matrix
C = zeros(na*nc,nk,'like',data);

for k = 1:nk
    x = kernel.x(k);
    y = kernel.y(k);
    tmp = circshift(data,[x y]); 
    C(:,k) = tmp(acs);
end

% put in matrix form
C = reshape(C,na,nc*nk);
fprintf('ESPIRIT calibration matrix = %ix%i\n',size(C));

% define dataspace vectors
[~,S,V] = svd(C,'econ');
S = diag(S);
nv = nnz(S > noise_floor);
V = reshape(V(:,1:nv),nc,nk,nv); % only keep dataspace
fprintf('ESPIRIT dataspace vectors = %i (out of %i)\n',nv,nc*nk)

plot(S); xlim([0 numel(S)]); title('svals');
line(xlim,[noise_floor noise_floor],'linestyle',':'); drawnow
if nv==0; error('No dataspace vectors - check noise std.'); end

% dataspace vectors as convolution kernels
C = zeros(nv,nc,nx,ny,'like',data);
for k = 1:nk
    x = mod(kernel.x(k)+nx-1,nx)+1; % wrap in x
    y = mod(kernel.y(k)+ny-1,ny)+1; % wrap in y
    C(:,:,x,y) = permute(V(:,k,:),[3 1 2]);
end

%% ESPIRIT coil sensitivities

% fft convolution <=> image multiplication
C = fft(fft(C,nx,3),ny,4); % nv nc nx ny

% optimal passband per pixel
[~,~,C] = pagesvd(C,'econ');

% discard small features
C = C(:,1:opts.ni,:,:);

% reorder: nx ny nc ni
C = permute(C,[3 4 1 2]);

% normalize to first coil phase
C = bsxfun(@times,C,exp(-i*angle(C(:,:,1,:))));

%% switch to GPU (move up if pagesvd available on GPU)

if opts.gpu
    C = gpuArray(C);
    mask = gpuArray(mask);
    data = gpuArray(data);
end

%% solve for image components

% min (1/2)||A'Ax-A'b||_2 + lambda||Qx||_1
AA = @(x)myfunc(x,C,mask,opts.beta);
Ab = bsxfun(@times,conj(C),fft2(data));
Ab = reshape(sum(Ab,3),[],1);
try
    Q = DWT([nx ny],'db1'); % DWT wavelet transform
catch
    Q = 1;
    warning('DWT wavelet transform failed => using sparsity in image.');
end

% solve by pcg/minres
if opts.sparsity
    [im lambda] = pcgL1(AA,Ab,opts.sparsity,opts.tol,opts.maxit,Q);
    z = abs(Q * im); sparsity = nnz(z <= 2*eps(max(z))) / numel(z);
    fprintf('ESPIRIT sparsity %f (lambda=%.2e)\n',sparsity,lambda);
else
    [im,~,~,~,resvec] = minres(AA,Ab,opts.tol,opts.maxit);
end
im = reshape(im,nx,ny,opts.ni);

% display
for k = 1:opts.ni
    subplot(1,opts.ni,k); ims(abs(im(:,:,k)));
    title(sprintf('ESPIRIT component %i',k));
end

% avoid dumping output to screen
if nargout==0; clear; end


%% ESPIRIT operator
function r = myfunc(im,C,mask,beta)

[nx ny nc ni] = size(C);
im = reshape(im,nx,ny,1,ni);

% normal equations
r = bsxfun(@times,C,im);
r = sum(r,4);
r = ifft2(r);
r = bsxfun(@times,r,mask);
r = fft2(r);
r = bsxfun(@times,conj(C),r);
r = sum(r,3);

% Tikhonov
r = r+beta^2*im;

% vector for solver
r = reshape(r,[],1);


% lsqr version - too clunky with rho / lsqrL1.m
%
% tricky: if rho is non-empty it appends onto A
% as [A;rhoI]. caller must also append a vector
% onto b as [b;rho*z]
%A = @(x,flag,rho)myfun2(x,flag,rho,C,mask,opts.beta);
%b = [reshape(data,[],1);zeros(nx*ny*opts.ni,1,'like',data)];
%b = b * sqrt(nx*ny);
% 
%[im,~,~,~,resvec] = lsqr(@(x,flag)A(x,flag,[]),b,opts.tol,opts.maxit);
%[im lambda resvec] = lsqrL1(A,b,opts.sparsity,opts.tol,opts.maxit);
% 
% function r = myfunc(im,flag,rho,C,mask,beta)
% 
% [nx ny nc ni] = size(C);
% 
% % flag = 'transp'(lsqr) or 2(lsmr) 
% if isequal(flag,'transp') || isequal(flag,2) 
%     
%     im = reshape(im,nx,ny,[],1);
% 
%     y = beta * im(:,:,nc+1:nc+2);   
%     if ~isempty(rho)
%         z = rho * im(:,:,nc+3:nc+4);
%     end
% 
%     r = im(:,:,1:nc);
%     r = bsxfun(@times,r,mask);
%     r = fft2(r) / sqrt(nx*ny);
%     r = bsxfun(@times,conj(C),r);
%     r = sum(r,3);
%     r = r+reshape(y,nx,ny,1,2);
% 
%     if ~isempty(rho)
%         r = r+reshape(z,nx,ny,1,2);
%     end
% 
% else
%     
%     im = reshape(im,nx,ny,1,ni);
%     r = bsxfun(@times,C,im);
%     r = ifft2(r) * sqrt(nx*ny);
%     r = bsxfun(@times,r,mask);
%     r = sum(r,4);
%     r = cat(3,r,beta*reshape(im,nx,ny,ni));
% 
%     if ~isempty(rho)
%         r = cat(3,r,rho*reshape(im,nx,ny,ni));
%     end   
%     
% end
% 
% % vector for solver
% r = reshape(r,[],1);