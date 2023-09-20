function im = espirit2(data,varargin)
%im = espirit2(data,index,varargin)
%
% Implementation of ESPIRIT (in 2nd-dimension only).
% Uses pagesvd.cpp mex-file or builtin for R2021b.
%
% Inputs:
% - data is kspace (nx ny nc) with zeros in empty lines
%
% Output:
% - im is the coil-combined image(s) (nx ny ni)
%
% Example:
if nargin==0
    disp('Running example...')
    load brain_alias_8ch.mat
    varargin = {'std',5,'beta',0.1};
    clearvars -except data varargin
end

%% options

opts.width = 5; % kernel width
opts.radial = 0; % use radial kernel
opts.ni = 2; % no. image components
opts.tol = 1e-6; % pcg tolerange
opts.maxit = 1000; % pcg max iterations
opts.std = []; % noise std dev, if available
opts.lambda = 0; % L1 sparsity regularization
opts.beta = 0; % L2 Tikhonov regularization
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
A = zeros(na*nc,nk,'like',data);

for k = 1:nk
    x = kernel.x(k);
    y = kernel.y(k);
    tmp = circshift(data,[x y]); 
    A(:,k) = tmp(acs);
end

% put in matrix form
A = reshape(A,na,nc*nk);
fprintf('ESPIRIT calibration matrix = %ix%i\n',size(A));

% define dataspace vectors
[~,S,V] = svd(A,'econ');
S = diag(S);
nv = nnz(S > noise_floor);
V = reshape(V(:,1:nv),nc,nk,nv); % only keep dataspace
fprintf('ESPIRIT dataspace vectors = %i (out of %i)\n',nv,nc*nk)

plot(S); xlim([0 numel(S)]); title('svals');
line(xlim,[noise_floor noise_floor],'linestyle',':'); drawnow

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

%% switch to GPU (move earlier if pagesvd available on GPU)
if opts.gpu
    try
    C = gpuArray(C);
    data = gpuArray(data);
    nindex = gpuArray(nindex);
    end
end

%% solve for image components

% linear operators (solve A'Ax=A'b)
AA = @(x)myespirit(C,x,mask,opts.beta);
Ab = bsxfun(@times,conj(C),fft2(data));
Ab = reshape(sum(Ab,3),[],1);

% solve by pcg
if opts.lambda
    im = pcgL1(AA,Ab,opts.lambda,opts.maxit);
else
    im = pcg(AA,Ab,opts.tol,opts.maxit);
end

% display
im = reshape(im,nx,ny,opts.ni);
for k = 1:opts.ni
    subplot(1,opts.ni,k); imagesc(abs(im(:,:,k)));
    title(sprintf('ESPIRIT component %i',k));
end

% avoid dumping output to screen
if nargout==0; clear; end


%% ESPIRIT operator
function r = myespirit(C,im,mask,beta)

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
