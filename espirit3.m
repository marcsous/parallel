function im = espirit3(data,varargin)
%im = espirit3(data,varargin)
%
% Implementation of ESPIRIT (in 3D).
%
% Inputs:
% - data is kspace (nx ny nz nc) with zeros in empty points
%
% Output:
% - im is the coil-combined image(s) (nx ny nz ni)
%
% Example:
if nargin==0
    disp('Running example...')
    load phantom3D_6coil.mat
    mask = false(size(data,1),size(data,2),size(data,3));
    mask(:,1:2:end,1:2:end) = 1; % undersample 2x2
    mask(:,3:4:end,:) = circshift(mask(:,3:4:end,:),[0 0 1]); % shifted
    mask(size(data,1)/2+(-9:9),size(data,2)/2+(-9:9),size(data,3)/2+(-9:9)) = 1; % self calibration
    varargin = {'std',1e-5,'beta',0.1}; % set some options
    data = bsxfun(@times,data,mask); clearvars -except data varargin
end

%% options

opts.width = 3; % kernel width (scalar)
opts.radial = 0; % use radial kernel
opts.ni = 1; % no. image components
opts.tol = 1e-6; % pcg tolerange
opts.maxit = 1000; % max pcg iterations
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
[nx ny nz nc] = size(data);

% circular wrap requires even dimensions
if any(mod([nx ny nz],2))
    error('Code requires even numbered dimensions');
end
fprintf('ESPIRIT dataset = [%i %i %i %i]\n',nx,ny,nz,nc)

% sampling mask [nx ny nz]
mask = any(data,4); 

% estimate noise std (heuristic)
if isempty(opts.std)
    tmp = data(repmat(mask,[1 1 1 nc]));
    tmp = sort([real(tmp); imag(tmp)]);
    k = ceil(numel(tmp)/10); tmp = tmp(k:end-k+1); % trim 20%
    opts.std = 1.4826 * median(abs(tmp-median(tmp))) * sqrt(2);
end
noise_floor = opts.std * sqrt(nnz(data)/nc);

fprintf('ESPIRIT noise std = %.1e\n',opts.std)

%% ESPIRIT setup

% convolution kernel indicies
[x y z] = ndgrid(-fix(opts.width/2):fix(opts.width/2));
if opts.radial
    k = sqrt(x.^2+y.^2+z.^2)<=opts.width/2;
else
    k = abs(x)<=opts.width/2 & abs(y)<=opts.width/2 & abs(z)<=opts.width/2;
end
nk = nnz(k);
kernel.x = x(k);
kernel.y = y(k);
kernel.z = z(k);
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
acs = repmat(acs,[1 1 1 nc]);

fprintf('ESPIRIT ACS lines = %i\n',round(na/nx));

%% calibration matrix
A = zeros(na*nc,nk,'like',data);

for k = 1:nk
    x = kernel.x(k);
    y = kernel.y(k);
    z = kernel.z(k);
    tmp = circshift(data,[x y z]); 
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
C = zeros(nv,nc,nx,ny,nz,'like',data);
for k = 1:nk
    x = mod(kernel.x(k)+nx-1,nx)+1; % wrap in x
    y = mod(kernel.y(k)+ny-1,ny)+1; % wrap in y
    z = mod(kernel.z(k)+nz-1,nz)+1; % wrap in z    
    C(:,:,x,y,z) = permute(V(:,k,:),[3 1 2]);
end

%% ESPIRIT coil sensitivities

% fft convolution <=> image multiplication
C = fft(fft(fft(C,nx,3),ny,4),nz,5); % nv nc nx ny nz

% optimal passband per pixel
[~,~,C] = pagesvd(C,'econ');

% discard small features
C = C(:,1:opts.ni,:,:,:);

% reorder: nx ny nz nc ni
C = permute(C,[3 4 5 1 2]);

%% switch to GPU (move earlier if pagesvd available on GPU)
if opts.gpu
    C = gpuArray(C);
    mask = gpuArray(mask);
    data = gpuArray(data);
end

%% solve for image components

% linear operators (solve A'Ax=A'b)
AA = @(x)myespirit(C,x,mask,opts.beta);
Ab = bsxfun(@times,conj(C),fft3(data));
Ab = reshape(sum(Ab,4),[],1);

% solve by pcg/minres
if opts.lambda
    im = pcgL1(AA,Ab,opts.lambda,opts.maxit);
else
    im = minres(AA,Ab,opts.tol,opts.maxit);
end

% display
im = reshape(im,nx,ny,nz,opts.ni);
slice = floor(nx/2+1); % middle slice in x
for k = 1:opts.ni
    subplot(1,opts.ni,k);
    imagesc(squeeze(abs(im(slice,:,:,k))));
    title(sprintf('ESPIRIT component %i',k));
    xlabel('z'); ylabel('y'); drawnow;
end

% avoid dumping output to screen
if nargout==0; clear; end

%% ESPIRIT operator
function r = myespirit(C,im,mask,beta)

[nx ny nz nc ni] = size(C);
im = reshape(im,nx,ny,nz,1,ni);

% normal equations
r = bsxfun(@times,C,im);
r = sum(r,5);
r = ifft3(r);
r = bsxfun(@times,r,mask);
r = fft3(r);
r = bsxfun(@times,conj(C),r);
r = sum(r,4);

% Tikhonov
r = r+beta^2*im;

% vector for solver
r = reshape(r,[],1);
