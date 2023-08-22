function im = espirit2(data,index)

% Implementation of ESPIRIT (in 2nd-dimension only).
% Line spacing must be R with acs spacing 1.
%
% Inputs:
% - data is kspace (nx ny nc) with zeros in empty lines
% - index is a vector of the location of acquired lines
%
% Output:
% - im is the coil-combined image(s) (nx ny ni)
%
% Example:
if nargin==0
    disp('Running example...')
    load brain_alias_8ch.mat
    index = find(any(rms(data,3)));
    varargin = {'std',5,'beta',0.1};
    clearvars -except data index varargin
end

%% options

opts.width = 5; % kernel width
opts.radial = 0; % use radial kernel
opts.ni = 2; % no. image components
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

% acquired lines - clean up (no duplicates, sorted ascending)
index = unique(index);
if index(1)<1 || index(end)>ny
    error('index out of bounds');
end

% not index (non acquired lines)
nindex = setdiff(1:ny,index);

% make sure input data array is clean
data(:,nindex,:) = 0;

% estimate noise std (heuristic)
if isempty(opts.std)
    tmp = nonzeros(data); tmp = sort([real(tmp); imag(tmp)]);
    k = ceil(numel(tmp)/10); tmp = tmp(k:end-k+1); % trim 20%
    opts.std = 1.4826 * median(abs(tmp-median(tmp))) * sqrt(2);
end
noise_floor = opts.std * sqrt(nnz(data)/nc);

line([noise_floor noise_floor],ylim,'linestyle',':')
fprintf('ESPIRIT noise std = %.1e\n',opts.std)

%% ESPIRIT setup

% convolution kernel indicies
[x y] = ndgrid(-fix(opts.width/2):fix(opts.width/2));
if opts.radial
    k = hypot(x,y)<=opts.width/2;
else
    k = abs(x)<=opts.width/2 & abs(y)<=opts.width/2;
end
nk = nnz(k);
kernel.x = x(k);
kernel.y = y(k);
idx = unique(kernel.x);
idy = unique(kernel.y);

fprintf('ESPIRIT kernel width = %i\n',opts.width)
fprintf('ESPIRIT radial kernel = %i\n',opts.radial)
fprintf('ESPIRIT kernel points = %i\n',nk)

% ACS indices in x (without wrap)
acs.x = 1:nx;
acs.x(acs.x<1+max(idx) | acs.x>nx+min(idx)) = [];

% ACS indices in y (with wrap)
acs.y = [];
for j = 1:numel(index)
    y = mod(index(j)-idy+ny-1,ny)+1; % wrap with unit offset
    if all(ismember(y,index))
        acs.y = [acs.y index(j)];
    end
end

if numel(acs.y)==0
    error('ESPIRIT ACS lines = none')
else
    disp(['ESPIRIT ACS lines = ' num2str(acs.y)])
end

%% ESPIRIT calibration

% make calibration matrix
A = zeros(numel(acs.x),numel(acs.y),nc,nk,'like',data);

for k = 1:nk
    x = acs.x-kernel.x(k);
    y = acs.y-kernel.y(k);
    y = mod(y+ny-1,ny)+1; % wrap in y
    A(:,:,:,k) = data(x,y,:);
end

% put in matrix form
A = reshape(A,numel(acs.x)*numel(acs.y),nc*nk);
fprintf('ESPIRIT calibration matrix = %ix%i\n',size(A));

% define dataspace vectors
[~,S,V] = svd(A,'econ');
S = diag(S);
nv = nnz(S > noise_floor);
V = reshape(V(:,1:nv),nc,nk,nv); % only keep dataspace
fprintf('ESPIRIT dataspace vectors = %i (out of %i)\n',nv,nc*nk)

subplot(1,2,2)
plot(S); xlim([0 numel(S)]); title('svals');
line(xlim,[noise_floor noise_floor],'linestyle',':');

% dataspace vectors as convolution kernels
G = zeros(nv,nc,numel(idx),numel(idy),'like',data);
for k = 1:nk
    x = kernel.x(k)+max(idx)+1;
    y = kernel.y(k)+max(idy)+1;
    G(:,:,x,y) = permute(V(:,k,:),[3 1 2]);
end

% convolution in kspace <=> multiplication in image
G = fft(fft(G,nx,3),ny,4);

%% ESPIRIT coil sensitivities (image)
C = zeros(nx,ny,nc,opts.ni,'like',data);
 
% matched filter for optimal per pixel passband
if verLessThan('matlab','9.11')
    for x = 1:nx
        for y = 1:ny
            [~,~,V] = svd(G(:,:,x,y),'econ');
            C(x,y,:,:) = V(:,1:opts.ni);
        end
        subplot(1,2,1); imagesc(abs(C(:,:,1)));
        title('ESPIRIT coil 1'); drawnow
    end
else
    [~,~,C] = pagesvd(G,'econ'); % batch svd
    C = permute(C(:,1:opts.ni,:,:),[3 4 1 2]);
end

subplot(1,2,1); imagesc(abs(C(:,:,1)));
title('ESPIRIT coil 1'); drawnow 

%% solve for image components

% use gpu?
if opts.gpu 
    C = gpuArray(C);
    data = gpuArray(data);
end

% linear operators (solve A'Ax=A'b)
AA = @(x)myespirit(C,x,nindex,opts.beta);
Ab = bsxfun(@times,conj(C),fft2(data));
Ab = reshape(sum(Ab,3),[],1);

% solve by pcg
if opts.lambda
    im = pcgL1(AA,Ab,opts.lambda,opts.maxit);
else
    im = pcg(AA,Ab,0,opts.maxit);
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
function r = myespirit(C,im,nindex,beta)

[nx ny nc ni] = size(C);
im = reshape(im,nx,ny,1,ni);

% normal equations
r = bsxfun(@times,C,im);
r = sum(r,4);
r = ifft2(r);
r(:,nindex,:) = 0;
r = fft2(r);
r = bsxfun(@times,conj(C),r);
r = sum(r,3);

% Tikhonov
r = r+beta^2*im;

% vector for solver
r = reshape(r,[],1);
