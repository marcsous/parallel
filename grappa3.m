function ksp = grappa3(data,mask,varargin)
%
% Implementation of GRAPPA for 3D images (yz-direction).
%
% Only does regular 2x2 sampling pattern. To use with
% other sampling patterns, modify convolution patterns.
%
% Inputs:
% -data is kspace [nx ny nz nc] with zeros in empty lines
% -mask is binary array [nx ny nz] or [ny nz]
% -varargin: pairs of options/values (e.g. 'tol',1)
%
% Output:
% -ksp is reconstructed kspace [nx ny nz nc] for each coil
%
% Comments:
% -automatic detection of speedup factor and acs lines 
% -best with center of kspace at center of the array
% -use uniform outer line spacing and fully sampled acs
%
%% example dataset

if nargin==0
    disp('Running example...')
    load phantom3D_6coil.mat
    data = fftshift(data);
    [nx ny nz nc] = size(data);
    mask = false(ny,nz); % sampling mask
    mask(1:2:ny,1:2:nz) = 1; % undersampling
    k = -10:10; % fully sample center of kspace
    mask(ceil(ny/2)+k,ceil(nz/2)+k) = 1; % calibration
    clearvars -except data mask varargin
end

%% options

opts.idx = -2:2; % readout convolution pattern
opts.cal = []; % separate calibration data, if available
opts.tol = []; % svd tolerance for calibration

% convolution patterns
opts.p{1} = [1 0 1; 0 0 0; 1 0 1]; % diagonal
opts.p{2} = [0 1 0; 1 0 1; 0 1 0]; % crosses

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

%% initialize

% argument checks
if ndims(data)<3 || ndims(data)>4
    error('Argument ''data'' must be a 4d array.')
end
[nx ny nz nc] = size(data);

if ~exist('mask','var') || isempty(mask)
    mask = any(data,4); % 3d mask [nx ny nz]
    warning('Argument ''mask'' not supplied - guessing.')
else
    if ~isa(mask,'logical')
        error('Argument ''mask'' type must be logical.')
    end
    if isequal(size(mask),[ny nz]) || isequal(size(mask),[1 ny nz])
        mask = repmat(reshape(mask,1,ny,nz),nx,1,1);
    elseif ~isequal(size(mask),[nx ny nz])
        error('Argument ''mask'' size incompatible with data size.')
    end
end

% make sure data is clean
data = mask.*data;

%% detect sampling

% indices of sampled points
kx = find(any(any(mask,2),3));
ky = find(any(any(mask,1),3));
kz = find(any(any(mask,1),2));

% max speed up in each direction
Rx = max(diff(kx));
Ry = max(diff(ky));
Rz = max(diff(kz));

fprintf('Data size = %s\n',sprintf('%i ',size(data)));
fprintf('Readout points = %i (out of %i)\n',numel(kx),nx);
fprintf('Sampling in ky-kz = %ix%i\n',Ry,Rz);
fprintf('Overall speedup factor %f\n',numel(mask)/nnz(mask));
disp(opts);

% basic checks
if Rx>1
    error('sampling must be contiguous in kx-direction.')
end
if Ry>2
    error('sampling must be 2-fold in ky-direction.')
end
if Rz>2
    error('sampling must be 2-fold in kz-direction.')
end
if Ry==1 || Rz==1
    error('Speed up in 1D only (Ry=%i Rz=%i): no 2nd pass needed\n',Ry,Rz);
end
if Ry*Rz>nc
    error('Speed up factor greater than no. coils (%i vs %i)\n',Ry*Rz,nc);
end

% sampling pattern after each pass of the recontruction
for j = 0:numel(opts.p)
    if j==0
        yz = reshape(any(mask),ny,nz);
    else
        s{j} = conv2(yz,opts.p{j},'same')>=nnz(opts.p{j});
        yz(s{j}) = 1;
    end
    fprintf('Kspace coverage after pass %i: %f\n',j,nnz(yz)/numel(yz));
end

% display
subplot(1,2,1); imagesc(yz+reshape(any(mask),ny,nz));
title('sampling'); xlabel('kz'); ylabel('ky'); 

subplot(1,2,2); im = sum(abs(ifft3(data)),4);
slice = ceil(nx/2); imagesc(squeeze(im(slice,:,:)));
title(sprintf('%s (R=%ix%i)',mfilename,Ry,Rz));
xlabel('z'); ylabel('y'); drawnow;

if nnz(yz)/numel(yz) < 0.9
    error('inadequate coverage - check sampling patterns.')
end

%% GRAPPA acs

if isempty(opts.cal)
    
    % data is self-calibrated
    cal = data;
    
    % phase encode sampling    
    yz = reshape(any(mask),ny,nz);
    
    % valid points along kx
    valid = kx(1)+max(opts.idx):kx(end)+min(opts.idx);
    
else
    
    % separate calibration data
    cal = cast(opts.cal,'like',data);
    if size(cal,4)~=nc
        error('separate calibration data must have %i coils.',nc);
    end
    
    % phase encode sampling (assume fully sampled)
    yz = true(size(cal,2),size(cal,3));
    
    % valid points along kx (assume fully sampled)
    valid = 1+max(opts.idx):size(cal,1)+min(opts.idx);

end

% detect ACS lines for each sampling pattern
for j = 1:numel(opts.p)
    
    % center point of the convolution
    c{j} = ceil(size(opts.p{j})/2);
    
    % acs sampling pattern
    a = opts.p{j}; a(c{j}) = 1;
    acs{j} = find(conv2(yz,a,'same')>=nnz(a));

    % exclude acs lines from reconstruction
    s{j}(acs{j}) = 0;
    
    fprintf('No. ACS lines for pattern %i = %i\n',j,numel(acs{j}));
end

% reshape to use indices (faster)
cal = reshape(cal,size(cal,1),[],nc);

%% see if gpu is possible

try
    gpu = gpuDevice;
    if verLessThan('matlab','8.4'); error('GPU needs MATLAB R2014b'); end
    cal = gpuArray(cal);   
    data = gpuArray(data);
    mask = gpuArray(mask);
    fprintf('GPU found = %s (%.1f Gb)\n',gpu.Name,gpu.AvailableMemory/1e9);
catch ME
    cal = gather(cal);
    data = gather(data);
    mask = gather(mask);
    warning('%s. Using CPU.', ME.message);
end

%% GRAPPA calibration

tic
for j = 1:numel(opts.p)
    
    % convolution matrix (compatible with convn)
    A = zeros(numel(valid),numel(acs{j}),numel(opts.idx),nnz(opts.p{j}),nc,'like',data);

    for k = 1:numel(acs{j})
        
        % current acs point in ky and kz
        [y z] = ind2sub([ny nz],acs{j}(k));
        
        % offsets to neighbors in ky and kz
        [dy dz] = ind2sub(size(opts.p{j}),find(opts.p{j}));
        
        % neighbors as indices
        index = sub2ind([ny nz],y-dy+c{j}(1),z-dz+c{j}(2));

        for m = 1:numel(opts.idx)
            A(:,k,m,:,:) = cal(valid-opts.idx(m),index,:);
        end
        
    end
    B = cal(valid,acs{j},:);
    
    % reshape into matrices and solve AX=B
    B = reshape(B,[],nc);
    A = reshape(A,size(B,1),[]);

    % linear solution X = pinv(A)*B
    [V S] = svd(A'*A);
    S = sqrt(diag(S));
    if isempty(opts.tol)
        tol = max(size(A))*eps(S(1)); % pinv default
    else
        tol = opts.tol;
    end
    S = S./(S.^2+tol^2); % tikhonov
    S = diag(S.^2);
    X = V*(S*(V'*(A'*B)));
    
    % resize for convolution and pad with zeros: extra work but easier
    X = reshape(X,numel(opts.idx),[],nc,nc);
    
    Y{j} = zeros(numel(opts.idx),numel(opts.p{j}),nc,nc,'like',X);
    for k = 1:nc
        for m = 1:nc
            Y{j}(:,find(opts.p{j}),k,m) = X(:,:,k,m);
        end
    end
    Y{j} = reshape(Y{j},[numel(opts.idx) size(opts.p{j}) nc nc]);
    clear A B X V S % reduce memory usage for gpu

end
fprintf('SVD tolerance = %.2e\n',tol);
fprintf('GRAPPA calibration: '); toc;

%% GRAPPA recon in passes
tic;

for j = 1:numel(opts.p)

    ksp = data;
    
    for m = 1:nc

        ksp_coil_m = ksp(:,:,:,m);
        
        for k = 1:nc
            tmp = convn(data(:,:,:,k),Y{j}(:,:,:,k,m),'same');
            ksp_coil_m(:,s{j}) = ksp_coil_m(:,s{j})+tmp(:,s{j});
        end
        
        ksp(:,:,:,m) = ksp_coil_m;
        
    end
    
    data = ksp;
    
end

fprintf('GRAPPA reconstruction: '); toc;

%% display

subplot(1,2,2); im = sum(abs(ifft3(data)),4);
imagesc(squeeze(im(slice,:,:)));
title(sprintf('%s (R=%ix%i)',mfilename,Ry,Rz));
xlabel('z'); ylabel('y'); drawnow;

if nargout==0; clear; end % avoid dumping to screen

