function ksp = grappa3(data,mask,varargin)
%
% Implementation of GRAPPA for 3D images (yz-direction).
%
% Configured for 2x2 sampling but may be modified for
% others by setting up the convolution patterns (opts.p).
% E.g. two patterns that are pre-configured
%
% (1) 2x2 regular: o x o x  use p = |1 0 1| and |0 1 0|
%                  x x x x          |0 0 0|     |1 0 1|
%                  o x o x          |1 0 1|     |0 1 0|
%
% (2) 2x2 shifted: o x o x  use p = |0 1 0| and |1 1 1|
%                  x x x x          |0 0 0|     |0 0 0|
%                  x o x o          |1 0 1|     |1 1 1|
%                                   |0 0 0|
%                                   |0 1 0|
% Inputs:
% -data is kspace [nx ny nz nc] with zeros in empty lines
% -mask is binary array [nx ny nz] or [ny nz]
% -varargin: pairs of options/values (e.g. 'pattern',2)
%
% Output:
% -ksp is reconstructed kspace [nx ny nz nc] for each coil
%
% Comments:
% - works best with center of kspace at center of the array
%   because we lack a built-in circular convolution. To do:
%   implement cconvn (based on circshift?). Bonus if it can
%   ignore zeros in the convolution kernel (~2x speed up).
%
%% example dataset

if nargin==0
    disp('Running example...')
    load phantom3D_6coil.mat
    data = fftshift(data);
    mask = zeros(121,96); % sampling mask
    mask(1:2:end,1:2:end) = 1; % undersample 2x2
    mask(3:4:end,:) = circshift(mask(3:4:end,:),[0 1]); % pattern 2
    varargin{1} = 'pattern';
    varargin{2} = 2;
    varargin{3} = 'cal';
    varargin{4} = data(:,50:70,40:60,:); % separate calibration
end

%% options

opts.idx = -2:2; % readout convolution pattern
opts.cal = []; % separate calibration, if available
opts.tol = []; % svd tolerance for calibration
opts.pattern = 1; % default to regular 2x2 sampling

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

%% ky-kz convolution patterns

switch opts.pattern
    case 1;
        opts.p{1} = [1 0 1; 0 0 0; 1 0 1]; % diagonal
        opts.p{2} = [0 1 0; 1 0 1; 0 1 0]; % crosses
    case 2;
        opts.p{1} = [0 1 0;0 0 0;1 0 1;0 0 0;0 1 0]; % diamond
        opts.p{2} = [1 1 1; 0 0 0;1 1 1]; % rectangle
    otherwise;
        error('pattern not defined');
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
    if nnz(mask~=0 & mask~=1)
        error('Argument ''mask'' type must be logical.')
    end
    if isequal(size(mask),[ny nz]) || isequal(size(mask),[1 ny nz])
        mask = repmat(reshape(mask,1,ny,nz),nx,1,1);
    elseif ~isequal(size(mask),[nx ny nz])
        error('Argument ''mask'' size incompatible with data size.')
    end
end
mask = reshape(mask>0,nx,ny,nz); % ensure size/class compatibility

% non-sampled points must be zero
data = bsxfun(@times,data,mask);

%% detect sampling

% indices of sampled points
kx = find(any(any(mask,2),3));
ky = find(any(any(mask,1),3));
kz = find(any(any(mask,1),2));

% max speed up in each direction
Rx = max(diff(kx));
Ry = max(diff(ky));
Rz = max(diff(kz));

% basic checks
if Rx>1
    error('Sampling must be contiguous in kx-direction.')
end
if Ry*Rz>nc
    error('Speed up greater than no. coils (%i vs %i)',Ry*Rz,nc);
end

% phase encode sampling pattern after each pass of recontruction
for j = 0:numel(opts.p)
    if j==0
        yz = reshape(any(mask),ny,nz); % initial ky-kz sampling
    else
        s{j} = convn(yz,opts.p{j},'same')>=nnz(opts.p{j});
        yz(s{j}) = 1; % we now consider these lines sampled
    end
    fprintf('Kspace coverage after pass %i: %f\n',j,nnz(yz)/(ny*nz));
end

% display
fprintf('Data size = %s\n',sprintf('%i ',size(data)));
fprintf('Readout points = %i (out of %i)\n',numel(kx),nx);
fprintf('Sampling in ky-kz = %ix%i\n',Ry,Rz);
disp(opts);

subplot(1,2,1); imagesc(yz+reshape(any(mask),ny,nz));
title('sampling'); xlabel('kz'); ylabel('ky'); 

subplot(1,2,2); im = sum(abs(ifft3(data)),4);
slice = ceil(nx/2); imagesc(squeeze(im(slice,:,:)));
title(sprintf('%s (R=2x2)',mfilename));
xlabel('z'); ylabel('y'); drawnow;

% needs to check for adequate kspace coverage
if nnz(yz)/(ny*nz) < 0.9
    error('Inadequate coverage - check sampling patterns.')
end

%% see if gpu is possible

try
    gpu = gpuDevice;
    if verLessThan('matlab','8.4'); error('GPU needs MATLAB R2014b'); end
    data = gpuArray(data);
    mask = gpuArray(mask);
    fprintf('GPU found = %s (%.1f Gb)\n',gpu.Name,gpu.AvailableMemory/1e9);
catch ME
    data = gather(data);
    mask = gather(mask);
    warning('%s. Using CPU.', ME.message);
end

%% GRAPPA acs

if isempty(opts.cal)
    
    % data is self-calibrated
    cal = data;
    
    % phase encode sampling
    yz = reshape(gather(any(mask)),ny,nz);

    % valid points along kx
    valid = kx(1)+max(opts.idx):kx(end)+min(opts.idx);
    
else
    
    % separate calibration data
    cal = cast(opts.cal,'like',data);
    
    if size(cal,4)~=nc || ndims(cal)~=ndims(data)
        error('separate calibration data must have %i coils.',nc);
    end
    
    % phase encode sampling (assume fully sampled)
    yz = true(size(cal,2),size(cal,3));
    
    % valid points along kx (assume fully sampled)
    valid = 1+max(opts.idx):size(cal,1)+min(opts.idx);

end

nv = numel(valid);
if nv<1
    error('Not enough ACS points in kx (%i)',nv);
end

% detect ACS lines for each sampling pattern
for j = 1:numel(opts.p)
    
    % center point of the convolution
    c{j} = ceil(size(opts.p{j})/2);
    
    % acs sampling pattern
    a = opts.p{j}; a(c{j}) = 1; % center point
    acs{j} = find(convn(yz,a,'same')>=nnz(a));
    na(j) =  numel(acs{j});
    
    fprintf('No. ACS lines for pattern %i = %i ',j,na(j));
    if isempty(opts.cal)
        fprintf('(self cal)\n');
    else
        fprintf('(separate cal)\n');
    end
    if na(j)<1
        error('Not enough ACS lines (%i)',na(j));
    end

    % exclude acs lines from reconstruction
    if isempty(opts.cal); s{j}(acs{j}) = 0; end
    
end

%% GRAPPA calibration
tic;

% concatenate ky-kz to use indices (easier!)
cal = reshape(cal,size(cal,1),[],nc);

for j = 1:numel(opts.p)
    
    % convolution matrix (compatible with convn)
    A = zeros(nv,na(j),numel(opts.idx),nnz(opts.p{j}),nc,'like',data);

    % acs points in ky and kz
    [y z] = ind2sub(size(yz),acs{j});
    
    % offsets to neighbors in ky and kz
    [dy dz] = ind2sub(size(opts.p{j}),find(opts.p{j}));

    % center the offsets
    dy = dy-c{j}(1); dz = dz-c{j}(2);
    
    % convolution matrix
    for k = 1:na(j)

        % neighbors in ky and kz as indices (idy)
        idy = sub2ind(size(yz),y(k)-dy,z(k)-dz);
        
        for m = 1:numel(opts.idx)
            A(:,k,m,:,:) = cal(valid-opts.idx(m),idy,:);
        end
        
    end
    B = cal(valid,acs{j},:);

    % reshape into matrix form
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
    invS = S./(S.^2+tol^2); % tikhonov
    X = V*(invS.^2.*(V'*(A'*B)));
    clear A B V % reduce memory for next loop

    % resize for convn and pad with zeros: extra work but easier
    X = reshape(X,numel(opts.idx),[],nc,nc);
    
    Y{j} = zeros(numel(opts.idx),numel(opts.p{j}),nc,nc,'like',X);
    for k = 1:nc
        for m = 1:nc
            Y{j}(:,find(opts.p{j}),k,m) = X(:,:,k,m);
        end
    end
    Y{j} = reshape(Y{j},[numel(opts.idx) size(opts.p{j}) nc nc]);

end
fprintf('SVD tolerance = %.2e (%.2f%%)\n',tol,100*tol/S(1));
fprintf('GRAPPA calibration: '); toc;

%% GRAPPA recon in multiple passes
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

