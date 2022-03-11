function ksp = grappa3(data,mask,varargin)
%
% Implementation of GRAPPA for 3D images (yz-direction).
%
% Configured for 2x or 2x2 sampling but may be modified
% by setting up the convolution patterns. The following
% patterns are pre-configured:
%
% (1) 2x2 regular: x o x o  opts.p = |1 0 1| and |0 1 0|
%                  o o o o           |0 0 0|     |1 0 1|
%                  x o x o           |1 0 1|     |0 1 0|
%                  o o o o
%
% (2) 2x2 shifted: x o x o  opts.p = |0 1 0| and |1 1 1|
%                  o o o o           |0 0 0|     |0 0 0|
%                  o x o x           |1 0 1|     |1 1 1|
%                  o o o o           |0 0 0|
%                                    |0 1 0|
%
% (3) 2x1 regular: x x x x  opts.p = |1 1 1|
%                  o o o o           |0 0 0|
%                  x x x x           |1 1 1|
%                  o o o o
%
% (4) 1x2 regular: x o x o  opts.p = |1 0 1|
%                  x o x o           |1 0 1|
%                  x o x o           |1 0 1|
%                  x o x o
%
% (5) 2x checkers: x o x o  opts.p = |0 1 0|
%                  o x o x           |1 0 1|
%                  x o x o           |0 1 0|
%                  o x o x
%
% Otherwise patterns may be passed by cell array (see below).
%
% Inputs:
% -data is kspace [nx ny nz nc] with zeros in empty lines
% -mask is binary array [nx ny nz] or [ny nz]
% -varargin: pairs of options/values (e.g. 'pattern',2)
%
% Output:
% -ksp is reconstructed kspace [nx ny nz nc] for each coil
%
%% example dataset

if nargin==0
    disp('Running example...')
    load phantom3D_6coil.mat
    data = fftshift(data); % center kspace
    data(:,end,:,:) = []; % remove weird odd dimension
    mask = zeros(size(data,2),size(data,3));
    mask(1:2:end,1:2:end) = 1; % undersample 2x2
    mask(3:4:end,:) = circshift(mask(3:4:end,:),[0 1]); % pattern 2
    varargin{1} = 'pattern'; varargin{2} = 2;
    varargin{3} = 'cal'; varargin{4} = data(21:110,51:70,41:60,:); % separate calibration
end

%% options

opts.idx = -2:2; % readout convolution pattern
opts.cal = []; % separate calibration, if available
opts.tol = []; % svd tolerance for calibration
opts.pattern = 1; % scalar 1-4 or cell array (see below)
opts.readout = 1; % readout dimension (1, 2 or 3)

% circular convolution fills kspace all the way to the
% edges so kspace doesn't need to be centered. however
% it is slow. need a built-in 'circ' convn option.
opts.conv = 'same'; % 'same' or 'circ'

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

if isa(opts.pattern,'cell')
    opts.p = opts.pattern; % no error checks
    opts.pattern = 0; % user defined pattern
else
    switch opts.pattern
        case 1;
            opts.p{1} = [1 0 1;0 0 0;1 0 1]; % diagonal
            opts.p{2} = [0 1 0;1 0 1;0 1 0]; % crosses
        case 2;
            opts.p{1} = [0 1 0;0 0 0;1 0 1;0 0 0;0 1 0]; % diamond
            opts.p{2} = [1 1 1;0 0 0;1 1 1]; % rectangle
        case 3;
            opts.p{1} = [1 1 1;0 0 0;1 1 1]; % y only
        case 4;
            opts.p{1} = [1 0 1;1 0 1;1 0 1]; % z only
        case 5;
            opts.p{1} = [0 1 0;1 0 1;0 1 0]; % yz checkerboard
        otherwise;
            error('pattern not recognized');
    end
end

% make logical - catch any bad user defined patterns
for k = 1:numel(opts.p)
    if ~isnumeric(opts.p{k}) || ~ismatrix(opts.p{k})
        error('pattern %k must be numeric matrix',k);
    end
    opts.p{k} = logical(opts.p{k});
end

%% initialize

% argument checks
if ndims(data)<3 || ndims(data)>4
    error('Argument ''data'' must be a 4d array.')
end

% switch readout direction
if opts.readout==2
    data = permute(data,[2 1 3 4]);
    if exist('mask','var'); mask = permute(mask,[2 1 3 4]); end
elseif opts.readout==3
    data = permute(data,[3 2 1 4]);
    if exist('mask','var'); mask = permute(mask,[3 2 1 4]); end
elseif opts.readout~=1
    error('readout dimension must be 1, 2 or 3');
end
[nx ny nz nc] = size(data);

% better to use even numbers (for circular convolution)
if mod(nx,2) || mod(ny,2) || mod(nz,2)
    warning('better to use even array sizes ([%i %i %i])',nx,ny,nz);
end
if ~exist('mask','var') || isempty(mask)
    mask = any(data,4); % 3d mask [nx ny nz]
    disp('Argument ''mask'' not supplied - guessing.')
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

% define the convolution function
if isequal(opts.conv,'circ') && exist('cconvn.m','file')
    grappaconv = @(A,B)cconvn(A,B);
elseif isequal(opts.conv,'same')
    grappaconv = @(A,B)convn(A,B,'same'); % no 'circ' option
else
    error('convolution options are: ''circ'' or ''same''');
end

%% detect sampling

% speedup factor
R = numel(mask)/nnz(mask);

% indices of sampled points
kx = find(any(any(mask,2),3));

% basic checks
if max(diff(kx))>1
    error('Sampling must be contiguous in kx-direction.')
end
if R>nc
    warning('Speed up greater than no. coils (%.1f vs %i)',R,nc);
end

% sampling pattern after each pass of reconstruction
for j = 0:numel(opts.p)
    if j==0
        yz = reshape(any(mask),ny,nz); % initial ky-kz sampling
    else
        s{j} = grappaconv(yz,opts.p{j})>=nnz(opts.p{j});
        yz(s{j}) = 1; % we can now consider these lines sampled
    end
    fprintf('Kspace coverage after pass %i: %f\n',j,nnz(yz)/(ny*nz));
end

% display
fprintf('Data size = %s\n',sprintf('%i ',size(data)));
fprintf('Readout points = %i (out of %i)\n',numel(kx),nx);
disp(opts);

subplot(1,3,1); imagesc(yz+reshape(any(mask),ny,nz),[0 2]);
title(sprintf('%s (pattern=%i)',mfilename,opts.pattern));
xlabel('kz'); ylabel('ky'); 

subplot(1,3,2); im = sum(abs(ifft3(data)),4);
[~,slice] = max(sum(reshape(im,[],nz))); % pick a slice with signal
imagesc(squeeze(im(slice,:,:))); title(sprintf('slice %i (R=%.1f)',slice,R));
xlabel('z'); ylabel('y'); drawnow;

% needs to check for adequate kspace coverage
if nnz(yz)/numel(yz) < 0.9
    warning('inadequate coverage - check sampling patterns.')
end

%% see if gpu is possible

try
    gpu = gpuDevice;
    data = gpuArray(data);
    mask = gpuArray(mask);
    fprintf('GPU found = %s (%.1f Gb)\n',gpu.Name,gpu.AvailableMemory/1e9);
catch ME
    data = gather(data);
    mask = gather(mask);
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
    ind = sub2ind(size(opts.p{j}),c{j}(1),c{j}(2));
    
    % find patterns that satisfy acs sampling
    a = opts.p{j}; a(ind) = 1; % center point (i.e. target)
    acs{j} = find(convn(yz,a,'same')>=nnz(a)); % can't be sure ACS is symmetric, don't wrap
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

    % exclude ACS lines from reconstruction
    if isempty(opts.cal); s{j}(acs{j}) = 0; end
    
end

%% GRAPPA calibration: linear equation AX=B
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

        % neighbors in ky and kz as indices
        wrapy = mod(y(k)-dy+size(yz,1)-1,size(yz,1))+1;
        wrapz = mod(z(k)-dz+size(yz,2)-1,size(yz,2))+1;
        idy = sub2ind(size(yz),wrapy,wrapz);
        
        for m = 1:numel(opts.idx)
            A(:,k,m,:,:) = cal(valid-opts.idx(m),idy,:);
        end
        
    end
    B = cal(valid,acs{j},:);

    % reshape into matrix form
    B = reshape(B,[],nc);
    A = reshape(A,size(B,1),[]);

    % linear solution X = pinv(A)*B
    [V S] = svd(A'*A); S = diag(S);
    if isempty(opts.tol)
        tol = max(size(A))*eps(S(1)); % pinv default
    else
        tol = opts.tol;
    end
    invS = sqrt(S)./(S+tol); % tikhonov
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
            tmp = grappaconv(data(:,:,:,k),Y{j}(:,:,:,k,m));
            ksp_coil_m(:,s{j}) = ksp_coil_m(:,s{j})+tmp(:,s{j});
        end

        ksp(:,:,:,m) = ksp_coil_m;
        
    end
    
    data = ksp;
    
end

fprintf('GRAPPA reconstruction: '); toc;

% unswitch readout direction
if opts.readout==2
    ksp = permute(ksp,[2 1 3 4]);
elseif opts.readout==3
    ksp = permute(ksp,[3 2 1 4]);
end

%% display

subplot(1,3,3); im = sum(abs(ifft3(ksp)),4);
imagesc(squeeze(im(slice,:,:)));
title(sprintf('slice %i (R=%.1f)',slice,R));
xlabel('z'); ylabel('y'); drawnow;

if nargout==0; clear; end % avoid dumping to screen

