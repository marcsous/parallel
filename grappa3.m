function ksp = grappa3(data,varargin)
%
% Implementation of GRAPPA for 3D images (yz-direction).
%
% Inputs:
% -data: kspace [nx ny nz nc] with zeros in empty lines
% -varargin: pairs of options/values (e.g. 'pattern',2)
%
% Output:
% -ksp is reconstructed kspace [nx ny nz nc] 
%
% Patterns tell the code which neighbors to use to construct
% the center point. Patterns are not unique - they are all
% just vectors in the SAKE nullspace - but some work better.
% E.g. 4 neighbors is better than 2 but more calibration is
% required. Several patterns are preconfigured (line 83).
%
% (1) 2x1  y-only: x x x x  pattern = |1| or |1 1 1|
%                  o o o o            |0|    |0 0 0|
%                  x x x x            |1|    |1 1 1|
%                  o o o o
%
% (2) 1x2  z-only: x o x o  pattern = |1 0 1| or |1 0 1|
%                  x o x o                       |1 0 1|
%                  x o x o                       |1 0 1|
%                  x o x o
%
% (3) 2x  shifted: x o x o  pattern = |1| or |1 0 1| or |0 1 0|
%                  o x o x            |0|               |1 0 1|
%                  x o x o            |1|               |0 1 0|
%                  o x o x
%
% (4) 2x2 regular: x o x o  pattern = |1 0 1| and |1|
%                  o o o o                        [0|
%                  x o x o                        |1|
%                  o o o o
%
% (5) 2x2 shifted: x o x o  pattern = |1 0 1| and |1|
%                  o o o o                        |0|
%                  o x o x                        |1|
%                  o o o o
%
% (6) 3x2 regular: x 0 x 0  pattern = |1 0 1|, |1| and |1|
%                  0 0 0 0                     |0|     |0|
%                  0 0 0 0                     |0|     |1|
%                  x 0 x 0                     |1|
%
%% example dataset

if nargin==0
    disp('Running example...')
    load phantom3D_6coil.mat % 25Mb file size for github
    mask = false(size(data,1),size(data,2),size(data,3));
    mask(:,1:2:end,1:2:end) = 1; % undersample 2x2
    mask(:,3:4:end,:) = circshift(mask(:,3:4:end,:),[0 0 1]); % pattern 5
    varargin{1} = 'pattern'; varargin{2} = 5; % specify pattern 5
    %mask(:,size(data,2)/2+(-9:9),size(data,3)/2+(-9:9)) = 1; % self calibration
    varargin{3} = 'cal'; varargin{4} = data(:,size(data,2)/2+(-9:9),size(data,3)/2+(-9:9),:); % separate calibration
    data = bsxfun(@times,data,mask); clearvars -except data varargin
end

%% handle options

opts.idx = -2:2; % readout convolution pattern
opts.cal = []; % separate calibration, if available
opts.tol = []; % svd tolerance for calibration
opts.pattern = 1; % scalar 1-6 or cell array (see above)
opts.readout = 1; % readout dimension (1, 2 or 3)
opts.gpu = 1; % use GPU (sometimes faster without)

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

%% ky-kz convolution patterns

if isa(opts.pattern,'cell')
    pattern = opts.pattern; % no error checks
    opts.pattern = 0; % user defined pattern
elseif ~isscalar(opts.pattern)
    error('pattern must be scalar or cell array');
else
    switch opts.pattern
        case 1 % 2x1; 
            pattern{1} = [1 1 1;0 0 0;1 1 1]; 
        case 2 % 1x2
            pattern{1} = [1 0 1;1 0 1;1 0 1]; 
        case 3 % 2x shifted
            pattern{1} = [0 1 0;1 0 1;0 1 0]; 
        case 4 % 2x2 regular
            pattern{1} = [1;0;1];
            pattern{2} = [1 0 1];         
        case 5 % 2x2 shifted
            pattern{1} = [1 0 1];
            pattern{2} = [1;0;1]; 
        case 6 % 3x2 regular
            pattern{1} = [1 0 1];           
            pattern{2} = [1;0;0;1];
            pattern{3} = [1;0;1];        
        otherwise
            error('pattern not recognized');
    end
end

% make logical - catch any bad user-defined patterns
for j = 1:numel(pattern)
    if ~isnumeric(pattern{j}) || ~ismatrix(pattern{j})
        error('pattern %k must be numeric matrix',j);
    end
    pattern{j} = logical(pattern{j});
end

%% initialize

% argument checks
if ndims(data)<3 || ndims(data)>4 || ~isfloat(data) || isreal(data)
    error('Argument ''data'' must be a 4d complex float array.')
end

% switch readout direction
if opts.readout==2
    data = permute(data,[2 1 3 4]);
    if isfield(opts,'cal'); opts.cal = permute(opts.cal,[2 1 3 4]); end
elseif opts.readout==3
    data = permute(data,[3 2 1 4]);
    if isfield(opts,'cal'); opts.cal = permute(opts.cal,[3 2 1 4]); end
elseif opts.readout~=1
    error('readout dimension must be 1, 2 or 3');
end
[nx ny nz nc] = size(data);

% sampling mask [nx ny nz]
mask = any(data,4); 

%% basic checks

% overall speedup factor
R = numel(mask)/nnz(mask);
if R>nc
    warning('Speed up greater than no. coils (%.2f vs %i)',R,nc);
end

% indices of sampled points
kx = find(any(any(mask,2),3));
if max(diff(kx))>1
    warning('Sampling must be contiguous in kx-direction.')
end

% center point (target) of the convolution
for j = 1:numel(pattern)
    c{j} = floor(1+size(pattern{j})/2); % ok for odd & even sizes
    if pattern{j}(c{j}(1),c{j}(2))
        error('Target of pattern %i is not zero: [%s]',j,num2str(pattern{j},'%i '));
    end
end

% initial ky-kz sampling
yz = reshape(any(mask),ny,nz);
fprintf('Kspace coverage before recon: %f\n',1/R);

% sampling pattern after each pass of reconstruction
for j = 1:numel(pattern)
    s{j} = cconvn(yz,pattern{j})==nnz(pattern{j});
    yz(s{j}) = 1; % we can now consider these lines sampled
    fprintf('Kspace coverage after pass %i: %f\n',j,nnz(yz)/(ny*nz));
end

%% display

fprintf('Data size = %s\n',sprintf('%i ',size(data)));
fprintf('Readout points = %i (out of %i)\n',numel(kx),nx);
disp(opts);

subplot(1,3,1); imagesc(yz+reshape(any(mask),ny,nz),[0 2]);
title(sprintf('sampling (pattern=%i)',opts.pattern));
xlabel('kz'); ylabel('ky'); 

subplot(1,3,2); im = sum(abs(ifft3(data)),4);
slice = floor(nx/2+1); % the middle slice in x
imagesc(squeeze(im(slice,:,:))); title(sprintf('slice %i (R=%.1f)',slice,R));
xlabel('z'); ylabel('y'); drawnow;

% check for adequate kspace coverage post-reconstruction
if nnz(yz)/numel(yz) < 0.95
    warning('inadequate coverage - check patterns (could be partial Fourier?).')
end

%% see if gpu is possible

if opts.gpu
    data = gpuArray(data);
    mask = gpuArray(mask);
    gpu = gpuDevice;
    fprintf('GPU found = %s (%.1f Gb)\n',gpu.Name,gpu.AvailableMemory/1e9);
end

%% detect acs

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

% ACS lines for each sampling pattern
for j = 1:numel(pattern)
    
    % center point expressed as an index
    ind = sub2ind(size(pattern{j}),c{j}(1),c{j}(2));
    
   % find patterns that satisfy acs sampling
    a = pattern{j}; a(ind) = 1;
    acs{j} = find(convn(yz,a,'same')==nnz(a)); % can't be sure ACS is symmetric, so can't use cconvn
    na(j) = numel(acs{j});
    
    fprintf('No. ACS lines for pattern %i = %i ',j,na(j));
    if isempty(opts.cal)
        fprintf('(self cal)\n');
    else
        fprintf('(separate cal)\n');
    end
    if na(j)<1
        error('Not enough ACS lines (%i)',na(j));
    end

    % exclude ACS lines from being reconstructed
    if isempty(opts.cal); s{j}(acs{j}) = 0; end
    
end

%% calibration: linear equation AX=B
t = tic();

% concatenate ky-kz to use indices (easier!)
cal = reshape(cal,size(cal,1),[],nc);

for j = 1:numel(pattern)
    
    % convolution matrix (compatible with convn)
    A = zeros(nv,na(j),numel(opts.idx),nnz(pattern{j}),nc,'like',data);

    % acs points in ky and kz
    [y z] = ind2sub(size(yz),acs{j});
    
    % offsets to neighbors in ky and kz
    [dy dz] = ind2sub(size(pattern{j}),find(pattern{j}));

    % center the offsets
    dy = dy-c{j}(1);
    dz = dz-c{j}(2);
    
    % convolution matrix
    for k = 1:nnz(pattern{j})
        
        % neighbors in ky and kz as indices
        idyz = sub2ind(size(yz),y-dy(k),z-dz(k));
        
        for m = 1:numel(opts.idx)
            A(:,:,m,k,:) = cal(valid-opts.idx(m),idyz,:);
        end
        
    end
    B = cal(valid,acs{j},:);

    % reshape into matrix form
    B = reshape(B,[],nc);
    A = reshape(A,size(B,1),[]);

    % linear solution X = pinv(A)*B
    [V S] = svd(A'*A); S = diag(S);
    if isempty(opts.tol)
        tol = eps(S(1));
    else
        tol = opts.tol;
    end
    invS = sqrt(S)./(S+tol); % tikhonov
    invS(~isfinite(invS.^2)) = 0;
    X = V*(invS.^2.*(V'*(A'*B)));
    
    % resize for convn and pad with zeros: extra work but easier
    X = reshape(X,numel(opts.idx),[],nc,nc);
    
    Y{j} = zeros(numel(opts.idx),numel(pattern{j}),nc,nc,'like',X);
    for k = 1:nc
        for m = 1:nc
            Y{j}(:,find(pattern{j}),k,m) = X(:,:,k,m);
        end
    end
    Y{j} = reshape(Y{j},[numel(opts.idx) size(pattern{j}) nc nc]);

    clear A B V X invS % reduce memory for next loop
end
fprintf('SVD tolerance = %.1e (%.1e%%)\n',tol,100*tol/S(1));
fprintf('GRAPPA calibration: '); toc(t);

%% reconstruction in multiple passes
t = tic();

for j = 1:numel(pattern)

    ksp = data;
    
    for m = 1:nc

        ksp_coil_m = ksp(:,:,:,m);
        
        for k = 1:nc
            tmp = cconvn(data(:,:,:,k),Y{j}(:,:,:,k,m));
            ksp_coil_m(:,s{j}) = ksp_coil_m(:,s{j})+tmp(:,s{j});
        end

        ksp(:,:,:,m) = ksp_coil_m;
        
    end
    
    data = ksp;
    
end

fprintf('GRAPPA reconstruction: '); toc(t);

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

