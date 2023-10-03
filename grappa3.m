function ksp = grappa3(data,varargin)
%ksp = grappa3(data,varargin)
% Implementation of GRAPPA for 3D kspace (yz-direction).
% All accelerations in y and z are supported. Specify 
% the sampling via the option 'pattern' (see below).
%
% Inputs:
% -data: kspace [nx ny nz nc] with zeros in empty lines
% -varargin: pairs of options/values (e.g. 'pattern',2)
%
% Output:
% -ksp is reconstructed kspace [nx ny nz nc] 
%
%% Convolution patterns (1D or 2D)
%
% Patterns tell the code which neighbors to use to construct
% the center point. E.g. [1 0 1] means use 2 neighbors along
% the z-direction to fill in the zero (resulting in [1 1 1]). 
% Patterns are not unique but some work better. 4 neighbors
% may be better than 2 but 8 may be worse. Several patterns
% are configured (see code around line 103).
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
% (6) 3x1  y-only: x x x x  pattern = |1| and |1|
%                  o o o o            |0|     |0|
%                  o o o o            |0|     |1|
%                  x x x x            |1|
%
% (7) 3x  shifted: x o o x  pattern = |0 0 1| and |0 1 0| 
%                  o x o o            |1 0 0|     |1 0 1|
%                  o o x o            |0 1 0|     |0 1 0|
%                  x o o x
%
% (8) 3x2 regular: x o x o  pattern = |1 0 1|, |1| and |1|
%                  o o o o                     |0|     |0|
%                  o o o o                     |0|     |1|
%                  x o x o                     |1|
%
% (9) 3x3 regular: x o o x  pattern = |1 0 0 1|, |1 0 1|, |1| and |1|
%                  o o o o                                |0|     |0|
%                  o o o o                                |0|     |1|
%                  x o o x                                |1|
%
%% example dataset

if nargin==0
    disp('Running example...')
    load phantom3D_6coil.mat % 25Mb file size for github
    mask = false(size(data,1),size(data,2),size(data,3));
    mask(:,1:2:end,1:2:end) = 1; % undersample 2x2
    mask(:,3:4:end,:) = circshift(mask(:,3:4:end,:),[0 0 1]);
    varargin{1} = 'pattern'; varargin{2} = 5; % pattern 5
    %mask(:,size(data,2)/2+(-9:9),size(data,3)/2+(-9:9)) = 1; % self calibration
    varargin{3} = 'cal'; varargin{4} = data(:,size(data,2)/2+(-9:9),size(data,3)/2+(-9:9),:); % separate calibration
    data = bsxfun(@times,data,mask); clearvars -except data varargin
end

%% handle options

opts.idx = -2:2; % readout convolution pattern
opts.cal = []; % separate calibration, if available
opts.tol = []; % svd tolerance for calibration
opts.pattern = []; % scalar 1-6 or cell array (see code)
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
    error('Pattern must be scalar or cell array.');
else
    switch opts.pattern
        case 1 % 2x1; 
            pattern{1} = [1 1 1;0 0 0;1 1 1];
        case 2 % 1x2
            pattern{1} = [1 0 1;1 0 1;1 0 1];
        case 3 % 2x shifted
            pattern{1} = [0 1 0;1 0 1;0 1 0];
        case 4 % 2x2 regular
            pattern{1} = [1 0 1;0 0 0;1 0 1]; 
            pattern{2} = [0 1 0;1 0 1;0 1 0];      
        case 5 % 2x2 shifted
            pattern{1} = [0 1 0;0 0 0;1 0 1;0 0 0;0 1 0]; 
            pattern{2} = [1 1 1;0 0 0;1 1 1];
        case 6 % 3x1 regular
            pattern{1} = [1 1 1;0 0 0;0 0 0;1 1 1];
            pattern{2} = [1 1 1;0 0 0;1 1 1];
        case 7 % 3x shifted
            pattern{1} = [0 0 1;1 0 0;0 1 0];
            pattern{2} = [0 1 0;1 0 1;0 1 0]; 
        case 8 % 3x2 regular
            pattern{1} = [1 0 1];
            pattern{2} = [1;0;0;1];
            pattern{3} = [1;0;1];
        case 9 % 3x3 regular
            pattern{1} = [1 0 0 1];
            pattern{2} = [1 0 1];
            pattern{3} = [1;0;0;1];
            pattern{4} = [1;0;1];  
        otherwise
            error('Pattern not recognized.');
    end
end

% catch any bad user-defined patterns
for j = 1:numel(pattern)
    if ~isnumeric(pattern{j}) || ~ismatrix(pattern{j})
        error('Pattern %k must be numeric matrix.',j);
    end
    pattern{j} = logical(pattern{j});

    % find center point of the convolution
    center{j} = floor(1+size(pattern{j})/2); % ok for odd/even size
    if pattern{j}(center{j}(1),center{j}(2))
        error('Target of pattern %i is not zero: [%s]',j,num2str(pattern{j}));
    end
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
    error('Readout dimension must be 1, 2 or 3');
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

% initial ky-kz sampling
yz = reshape(any(mask),ny,nz);
fprintf('Kspace coverage before recon: %f\n',1/R);

% sampling pattern after each pass of reconstruction
for j = 1:numel(pattern)
    s{j} = cconvn(yz,pattern{j})==nnz(pattern{j});
    yz(s{j}) = 1; % we now consider these lines sampled
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
        error('Separate calibration data must have %i coils.',nc);
    end
    
    % phase encode sampling (assume fully sampled)
    yz = true(size(cal,2),size(cal,3));
    
    % valid points along kx (assume fully sampled)
    valid = 1+max(opts.idx):size(cal,1)+min(opts.idx);

end

nv = numel(valid);
if nv<1
    error('Not enough ACS points in kx (%i).',nv);
end

% acs for each pattern (don't assume cal is symmetric => convn)
for j = 1:numel(pattern)
    
    % matching lines in cal
    a = pattern{j};
    a(center{j}) = 1;
    acs{j} = find(convn(yz,a,'same')==nnz(a));
    na(j) = numel(acs{j});
    
    fprintf('No. ACS lines for pattern %i = %i ',j,na(j));
    if isempty(opts.cal)
        fprintf('(self cal)\n');
    else
        fprintf('(separate cal)\n');
    end
    if na(j)<1
        error('Not enough ACS lines (%i).',na(j));
    end

    % exclude acs lines from being reconstructed
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
    dy = dy-center{j}(1);
    dz = dz-center{j}(2);
    
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
    
    % convert X from indices to convolution kernels
    Y = zeros(numel(opts.idx),numel(pattern{j}),nc,nc,'like',X);
    for k = 1:nc
        Y(:,find(pattern{j}),:,k) = reshape(X(:,k),numel(opts.idx),[],nc);
    end
    kernel{j} = reshape(Y,[numel(opts.idx) size(pattern{j}) nc nc]);
    
    clear A B V invS X Y % reduce memory for next loop
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
            tmp = cconvn(data(:,:,:,k),kernel{j}(:,:,:,k,m));
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

