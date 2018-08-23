function ksp = grappa2(data,mask,varargin)
%
% Implementation of GRAPPA for 2D images (y-direction).
%
% Inputs:
% -data is kspace [nx ny nc] with zeros in empty lines
% -mask is binary array [nx ny] or can be linear indices
% -varargin: pairs of options/values (e.g. 'tol',1)
%
% Output:
% -ksp is reconstructed kspace [nx ny nc] for each coil
%
% Notes:
% -automatic detection of speedup factor and acs lines 
% -best with center of kspace at center of the array
% -use uniform outer line spacing and fully sampled acs
%
%% example dataset

if nargin==0
    disp('Running example...')
    load phantom
    data = fftshift(fft2(data));
    mask = union(1:3:256,122:134);
end

%% options

opts.width = 2; % no. of neighbors (ky) to use in kernel
opts.idx = -opts.width:opts.width; % readout kernel (kx)
opts.cal = []; % separate calibration data, if available
opts.tol = []; % svd tolerance for calibration

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
if ndims(data)<2 || ndims(data)>3
    error('Argument ''data'' must be a 3d array.')
end
[nx ny nc] = size(data);

if ~exist('mask','var') || isempty(mask)
    mask = any(data,3); % 2d mask [nx ny]
    warning('Argument ''mask'' not supplied - guessing.')
elseif isvector(mask)
    if ~isa(mask,'logical')
        index = unique(mask); % allow indices
        mask = false(1,ny); mask(index) = 1;
    end
    mask = repmat(reshape(mask,1,ny),nx,1);
end
mask = reshape(mask,nx,ny); % catch size mismatch

% clean up data (nonsampled points must be zero)
data = bsxfun(@times,data,mask);

% indices of sampled phase encodes lines
pe = find(any(mask,1));

% detect speedup factor: equally spaced lines
for R = 1:nc+1
    eq = pe(1):R:pe(end);
    if all(ismember(eq,pe)); break; end
end
if R>nc; error('ky sampling pattern not supported.'); end

% indices of sampled readout points
ro = find(any(mask,2));

% can only handle contiguous readout points
if any(diff(ro)>1)
    error('data must be contiguous in kx-direction.')
end

% display
fprintf('Line spacing R = %i\n',R);
fprintf('Phase encodes = %i (out of %i)\n',numel(pe),ny);
fprintf('Readout points = %i (out of %i)\n',numel(ro),nx);
fprintf('Number of coils = %i\n',nc);

%% GRAPPA kernel and acs

% indices in ky for the kernel
for k = 1:opts.width
    idy(k) = 1+power(-1,k-1)*floor(k/2)*R;
end
idy = sort(idy);
idx = sort(opts.idx);
fprintf('Kernel: idx=[%s\b] and idy=[%s\b]\n',sprintf('%i ',idx),sprintf('%i ',idy))

% handle calibration data
if isempty(opts.cal)
    
    % data is self-calibrated
    cal = data;
    
    % detect ACS lines
    acs = [];
    for j = 1:numel(pe)
        if all(ismember(pe(j)-idy,pe))
            acs = [acs pe(j)];
        end
    end
    
    % valid points along kx-direction
    valid = ro(1)+max(idx):ro(end)+min(idx);

else
    
    % checks on separate calibration data
    cal = cast(opts.cal,'like',data);
    
    if size(cal,3)~=nc
        error('separate calibration data must have %i coils.',nc);
    end
    
    % ACS lines (assume fully sampled)
    acs = 1+max(idy):size(cal,2)+min(idy);
    
    % valid points along kx-direction (assume fully sampled)
    valid = 1+max(idx):size(cal,1)+min(idx);
    
end

na = numel(acs);
nv = numel(valid);

if na==0 || nv==0
    error('ACS lines = none')
else
    fprintf('ACS region = [%i x %i] ',nv,na);
    if isempty(opts.cal)
        fprintf('(self cal)\n');
    else
        fprintf('(separate cal)\n');
    end
end

%% GRAPPA calibration: solve AX=B

% convolution matrix (compatible with convn)
A = zeros(nv,na,numel(idx),numel(idy),nc,'like',data);
for j = 1:na
    for k = 1:numel(idx)
        A(:,j,k,:,:) = cal(valid-idx(k),acs(j)-idy,:);
    end
end
B = cal(valid,acs,:);

% reshape into matrices
A = reshape(A,nv*na,numel(idx)*numel(idy)*nc);
B = reshape(B,nv*na,nc);

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

fprintf('SVD tolerance = %.2e (%.2f%%)\n',tol,100*tol/S(1));

% reshape for convolution
X = reshape(X,numel(idx),numel(idy),nc,nc);

%% GRAPPA reconstruction

% variable to return
ksp = data;

for k = 1:R-1
    neq = mod(eq,ny)+1; % new lines to reconstruct
    ksp(:,neq,:) = 0; % wipe any existing nonzeros
    for m = 1:nc
        for j = 1:nc
            ksp(:,neq,m) = ksp(:,neq,m) + conv2(ksp(:,eq,j),X(:,:,j,m),'same');
        end
    end
    eq = neq; % new lines are now existing lines
end

% reinsert original data
ksp(ro,pe,:) = data(ro,pe,:);

%% handle partial Fourier sampling
% not implemented: options - homodyne or conjugate coils

if pe(1)>R || pe(end)<ny-R
    warning('partial ky data detected (range %i-%i).',pe([1 end]));
end
if ro(1)>R || ro(end)<nx-R
    warning('partial kx data detected (range %i-%i).',ro([1 end]));
end

%% display

if nargout==0
    im = abs(ifft2(fftshift(ksp)));
    imagesc(sum(im,3)); % magnitude image
    title(sprintf('%s (R=%i)',mfilename,R));
    clear
end

