function [out coils noise] = matched_filter(in,dim,np,Rn,cflag)
%function [out coils noise] = matched_filter(in,dim,np,Rn,cflag)
%
% Matched filter coil combination (Walsh MRM 2000;43:682)
%
% Inputs
%  in: array [nx nc ...], [nx ny nc ...] or [nx ny nz nc ...] 
%  dim: coil dimension (default=last)
%  np: target no. pixels in neighborhood (default=200)
%  Rn: noise correlation matrix [nc nc nz] (default=identity)
%  cflag: include center point in neighborhood (default=false)
%
% Outputs
%  out: combined image [same size as input with nc=1] 
%  coils: filters s.t. out = sum(coils.*in,dim)
%  noise: noise std estimate (maybe not reliable)
%
% Neighborhood does not include nz (thickness >> pixel).
% Dimensions after coil dim (e.g. TE, TI) are included.
%
%% size - 1D, 2D, 3D, extra dimensions
sz = size(in);

if isempty(in) || numel(sz)<2
    error('input must be an array of images');
end
if ~exist('dim','var') || isempty(dim)
    dim = numel(sz); % assume last dimension is coils
elseif ~isscalar(dim)
    error('dim must be a scalar');
end

% spatial dimensions
nx = sz(1);

switch dim
    case 2; ny = 1; nz = 1;
    case 3; ny = sz(2); nz = 1; 
    case 4; ny = sz(2); nz = sz(3);
    otherwise; error('dim is not valid');
end

% coil dimension
nc = sz(dim);

% extra dimensions
ne = prod(sz(dim:end)) / nc;

% force consistent shape internally
in = reshape(in,[nx ny nz nc ne]);

% check decorrelation matrix 
if ~exist('Rn','var')
    Rn = [];
elseif isequal(size(Rn),[nc nc])
    Rn = repmat(Rn,[1 1 nz]);
elseif ~isequal(size(Rn),[nc nc nz])
    error('Rn is the wrong size');
end

% center point flag
if ~exist('cflag','var') || isempty(cflag)
    cflag = false;
elseif ~islogical(cflag)
    error('cflag must be true or false');
end

%% neighborhood of np nearest pixels (symmetric about center)
if ~exist('np','var') || isempty(np)
    np = 200; % np = 200 is 90% optimal
else
    np = max(nc,np); % lower limit
end 

% catch silliness
if np > 1000
    error('neighborhood size (np=%i) is too large',np);
end

% define an LxL neighborhood
L = np/ne; % probably way too large
[x y] = ndgrid(-ceil(L/2):ceil(L/2));

% stay within bounds
x(abs(x)>=nx) = NaN;
y(abs(y)>=ny) = NaN;
valid = ~isnan(x+y);
x = x(valid);
y = y(valid);

% sort by radius
r = hypot(x,y);
[r k] = sort(reshape(r,[],1));

% exclude r=0 (self-correlation)
if ~cflag
    r = r(2:end);
    k = k(2:end);
end

% pick closest symmetric kernel to np points
ok = find(diff(r));
[~,j] = min(abs(ok-np/ne));
np = ok(j); k = k(1:np);
x = x(k); y = y(k);

%% display
fprintf('%s: [%ix%i',mfilename,nx,ny);
if nz>1; fprintf('x%i',nz); end
fprintf('] nc=%i ne=%i np=%i\n',nc,ne,np*ne);

%% construct filters: fft version

% permute for fast page operations
order = [5 4 3 1 2]; % [ne nc nz nx ny]

% neighborhood mask with periodic boundary (c.f. circshift)
mask = zeros(nx,ny,'like',real(in));
idx = sub2ind([nx ny],mod(x,nx)+1,mod(y,ny)+1);
mask(idx) = nx*ny; mask = ifft2(mask,'symmetric');

% permute for fast page operations
in = permute(in,order);

% coil correlation (Rs' * Rs)
C = pagemtimes(in,'ctranspose',in,'none');

% spatial correlation
C = fft(fft(C,[],4),[],5);
C = C.*reshape(mask,[1 1 1 nx ny]);
C = ifft(ifft(C,[],5),[],4);

% decorrelate coils: C_decorr = iRn' * C * iRn
if ~isempty(Rn)
    iRn = pagepinv(Rn) .* (sqrt(nc) ./ pagenorm(Rn,'fro'));
    C = pagemtimes(iRn,'ctranspose',pagemtimes(C,'none',iRn,'none'),'none');
end

% principal component
[V S] = pagesvd(C,'vector');
V = V(:,1,:,:,:,:);

% undo permute
V = ipermute(V,order);

% build coils
coils = reshape(V,sz(1:dim));

% std dev estimate
tmp = nonzeros(S(2:end,:));
noise = sqrt(mean(tmp) / (np*ne)); % normal eqns

%% dot-product filter with input 
in = ipermute(in,order);
in = reshape(in,sz);
out = sum(coils.*in,dim);

if nargout>2
    noise = mean(noise);
end
