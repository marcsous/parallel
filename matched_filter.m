function [out coils noise] = matched_filter(in,dim,np)
%function [out coils noise] = matched_filter(in,dim,np)
%
% Matched filter coil combination (Walsh MRM 2000;43:682)
%
% Inputs
%  in: array of complex images [2D or 3D] 
%  dim: coil dimension (default=last)
%  np: pixels in neighborhood (default=200)
%
% Outputs
%  out: combined image [same as input with nc=1] 
%  coils: filters s.t. out = sum(coils.*in,dim)
%  noise: noise std estimate (maybe not reliable)
%
% Neighborhood does not include slices (thickness >> pixel).
% Dimensions after the coil dim (e.g. TE,TI) are included.
%
%% size - 1D, 2D, 3D, extra dimensions
sz = size(in);

if isempty(in) || numel(sz)<3
    error('input must be an array of images');
end
if ~exist('dim','var') || isempty(dim)
    dim = numel(sz); % assume last dimension is coils
elseif ~isscalar(dim) || dim<2 || dim>numel(sz) || mod(dim,1)
    error('dim is out of range');
end

% coil dimension
nc = sz(dim);

% spatial dimensions
nx = sz(1);

switch dim
    case 2; nz = 1; ny = 1; 
    case 3; nz = 1; ny = sz(2);
    otherwise; ny = sz(2); nz = sz(3);
end

% extra dimensions
ne = prod(sz(dim:end)) / nc;

% form consistent shape
in = reshape(in,[nx ny nz nc ne]);

%% neighborhood of np nearest pixels (symmetric about center)
if ~exist('np','var') || isempty(np)
    np = 200; % np = 200 is 90% optimal
else
    np = max(nc,np); % lower limit
end 

% catch silliness (prevent crash)
if np > 1000
    error('neighborhood size (np=%i) is too large',np);
end

% get indices of a large circle (radius L)
L = sqrt(np/ne);
[x y] = ndgrid(-ceil(L/2):ceil(L/2));

% sort by radius
r = hypot(x,y);
[r k] = sort(reshape(r,[],1));

% pick nearest symmetric kernel to np points
ok = find(diff(r));
[~,j] = min(abs(ok-np/ne));
np = ok(j); k = k(1:np);
x = x(k); y = y(k);

fprintf('%s: [%i %i %i] nc=%i ne=%i np=%i r=%.3f\n',mfilename,nx,ny,nz,nc,ne,np*ne,r(np));

%% construct filter
coils = zeros(nx,ny,nz,nc,ne,np,'like',in);
for p = 1:np
    shift = [x(p) y(p)];
    coils(:,:,:,:,:,p) = circshift(in,shift);
end
coils = reshape(coils,[nx ny nz nc ne*np]);

% permute for fast page operations
order = [4 5 1 2 3];
in = permute(in,order);
coils = permute(coils,order);

% optimal filters (per pixel)
[coils noise] = pagesvd(coils,'econ','vector');

% keep largest component
coils = coils(:,1,:,:,:,:);

% dot product the filter with the signal
out = pagemtimes(coils,'ctranspose',in,'none');

%% original shape
out = ipermute(out,order);
sz(dim) = 1;
out = reshape(out,sz);

if nargout>1
    sz(dim) = nc;    
    coils = conj(coils);
    coils = ipermute(coils,order);
    coils = reshape(coils,sz(1:dim));
end

if nargout>2
    noise = mean(reshape(noise(2,:,:,:,:),[],1)) / sqrt(np*ne);
end