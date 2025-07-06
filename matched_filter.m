function [out coils noise] = matched_filter(in,dim,np)
% Matched filter coil combination (Walsh MRM 2000;43:682)
%
% Inputs
%  in: array of complex images [2D,3D,4D] 
%  dim: the coil dimension (default=last)
%  np: no. pixels in the neighborhood (default=200)
%
% Output:
%  out = combined image [same as input with nc=1] 
%  coils = the optimal filters used
%  noise = noise std estimate (maybe?)
%
% Neighborhood does not extend over slices (thickness >> pixel).
% Extra dimensions after the coil dim (e.g. TE,TI) are included.
%
%% size - 1D, 2D, 3D, extra dimensions
sz = size(in);

if isempty(in) || numel(sz)<3
    error('input must be an array of images');
end
if ~exist('dim','var') || isempty(dim)
    dim = numel(sz); % assume last dimension is coils
elseif ~isscalar(dim) || dim<1 || dim>numel(sz) || mod(dim,1)
    error('dim is out of range');
end

% coil dimension
nc = sz(dim);

% in-plane dimensions
nx = sz(1);
ny = sz(2);

% slice dimension
if dim==3
    nz = 1;
else
    nz = sz(3);
end

% extra dimensions
ne = prod(sz(dim:end)) / nc;

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

%% permute for fast page operations
in = reshape(in,[nx ny nz nc ne]);
order = [4 5 1 2 3]; % [nc ne nx ny nz]
in = permute(in,order); 

%% construct matched filter
coils = zeros(nc,ne,np,nx,ny,nz,'like',in);

for p = 1:np
    shift = [0 0 x(p) y(p)];
    coils(:,:,p,:,:,:) = circshift(in,shift);
end
coils = reshape(coils,nc,ne*np,nx,ny,nz);

% optimal filter per pixel
[coils noise] = pagesvd(coils,'econ','vector');

% keep largest component
coils = coils(:,1,:,:,:,:);

% coil combined image
out = pagemtimes(coils,'ctranspose',in,'none');

% noise std from 2nd component (?)
noise = mean(reshape(noise(2,:,:,:,:),[],1)) / sqrt(np);

%% original shape (collapsed coil dim)
out = ipermute(out,order);
sz(dim) = 1;
out = reshape(out,sz);
