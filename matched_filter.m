function [out coils noise] = matched_filter(in,dim,np)
%function [out coils noise] = matched_filter(in,dim,np)
%
% Matched filter coil combination (Walsh MRM 2000;43:682)
%
% Inputs
%  in: array [nx nc ...], [nx ny nc ...] or [nx ny nz nc ...] 
%  dim: coil dimension (default=last)
%  np: no. pixels in neighborhood (default=200)
%
% Outputs
%  out: combined image [same as input with nc=1] 
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
elseif ~isscalar(dim) || dim<2 || dim>4 || mod(dim,1)
    error('dim is not valid');
end

% coil dimension
nc = sz(dim);

% spatial dimensions
nx = sz(1);

switch dim
    case 2; ny = 1; nz = 1;
    case 3; ny = sz(2); nz = 1; 
    otherwise; ny = sz(2); nz = sz(3);
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

% indices of an LxL neighborhood
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

% pick closest symmetric kernel to np points
ok = find(diff(r));
[~,j] = min(abs(ok-np/ne));
np = ok(j); k = k(1:np);
x = x(k); y = y(k);

fprintf('%s: [%i %i %i] nc=%i ne=%i np=%i r=%.3f\n',mfilename,nx,ny,nz,nc,ne,np*ne,r(np));

%% construct filter
coils = zeros(nx,ny,nz,nc,ne,np,'like',in);
for p = 1:np
    shift = [x(p) y(p)];
    tmp = circshift(in,shift);
    coils(:,:,:,:,:,p) = reshape(tmp,[nx ny nz nc ne]);
end
coils = reshape(coils,[nx ny nz nc ne*np]);

% permute for fast page operations
order = [4 5 1 2 3];
coils = permute(coils,order);

% optimal filters (per pixel)
[coils noise] = pagesvd(coils,'econ','vector');
coils = coils(:,1,:,:,:,:); % largest component

% undo permute
coils = ipermute(coils,order);
coils = reshape(coils,sz(1:dim));

%% dot-product filter with input 

% dot doesn't do broadcast operations so do it manually
coils = conj(coils);
out = sum(coils.*in,dim);

% std dev estimate (dodgy)
if nargout>2
    noise = mean(reshape(noise(2,:,:,:,:),[],1)) / sqrt(np*ne);
end
