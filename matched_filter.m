function [im coils noise] = matched_filter(data,np)
% Matched filter coil combination (Walsh MRM 2000;43:682)
%
% Inputs:
%  data = complex images [nx ny nz nc (ne)] 
%  np = no. pixels in the neighborhood (100)
%
% Output:
%  im = the combined image [nx ny nz (ne)] 
%  coils = the optimal coil filters 
%  noise = noise std estimate (maybe?)

%% parse inputs
[nx ny nz nc ne] = size(data);

% try to accomodate 2D, 3D and a time dimension (echo)
if isempty(data)
    error('data cannot be empty');
elseif ndims(data)==3 
    nc = nz; nz = 1;
elseif ndims(data)<4 || ndims(data)>5
    error('data must be [nx ny nz nc ne]');
end

% coil dimension (must be 4)
dim = 4;
data = reshape(data,[nx ny nz nc ne]);

% np=200 is 90% optimal but most benefit comes earlier
if ~exist('np','var')
    np = 100;
else
    np = max(nc,np); % lower limit
end 

% warn about silliness
if np*ne > 1000
    warning('effective neighborhood size (ne*np=%i) is excessively large',np*ne);
end

%% neighborhood of np nearest pixels

% polygon of sides L
L(3) = min(nz,np^(1/3));
L(2) = (np/L(3))^(1/2);
L(1) = (np/L(3))^(1/2);

[x y z] = ndgrid(-ceil(L(1)/2):ceil(L(1)/2), ...
                 -ceil(L(2)/2):ceil(L(2)/2), ...
                  -fix(L(3)/2):fix(L(3)/2));

% sort by radius
r = sqrt(x.^2 + y.^2 + z.^2);
[r k] = sort(reshape(r,[],1));

% round to nearest symmetric kernel
ok = find(diff(r));
[~,j] = min(abs(np-ok));
np = ok(j); k = k(1:np);

% keep nearest np points
x = x(k); y = y(k); z = z(k);

disp([mfilename ': [' num2str(size(data)) '] np=' num2str(np) ' r=' num2str(r(np),'%f')]);

%% convolution matrix
coils = zeros(nx,ny,nz,nc,ne,np,'like',data);

for p = 1:np
    shift = [x(p) y(p) z(p)];
    coils(:,:,:,:,:,p) = circshift(data,shift);
end

% fold ne into np dimension
coils = reshape(coils,[nx ny nz nc ne*np]);

% reorder for pagesvd: np nc nx ny nz
coils = permute(coils,[5 4 1 2 3]);

% optimal filter per pixel
[~,noise,coils] = pagesvd(coils,'econ','vector');

% largest component only
coils = coils(:,1,:,:,:);

% reorder for dot: nx ny nz nc 1
coils = permute(coils,[3 4 5 1 2]);

% noise std estimate?
noise = mean(reshape(noise(2:end,:,:,:,:),[],1)) / sqrt(ne*np);

%% final image
im = sum(coils.*data,dim);

% collape coil dimension
im = reshape(im,[nx ny nz ne]);
