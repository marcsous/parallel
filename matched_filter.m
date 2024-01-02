function [im coils noise] = matched_filter(data,np)
% Matched filter coil combination.
% Ref: David Walsh MRM 2000;43:682
%
% Inputs:
%  data = complex images [nx ny nz nc (ne)] 
%  np = no. pixels in the neighborhood (200)
%
% Output:
%  im = the combined image [nx ny nz (ne)] 
%  coils = the optimal coil filters 
%  noise = noise std estimate

% run example
if nargin==0
    load head
    [nx ny nc] = size(data); % 2D
    data = reshape(data,nx,ny,1,nc);
end

%% parse inputs
if ndims(data)<4
    error('data size must be [nx ny nz nc]');
end
[nx ny nz nc ne] = size(data);
dim = 4; % coil dimension

% np=200 is 90% optimal but most benefit comes <50
if ~exist('np','var')
    np = 49;
else
    np = max(nc,np); % lower limit
end 

%% neighborhood of np nearest pixels

% polygon of sides L
L(3) = min(nz,np^(1/3));
L(2) = (np/L(3))^(1/2);
L(1) = (np/L(3))^(1/2);

[x y z] = ndgrid(-ceil(L(1)/2):ceil(L(1)/2), ...
                 -ceil(L(2)/2):ceil(L(2)/2), ...
                  -fix(L(3)/2):fix(L(3)/2));

% sort by distance
d = sqrt(x.^2 + y.^2 + z.^2);
[d k] = sort(reshape(d,[],1));

% keep nearest np
k = k(1:np);
d = d(1:np);
x = x(k); y = y(k); z = z(k);

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

% noise std estimate
noise = mean(reshape(noise(2:end,:,:,:,:),[],1)) / sqrt(ne*np);

%% final image
im = sum(coils.*data,dim);

% collape coil dimension
im = reshape(im,[nx ny nz ne]);

