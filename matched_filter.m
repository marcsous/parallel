function [im coils] = matched_filter(data,np)

% Matched filter coil combination (Walsh MRM 2000;43:682-690)
%
% Inputs:
%  data are the complex images [nx ny (nz) ncoils] 
%  np is no. pixels in the neighborhood (default 200)
%
% Output:
%  im is the [nx ny nz ni) combined image
%  coils are the optimal SNR coil filters 

% run example
if nargin==0
    load head
end

%% parse inputs
if ndims(data)<=2
    done = data; return;
elseif ndims(data)==3
    [nx ny nc] = size(data);
    nz = 1;    
elseif ndims(data)==4
    [nx ny nz nc] = size(data);
end

% 200 pixels is 90% optimal (Walsh)
if ~exist('np','var'); np = 200; end 

%% neighborhood of nearest np pixels

% polygon of sides L
L(3) = min(nz,np^(1/3));
L(2) = (np/L(3))^(1/2);
L(1) = (np/L(3))^(1/2);

[x y z] = ndgrid(-fix(L(1)/2):fix(L(1)/2), ...
                 -fix(L(2)/2):fix(L(2)/2), ...
                 -fix(L(3)/2):fix(L(3)/2));

[~,k] = sort(x.^2+y.^2+z.^2);
k = k(1:np); % the nearest np
x = x(k); y = y(k); z = z(k);

%% coil calibration matrix
coils = zeros(nx,ny,nz,nc,np,'like',data);

for p = 1:np
    shift = [x(p) y(p) z(p)];
    coils(:,:,:,:,p) = circshift(data,shift);
end

% reorder for pagesvd: ni nc nx ny nz
coils = permute(coils,[5 4 1 2 3]);

% optimal passband per pixel
[~,~,coils] = pagesvd(C,'econ');

% largest component only
coils = coils(:,1,:,:,:);

% reorder for dot: nx ny nz nc ni
coils = permute(coils,[3 4 5 1 2]);

%% final image
im = sum(coils.*reshape(data,size(coils)),4);
