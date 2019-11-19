function [image phase] = homodyne(kspace,dim,index,method,window)
%[image phase] = homodyne(kspace,dim,index,method,window)
%
% Partial Fourier reconstruction for 2D or 3D datasets.
% Leave kspace zeroed and the code will figure out dim
% and index automatically.
%
% In this code we obey the laws of physics (1 dim only).
%
% Inputs:
% -kspace is partially filled kspace data (2D or 3D)
%
% Optional inputs:
% -dim is the dimension to operate over [auto]
% -index is a vector of sampled points [auto]
% -method ('homodyne' or 'pocs') ['homodyne']
% -window filter ('ramp' or 'step') ['ramp']

[nx ny nz ne] = size(kspace);

if ne~=1 || nx==1 || ny==1
    error('only 2D or 3D kspace allowed');
end

% detect dim and index
mask = (kspace~=0);
kx = find(any(any(mask,2),3));
ky = find(any(any(mask,1),3));
kz = find(any(any(mask,1),2));

if any(diff(kx)~=1) || any(diff(ky)~=1) || any(diff(kz)~=1)
    error('kspace not centered or not contiguous');
end

if ~exist('dim','var') || isempty(dim)
    allowance = 1; % allow for the occasional true zero
    if numel(kx)<nx-allowance; dim(1) = 1; end
    if numel(ky)<ny-allowance; dim(2) = 1; end
    if numel(kz)<nz-allowance; dim(3) = 1; end
    if sum(dim)~=1; error('more than 1 partial dimension'); end
    dim = find(dim);
end

if ~exist('index','var') || isempty(index)
    if dim==1; index = kx; end
    if dim==2; index = ky; end        
    if dim==3; index = kz; end
end

% default choices
if ~exist('method','var') || isempty(method)
    method = 'homodyne';
end
if ~exist('window','var') || isempty(window)
    window = 'ramp';
end

% set up low/high pass filters
if dim==1; H = zeros(nx,1,1); end
if dim==2; H = zeros(1,ny,1); end
if dim==3; H = zeros(1,1,nz); end
H(index) = 1;

% high pass filter
H = H + flip(1-H); % step

if isequal(window,'ramp')
    tmp = find(H==1);
    tmp = [tmp(1)-1;tmp(:);tmp(end)+1];
    H(tmp) = linspace(H(tmp(1)),H(tmp(end)),numel(tmp));
end

% low pass filter
L = sqrt(max(0,1-(H-1).^2));

% low resolution phase
phase = bsxfun(@times,L,kspace);
phase = angle(ifftn(phase));

% reconstruction
switch(method)
    
    case 'homodyne';
        image = bsxfun(@times,H,kspace);
        image = ifftn(image).*exp(-i*phase);
        image = abs(real(image));
        
    case 'pocs';
        tmp = kspace;
        
        for iter = 1:10
            
            % abs and low res phase
            image = abs(ifftn(tmp));
            tmp = image.*exp(i*phase);
            
            % data consistency
            tmp = fftn(tmp);
            tmp(mask) = kspace(mask);
            
        end
        
    otherwise;
        error('unknown method ''%s''',method);

end


