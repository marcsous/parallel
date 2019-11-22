function [image phase] = homodyne(kspace,dim,index,method,window)
%[image phase] = homodyne(kspace,dim,index,method,window)
%
% Partial Fourier reconstruction for 2D or 3D datasets.
% Leave kspace zeroed where unsampled and the code will
% figure out dim and index automatically.
%
% In this code we obey the laws of physics (1 dim only).
%
% Inputs:
% -kspace is partially filled kspace data (2D or 3D)
%
% Optional inputs:
% -dim is the dimension to operate over [auto]
% -index is a vector of sampled points [auto]
% -method ('homodyne','pocs') ['homodyne']
% -window ('step','ramp','quad','cube') ['cube']

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

% fraction of sampling in kx, ky, kz
f = [numel(kx)/nx numel(ky)/ny numel(kz)/nz];

threshold = 0.98;
if all(f>threshold)
    error('kspace is fully sampled - no need for homodyne');
end

if ~exist('dim','var') || isempty(dim)
    [~,dim] = min(f);
    if sum(f<threshold) > 1
        warning('more than 1 partial dimension: [%s]. Using dim %i.',num2str(f,'%.2f '),dim);
    end
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
H = H + flip(1-H);

center = find(H==1); % symmetric center of kspace
center = [center(1)-1;center(:);center(end)+1]; % pad by 1 point
ramp = linspace(H(center(1)),H(center(end)),numel(center)); % ramp

switch window
    case 'step'
        H(center) = 1;
    case {'linear','ramp'}
        H(center) = ramp;
    case {'quadratic','quad'}
        H(center) = (ramp-1).^2.*sign(ramp-1)+1;
    case {'cubic','cube'}
        H(center) = (ramp-1).^3+1;
    case {'quartic'}
        H(center) = (ramp-1).^4.*sign(ramp-1)+1;    
    otherwise
        error('window not recognized');
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


