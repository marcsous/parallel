function [image phase] = homodyne(kspace,varargin)
%[image phase] = homodyne(kspace,varargin)
%
% Partial Fourier reconstruction for 2D or 3D datasets.
% Leave k-space zeroed where unsampled so the code can
% figure out the sampling automatically.
%
% In this code we obey the laws of physics (1 dim only).
%
% Inputs:
% -kspace is partially filled kspace (2D or 3D) single coil
% -varargin: pairs of options/values (e.g. 'radial',1)
%
% Options:
% -opts.method ('homodyne','pocs','least-squares','compressed-sensing')
% -opts.window ('step','ramp','quad','cube','quartic')

%% options

opts.method = 'homodyne'; % 'homodyne','pocs','least-squares','compressed-sensing'
opts.window = 'cubic'; % 'step','ramp','quad','cubic','quartic'
opts.removeOS = 0; % remove 2x oversampling in specified dimension (0=off)

% regularization terms (only apply to least squares/compressed sensing)
opts.damp = 1e-4; % L2 penalty on solution norm
opts.lambda = 1e-2; % L2 penalty on imag norm
opts.cs = 5e-4; % L1 penalty on tranform norm

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

%% handle looping over multi-coil / echo

[nx ny nz nc] = size(kspace);

if nx==1 || ny==1
    error('only 2D or 3D kspace allowed');
end

if nx==0 || ny==0 || nz==0 || nc==0
   error('empty kspace not allowed'); 
end

if nc>1

    for c = 1:nc
        [image(:,:,:,c) phase(:,:,:,c)] = homodyne(kspace(:,:,:,c),varargin{:});
    end
    
else
    
    % detect sampling
    mask = (kspace~=0);
    kx = find(any(any(mask,2),3));
    ky = find(any(any(mask,1),3));
    kz = find(any(any(mask,1),2));
    
    if any(diff(kx)~=1) || any(diff(ky)~=1) || any(diff(kz)~=1)
        error('kspace not centered or not contiguous');
    end
    
    % fraction of sampling in kx, ky, kz
    f = [numel(kx)/nx numel(ky)/ny];
    if nz>1; f(3) = numel(kz)/nz; end
    
    % some checks
    [~,dim] = min(f);
    fprintf('partial sampling: [%s]. Using dimension %i.\n',num2str(f,'%.2f '),dim);
    
    if min(f<0.5)
        error('kspace is too undersampled - must be at least 0.5');
    end
    
    if all(f>0.95)
        warning('kspace is fully sampled - skipping homodyne');
        opts.method = 'none'; % fully sampled - bypass recon
    end
    
    %% set up filters
    
    if ~isequal(opts.method,'none')
        
        if dim==1; H = zeros(nx,1,1); index = kx; end
        if dim==2; H = zeros(1,ny,1); index = ky; end
        if dim==3; H = zeros(1,1,nz); index = kz; end
        H(index) = 1;
        
        % high pass filter
        H = H + flip(1-H);
        
        % symmetric center of kspace
        center = find(H==1);
        center(end+1) = numel(H)/2+1; % make sure
        center = unique(center);
        center = [center(1)-1;center(:);center(end)+1]; % pad by 1 point
        ramp = linspace(H(center(1)),H(center(end)),numel(center)); % symmetric points sum to 2
        
        switch opts.window
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
                error('opts.window not recognized');
        end
        
        % low pass filter
        L = sqrt(max(0,1-(H-1).^2));
        
        % low resolution phase
        phase = bsxfun(@times,L,kspace);
        if false
            % smoothing in the other in-plane dimension (no clear benefit)
            if dim~=1; phase = bsxfun(@times,phase,sin(linspace(0,pi,nx)')); end
            if dim~=2; phase = bsxfun(@times,phase,sin(linspace(0,pi,ny) )); end
        end
        phase = angle(ifftn(ifftshift(phase)));
    end
    
    %% reconstruction
    
    maxit = 10; % no. of iterations to use for iterative opts.methods
    
    switch(opts.method)
        
        case 'homodyne'
            
            image = bsxfun(@times,H,kspace);
            image = ifftn(ifftshift(image)).*exp(-i*phase);
            image = abs(real(image));
            
        case 'pocs'
            
            tmp = kspace;
            
            for iter = 1:maxit
                
                % abs and low res phase
                image = abs(ifftn(tmp));
                tmp = image.*exp(i*phase);
                
                % data consistency
                tmp = fftshift(fftn(tmp));
                tmp(mask) = kspace(mask);
                
            end
            
        case 'least-squares'

            % L2 penalized least squares requires pcgpc.m
            b = reshape(exp(-i*phase).*ifftn(ifftshift(kspace)),[],1);
            tmp = pcgpc(@(x)pcpop(x,mask,phase,opts.lambda,opts.damp),b,[],maxit);
            image = abs(real(reshape(tmp,size(phase))));

        case 'compressed-sensing'
            
            % L1 penalized least squares requires pcgpc.m
            Q = DWT([nx ny nz],'db2'); % wavelet transform
            b = reshape(Q*(exp(-i*phase).*ifftn(ifftshift(kspace))),[],1);
            tmp = pcgL1(@(x)pcpop(x,mask,phase,opts.lambda,opts.damp,Q),b,opts.cs);
            image = abs(real(reshape(Q'*tmp,size(phase))));
            
        case 'none'
            
            tmp = ifftn(kspace);
            image = abs(tmp);
            phase = angle(tmp);
            
        otherwise
            
            error('unknown opts.method ''%s''',opts.method);
            
    end
    
    if opts.removeOS

        image = fftshift(image);
        phase = fftshift(phase);
        
        switch opts.removeOS
            case 1; ok = nx/4 + (1:nx/2);
                image = image(ok,:,:,:);
                phase = phase(ok,:,:,:);
            case 2; ok = ny/4 + (1:ny/2);
                image = image(:,ok,:,:);
                phase = phase(:,ok,:,:);
            case 3; ok = nz/4 + (1:nz/2);
                image = image(:,:,ok,:);
                phase = phase(:,:,ok,:);
        end
        
    end
    
end

%% phase constrained projection operator (image <- image)
function y = pcpop(x,mask,phase,lambda,damp,Q)
% y = P' * F' * W * F * P * x + i * imag(x) + damp * x
x = reshape(x,size(phase));
if exist('Q','var'); x = Q'*x; end
y = exp(i*phase).*x;
y = fftn(y);
y = fftshift(mask).*y;
y = ifftn(y);
y = exp(-i*phase).*y;
y = y + lambda*i*imag(x) + damp*x;
if exist('Q','var'); y = Q*y; end
y = reshape(y,[],1);
