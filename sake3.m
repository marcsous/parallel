function [ksp mask] = sake3(data,mask,varargin)
% ksp = sake3(data,mask,varargin)
%
% 3D MRI reconstruction based on matrix completion.
% Low memory version does not form matrix but is slow.
%
% Inputs:
%  -data [nx ny nz nc]: 3d kspace data array from nc coils
%  -mask [nx ny nz]: 3d sampling mask (or 2d [ny nz]) 
%  -varargin: pairs of options/values (e.g. 'radial',1)
%
% Outputs:
%  -ksp [nx ny nz nc]: 3d kspace data array from nc coils
%
% References:
%  -Haldar JP et al. LORAKS. IEEE Trans Med Imag 2014;33:668
%  -Shin PJ et al. SAKE. Magn Resonance Medicine 2014;72:959
%
%% example dataset

if nargin==0
    disp('Running example...')
    load phantom3D_6coil.mat
    data = fftshift(data);
    [nx ny nz nc] = size(data);
    mask = false(ny,nz); % sampling mask
    mask(1:2:ny,1:2:nz) = 1; % undersampling
    k = -10:10; % fully sample center of kspace
    mask(ceil(ny/2)+k,ceil(nz/2)+k) = 1; % calibration
    clearvars -except data mask varargin
end

%% setup

% default options
opts.width = 3; % kernel width
opts.radial = 1; % use radial kernel
opts.tol = 1e-4; % relative tolerance
opts.loraks = 0; % use phase constraint (loraks)
opts.maxit = 1e4; % maximum no. iterations
opts.noise = []; % noise std, if available
opts.cal = []; % separate calibration data, if available
opts.center = []; % center of kspace, if available

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

%% initialize

% argument checks
if ndims(data)<3 || ndims(data)>4
    error('Argument ''data'' must be a 4d array.')
end
[nx ny nz nc] = size(data);

if ~exist('mask','var') || isempty(mask)
    mask = any(data,4); % 3d mask [nx ny nz]
    warning('Argument ''mask'' not supplied - guessing.')
else
    if ~isa(mask,'logical')
        error('Argument ''mask'' type must be logical.')
    end
    if isequal(size(mask),[nx ny nz]) || isequal(size(mask),[1 ny nz]) || isequal(size(mask),[ny nz])
        mask = reshape(mask,[],ny,nz); % compatible shape for bsxfun
    else
        error('Argument ''mask'' size incompatible with data size.')
    end
end

% convolution kernel indicies
[x y z] = ndgrid(-ceil(opts.width/2):ceil(opts.width/2));
if opts.radial
    k = sqrt(x.^2+y.^2+z.^2)<=opts.width/2;
else
    k = abs(x)<=opts.width/2 & abs(y)<=opts.width/2 & abs(z)<=opts.width/2;
end
nk = nnz(k);
opts.kernel.x = x(k);
opts.kernel.y = y(k);
opts.kernel.z = z(k);

% dimensions of the data set
opts.dims = [nx ny nz nc nk];
if opts.loraks; opts.dims = [opts.dims 2]; end

% estimate center of kspace
if isempty(opts.center)
    [~,k] = max(reshape(data,[],nc));
    [x y z] = ind2sub([nx ny nz],k);
    opts.center(1) = gather(round(median(x)));
    opts.center(2) = gather(round(median(y)));
    opts.center(3) = gather(round(median(z)));
end

% indices for conjugate reflection about center
opts.flip.x = circshift(nx:-1:1,[0 2*opts.center(1)-1]);
opts.flip.y = circshift(ny:-1:1,[0 2*opts.center(2)-1]);
opts.flip.z = circshift(nz:-1:1,[0 2*opts.center(3)-1]);

% memory required for calibration matrix
k = gather(data(1)) * 0; % single or double
bytes = 2 * prod(opts.dims) * getfield(whos('k'),'bytes');

% sampling info
density = nnz(mask) / numel(mask);

% display
disp(rmfield(opts,{'flip','kernel'}));
fprintf('Sampling density = %f\n',density);
fprintf('Matrix = %ix%i (%.1f Gb)\n',nx*ny*nz,prod(opts.dims(4:end)),bytes/1e9);

%% see if gpu is possible

try
    gpu = gpuDevice;
    if verLessThan('matlab','8.4'); error('GPU needs MATLAB R2014b'); end
    data = gpuArray(data);
    mask = gpuArray(mask);
    fprintf('GPU found = %s (%.1f Gb)\n',gpu.Name,gpu.AvailableMemory/1e9);
catch ME
    data = gather(data);
    mask = gather(mask);
    warning('%s. Using CPU.', ME.message);
end

%% separate calibration data

if ~isempty(opts.cal)
    
    if size(opts.cal,4)~=nc
        error('Separate calibration data must have %i coils.',nc);
    end
    if opts.loraks
        error('Separate calibration not compatible with loraks.');
    end
    
    % caluculate row space
    cal = cast(opts.cal,'like',data);
    AA = make_data_matrix(cal,opts);
    [V,~] = svd(AA);

end

%% Cadzow algorithm

ksp = zeros(nx,ny,nz,nc,'like',data);

for iter = 1:opts.maxit
    
    % data consistency
    ksp = bsxfun(@times,data,mask)+bsxfun(@times,ksp,~mask);

    % normal calibration matrix
    AA = make_data_matrix(ksp,opts);

    % row space and singular values
    if isempty(opts.cal)
        [V S] = svd(AA);
        S = sqrt(diag(S));
    else
        S = svd(AA);
        S = sqrt(S);
    end

    % estimate noise from singular values
    if isempty(opts.noise)
        for j = 1:numel(S)
            h = hist(S(j:end),sqrt(numel(S)-j));
            [~,k] = max(h);
            if k>1; break; end
        end
        noise_floor = median(S(j:end));
        
        if noise_floor==0
            error('Noise floor estimation failed.');
        else
            opts.noise = noise_floor / sqrt(2*density*nx*ny*nz);
            disp(['Estimated noise std = ' num2str(opts.noise)]);
        end
    else
        noise_floor = opts.noise * sqrt(2*density*nx*ny*nz);
    end
    
    % minimum variance filter
    f = max(0,1-noise_floor.^2./S.^2);
    F = V * diag(f) * V';
 
    % hankel structure (average along anti-diagonals)  
    ksp = undo_data_matrix(F,ksp,opts);

    % convergence
    normA(iter) = sum(S);
    if iter==1; old = NaN; end
    tol(iter) = norm(ksp(:)-old(:))/norm(ksp(:));
    old = ksp;
    converged = tol(iter) < opts.tol;
    
    % display every few iterations
    if true || converged
        
        % plot singular values
        subplot(1,4,1); plot(S/S(1));
        hold on; plot(max(f,min(ylim)),'--'); hold off
        line(xlim,gather([noise_floor noise_floor]/S(1)),'linestyle',':','color','black');
        legend({'singular vals.','min. var. filter','noise floor'});
        title(sprintf('rank %i',nnz(f>0))); xlim([0 numel(S)+1]);
        
        % show current kspace
        subplot(1,4,2); slice = ceil(nx/2);
        temp = squeeze(log(sum(abs(ksp(slice,:,:,:)),4)));
        imagesc(temp); xlabel('kz'); ylabel('ky'); title('kspace');
        
        % show current image
        subplot(1,4,3);
        temp = ifft(ksp); temp = squeeze(temp(slice,:,:,:));
        imagesc(sum(abs(ifft2(temp)),3)); xlabel('z'); ylabel('y');
        title(sprintf('iter %i',iter));
        
        % plot change in norm
        subplot(1,4,4); warning('off','MATLAB:Axes:NegativeLimitsInLogAxis');
        [h,~,~] = plotyy(1:iter,tol,1:iter,normA);
        axis(h,'tight'); set(h(1),'YScale','log'); set(h(2),'YScale','log');
        title('metrics'); legend({'||Δk||/||k||','||A||_* norm'});
        xlim(h(1),[0 iter+1]); xlim(h(2),[0 iter+1]); xlabel('iters');
        drawnow;
        
    end

    % finish when nothing left to do
    if converged; break; end
 
end

% return on CPU
ksp = gather(ksp);
mask = gather(mask);

%% make normal calibration matrix (low memory)
function AA = make_data_matrix(data,opts)

nx = size(data,1);
ny = size(data,2);
nz = size(data,3);
nc = size(data,4);
nk = opts.dims(5);

AA = zeros(nc,nk,nc,nk,'like',data);
if opts.loraks
    BB = zeros(nc,nk,nc,nk,'like',data);
end

for j = 1:nk

    x = opts.kernel.x(j);
    y = opts.kernel.y(j);
    z = opts.kernel.z(j);
    row = circshift(data,[x y z]); % rows of A.'
    
    for k = j:nk
        
        x = opts.kernel.x(k);
        y = opts.kernel.y(k);
        z = opts.kernel.z(k);
        col = circshift(data,[x y z]); % cols of A

        % matrix multiply A'*A
        AA(:,j,:,k) = reshape(row,[],nc)' * reshape(col,[],nc);
 
        % fill conjugate symmetric entries
        AA(:,k,:,j) = squeeze(AA(:,j,:,k))';

        if opts.loraks
            col = conj(col(opts.flip.x,opts.flip.y,opts.flip.z,:));
            BB(:,j,:,k) = reshape(row,[],nc)' * reshape(col,[],nc);
            BB(:,k,:,j) = squeeze(BB(:,j,:,k)).';
        end
        
    end

end
AA = reshape(AA,nc*nk,nc*nk);

if opts.loraks
    BB = reshape(BB,nc*nk,nc*nk);
    AA = [AA BB;conj(BB) conj(AA)];
end

%% undo calibration matrix (low memory)
function ksp = undo_data_matrix(F,data,opts)

nx = opts.dims(1);
ny = opts.dims(2);
nz = opts.dims(3);
nc = opts.dims(4);
nk = opts.dims(5);

if ~opts.loraks
    F = reshape(F,nc,nk,nc,nk);
else
    F = reshape(F,nc,2*nk,nc,2*nk);
end
ksp = zeros(nx,ny,nz,nc,'like',data);

for j = 1:nk

    x = opts.kernel.x(j);
    y = opts.kernel.y(j);
    z = opts.kernel.z(j);
    colA = circshift(data,[x y z]); % cols of A

    if opts.loraks
        colZ = conj(colA(opts.flip.x,opts.flip.y,opts.flip.z,:)); % conj sym cols of A
    end
    
    for k = 1:nk
        
        chunkA = reshape(colA,[],nc) * squeeze(F(:,j,:,k));
        
        if opts.loraks
            chunkA = chunkA + reshape(colZ,[],nc) * squeeze(F(:,j+nk,:,k));
            chunkZ = reshape(colA,[],nc) * squeeze(F(:,j,:,k+nk));
            chunkZ = chunkZ + reshape(colZ,[],nc) * squeeze(F(:,j+nk,:,k+nk));
            chunkZ = reshape(chunkZ,nx,ny,nz,nc);
            chunkZ = conj(chunkZ(opts.flip.x,opts.flip.y,opts.flip.y,:));
            chunkA = reshape(chunkA,nx,ny,nz,nc) + chunkZ;
        else
            chunkA = reshape(chunkA,nx,ny,nz,nc);
        end

        % reorder and sum along rows      
        x = opts.kernel.x(k);
        y = opts.kernel.y(k);
        z = opts.kernel.z(k);
        ksp = ksp + circshift(chunkA,-[x y z]);
    
    end
    
end

% average
if ~opts.loraks
    ksp = ksp / nk;
else
    ksp = ksp / (2*nk);
end
