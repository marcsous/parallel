function ksp = sake3(data,mask,varargin)
% ksp = sake3(data,mask,varargin)
%
% 3D MRI reconstruction based on matrix completion.
% Low memory version does not form matrix but is slow.
%
% Singular value filtering is done based on opts.noise
% which is a key parameter that affects image quality.
%
% Inputs:
%  -data [nx ny nz nc]: 3D kspace data array from nc coils
%  -mask [nx ny nz]: 3D sampling mask (can be 2D [ny nz]) 
%  -varargin: pairs of options/values (e.g. 'radial',1)
%
% Outputs:
%  -ksp [nx ny nz nc]: 3D kspace data array from nc coils
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
    data(:,end,:,:) = []; % remove odd dimension
    mask = false(128,120,96); % sampling mask
    mask(:,1:2:end,1:2:end) = 1; % undersample
    varargin{1} = 'cal'; varargin{2} = data(:,51:70,41:60,:); % separate calibration
    data = bsxfun(@times,data,mask); % clean up data
end

%% setup

% default options
opts.width = 4; % kernel width
opts.radial = 1; % use radial kernel
opts.loraks = 0; % phase constraint (loraks)
opts.tol = 1e-5; % relative tolerance
opts.maxit = 1e2; % maximum no. iterations
opts.noise = []; % noise std, if available
opts.center = []; % center of kspace, if available
opts.cal = []; % separate calibration data, if available

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
if ndims(data)<3 || ndims(data)>4 || ~isfloat(data)
    error('''data'' must be a 4d float array.')
end
[nx ny nz nc] = size(data);

if ~exist('mask','var') || isempty(mask)
    mask = any(data,4); % 3D mask [nx ny nz]
    warning('''mask'' not supplied - guessing.')
else
    if nnz(mask~=0 & mask~=1)
        error('''mask'' must be binary.')
    end
    if isequal(size(mask),[ny nz]) || isequal(size(mask),[1 ny nz])
        mask = repmat(reshape(mask,1,ny,nz),nx,1,1);
    elseif ~isequal(size(mask),[nx ny nz])
        error('''mask'' size not compatible with ''data'' size.')
    end
end
mask = reshape(mask>0,nx,ny,nz); % ensire size/class compatibility

% convolution kernel indicies
[x y z] = ndgrid(-fix(opts.width/2):fix(opts.width/2));
if opts.radial
    k = sqrt(x.^2+y.^2+z.^2)<=opts.width/2;
else
    k = abs(x)<=opts.width/2 & abs(y)<=opts.width/2 & abs(z)<=opts.width/2;
end
nk = nnz(k);
opts.kernel.x = x(k);
opts.kernel.y = y(k);
opts.kernel.z = z(k);
opts.kernel.mask = k;

% dimensions of the data set
opts.dims = [nx ny nz nc nk 1];
if opts.loraks; opts.dims(6) = 2; end

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

% sampling info
matrix_density = nnz(mask) / numel(mask);

% display
disp(rmfield(opts,{'flip','kernel'}));
fprintf('Density = %f\n',matrix_density);

%% see if gpu is possible

try
    gpu = gpuDevice;
    if verLessThan('matlab','8.4'); error('GPU needs MATLAB R2014b.'); end
    data = gpuArray(data);
    mask = gpuArray(mask);
    fprintf('GPU found: %s (%.1f Gb)\n',gpu.Name,gpu.AvailableMemory/1e9);
catch ME
    data = gather(data);
    mask = gather(mask);
    warning('%s Using CPU.', ME.message);
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

ksp = zeros(size(data),'like',data);

for iter = 1:opts.maxit

    % data consistency
    ksp = bsxfun(@times,data,mask)+bsxfun(@times,ksp,~mask);
    
    % normal calibration matrix
    AA = make_data_matrix(ksp,opts);

    % row space and singular values
    if isempty(opts.cal)
        [V W] = svd(AA);
        W = sqrt(diag(W));
    else
        W = svd(AA);
        W = sqrt(W);
    end

    % estimate noise floor of singular values
    if isempty(opts.noise)
        for j = 1:numel(W)
            h = hist(W(j:end),sqrt(numel(W)-j));
            [~,k] = max(h);
            if k>1; break; end
        end
        sigma = median(W(j:end));
        opts.noise = sigma / sqrt(matrix_density*nx*ny);
        fprintf('Noise std estimate: %.2e\n',opts.noise);
    end
    sigma = opts.noise * sqrt(matrix_density*nx*ny);

    % minimum variance filter
    f = max(0,1-sigma.^2./W.^2); 
    F = V * diag(f) * V';
    
    % hankel structure (average along anti-diagonals)  
    ksp = undo_data_matrix(F,ksp,opts);
    
    % check convergence
    normA(iter) = sum(W);
    if iter==1
        tol(iter) = opts.tol;
    else
        tol(iter) = gather(norm(ksp(:)-old(:))/norm(ksp(:)));
    end
    old = ksp;
    converged = tol(iter) < opts.tol;
    
    % display every few iterations
    if mod(iter,2)==1 || converged
        display(W,f,sigma,ksp,iter,tol,opts,normA);
    end

    % finish when nothing left to do
    if converged; break; end
 
end

if nargout==0; clear; end % avoid dumping to screen

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
            chunkZ = conj(chunkZ(opts.flip.x,opts.flip.y,opts.flip.z,:));
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

%% show plots of various things
function display(W,f,sigma,ksp,iter,tol,opts,normA)

% plot singular values
subplot(1,4,1); plot(W/W(1)); title(sprintf('rank %i',nnz(f))); 
hold on; plot(f,'--'); hold off; xlim([0 numel(f)+1]);
line(xlim,gather([1 1]*sigma/W(1)),'linestyle',':','color','black');
legend({'singular vals.','sing. val. filter','noise floor'});

% show current kspace (center of kx)
subplot(1,4,2);
tmp = squeeze(log(sum(abs(ksp(opts.center(1),:,:,:)),4)));
imagesc(tmp); xlabel('kz'); ylabel('ky'); title('kspace');

% show current image (center of x)
subplot(1,4,3); slice = ceil(size(ksp,1)/2);
tmp = ifft(ksp); tmp = squeeze(tmp(slice,:,:,:));
imagesc(sum(abs(ifft2(tmp)),3)); xlabel('z'); ylabel('y');
title(sprintf('iter %i',iter));

% plot change in norm and tol
subplot(1,4,4); h = plotyy(1:iter,tol,1:iter,normA,'semilogy','semilogy');
axis(h,'tight'); xlabel('iters'); xlim(h,[0 iter+1]); title('metrics');
legend({'||Δk||/||k||','||A||_* norm'}); drawnow;
