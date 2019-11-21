function ksp = sake2(data,mask,varargin)
% ksp = sake2(data,mask,varargin)
%
% 2D MRI reconstruction based on matrix completion.
%
% Singular value filtering is done based on opts.noise
% which is a key parameter that affects image quality.
%
% Inputs:
%  -data [nx ny nc]: 2D kspace data array from nc coils
%  -mask [nx ny]: 2D sampling mask (can be 1D vector) 
%  -varargin: pairs of options/values (e.g. 'radial',1)
%
% Outputs:
%  -ksp [nx ny nc]: 2D kspace data array from nc coils
%
% References:
%  -Haldar JP et al. LORAKS. IEEE Trans Med Imag 2014;33:668
%  -Shin PJ et al. SAKE. Magn Resonance Medicine 2014;72:959
%
%% example dataset

if nargin==0
    disp('Running example...')
    load phantom
    data = fftshift(fft2(data));
    mask = false(256,256);
    mask(:,1:3:end) = 1; % undersample
    mask(:,120:136) = 1; % self-calibration
    cal = [];% data(64:192,120:136,:); % calibration
    data = bsxfun(@times,data,mask); % clean data
    varargin = {'cal',cal}; % separate cal
    clearvars -except data mask varargin
end

%% setup

% default options
opts.width = 6; % kernel width
opts.radial = 1; % use radial kernel
opts.loraks = 1; % phase constraint (loraks)
opts.tol = 1e-4; % relative tolerance
opts.maxit = 1e4; % maximum no. iterations
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
if ndims(data)<2 || ndims(data)>3 || ~isfloat(data)
    error('''data'' must be a 3d float array.')
end
[nx ny nc] = size(data);

if ~exist('mask','var') || isempty(mask)
    mask = any(data,3); % 2D mask [nx ny]
    warning('''mask'' not supplied - guessing.')
elseif isvector(mask)
    if any(mask~=0 & mask~=1)
        if any(mask<1 | mask>ny | mod(mask,1))
            error('''mask'' is incompatible.');
        end
        % convert linear indices to 1D binary mask
        index = mask; mask = false(1,ny); mask(index) = 1;
    end
    mask = repmat(reshape(mask,1,ny),nx,1);
end
mask = reshape(mask>0,nx,ny); % size/class compatibility

% convolution kernel indicies
[x y] = ndgrid(-fix(opts.width/2):fix(opts.width/2));
if opts.radial
    k = hypot(x,y)<=opts.width/2;
else
    k = abs(x)<=opts.width/2 & abs(y)<=opts.width/2;
end
nk = nnz(k);
opts.kernel.x = x(k);
opts.kernel.y = y(k);
opts.kernel.mask = k;

% dimensions of the dataset
opts.dims = [nx ny nc nk 1];
if opts.loraks; opts.dims(5) = 2; end

% estimate center of kspace
if isempty(opts.center)
    [~,k] = max(reshape(data,[],nc));
    [x y] = ind2sub([nx ny],k);
    opts.center(1) = gather(round(median(x)));
    opts.center(2) = gather(round(median(y)));
end

% indices for conjugate reflection about center
opts.flip.x = circshift(nx:-1:1,[0 2*opts.center(1)-1]);
opts.flip.y = circshift(ny:-1:1,[0 2*opts.center(2)-1]);

% density of data matrix
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
    
    if size(opts.cal,3)~=nc
        error('Separate calibration data must have %i coils.',nc);
    end
    if opts.loraks
        error('Separate calibration not compatible with loraks.');
    end
    
    % caluculate row space
    cal = cast(opts.cal,'like',data);
    A = make_data_matrix(cal,opts);
    [V,~] = svd(A'*A);
    
end

%% Cadzow algorithm

ksp = zeros(size(data),'like',data);

for iter = 1:opts.maxit

    % data consistency
    ksp = bsxfun(@times,data,mask)+bsxfun(@times,ksp,~mask);
    
    % calibration matrix
    A = make_data_matrix(ksp,opts);

    % row space and singular values
    if isempty(opts.cal)
        [V W] = svd(A'*A);
        W = sqrt(diag(W));
    else
        W = svd(A'*A);
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
    A = A * (V * diag(f) * V');
    
    % hankel structure (average along anti-diagonals)   
    ksp = undo_data_matrix(A,opts);
    
    % check convergence
    normA(iter) = sum(W);
    if iter==1
        tol(iter) = opts.tol;
    else
        tol(iter) = gather(norm(ksp(:)-old(:))/norm(ksp(:)));
    end
    old = ksp;
    converged = tol(iter) < opts.tol;
 
    % display progress every few iterations
    if mod(iter,10)==1 || converged
        display(W,f,sigma,ksp,iter,tol,normA);
    end

    % finish when nothing left to do
    if converged; break; end

end

if nargout==0; clear; end % avoid dumping to screen

%% make data matrix
function A = make_data_matrix(data,opts)

nx = size(data,1);
ny = size(data,2);
nc = size(data,3);
nk = opts.dims(4);

A = zeros(nx,ny,nc,nk,'like',data);

for k = 1:nk
    x = opts.kernel.x(k);
    y = opts.kernel.y(k);
    A(:,:,:,k) = circshift(data,[x y]);
end

if opts.loraks
    A = cat(5,A,conj(A(opts.flip.x,opts.flip.y,:,:)));
end

A = reshape(A,nx*ny,[]);

%% undo data matrix
function data = undo_data_matrix(A,opts)

nx = opts.dims(1);
ny = opts.dims(2);
nc = opts.dims(3);
nk = opts.dims(4);

A = reshape(A,nx,ny,nc,nk,[]);

if opts.loraks
    A(opts.flip.x,opts.flip.y,:,:,2) = conj(A(:,:,:,:,2));
end

for k = 1:nk
    x = opts.kernel.x(k);
    y = opts.kernel.y(k);
    A(:,:,:,k,:) = circshift(A(:,:,:,k,:),-[x y]);
end

data = mean(reshape(A,nx,ny,nc,[]),4);

%% show plots of various things
function display(W,f,sigma,ksp,iter,tol,normA)

% plot singular values
subplot(1,4,1); plot(W/W(1)); title(sprintf('rank %i/%i',nnz(f),numel(f))); 
hold on; plot(f,'--'); hold off; xlim([0 numel(f)+1]);
line(xlim,gather([1 1]*sigma/W(1)),'linestyle',':','color','black');
legend({'singular vals.','sing. val. filter','noise floor'});

% show current kspace
subplot(1,4,2); imagesc(log(sum(abs(ksp),3)));
xlabel('dim 2'); ylabel('dim 1'); title('kspace');

% show current image
subplot(1,4,3); imagesc((sum(abs(ifft2(ksp)),3)));
xlabel('dim 2'); ylabel('dim 1'); title(sprintf('iter %i',iter));

% plot change in norm and tol
subplot(1,4,4); h = plotyy(1:iter,tol,1:iter,normA,'semilogy','semilogy');
axis(h,'tight'); xlim(h,[0 iter+1]); xlabel('iters'); title('metrics');
legend({'||Δk||/||k||','||A||_* norm'}); drawnow;
