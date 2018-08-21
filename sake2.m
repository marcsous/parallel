function ksp = sake2(data,mask,varargin)
% ksp = sake2(data,mask,varargin)
%
% 2D MRI reconstruction based on matrix completion.
%
% Inputs:
%  -data [nx ny nc]: 2d kspace data array from nc coils
%  -mask [nx ny]: 2d sampling mask (or 1d vector, legacy) 
%  -varargin: pairs of options/values (e.g. 'radial',1)
%
% Outputs:
%  -ksp [nx ny nc]: 2d kspace data array from nc coils
%
% References:
%  -Haldar JP et al. LORAKS. IEEE Trans Med Imag 2014;33:668
%  -Shin PJ et al. SAKE. Magn Resonance Medicine 2014;72:959

%% setup

% default options
opts.width = 5; % kernel width
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
if ndims(data)<2 || ndims(data)>3
    error('Argument ''data'' must be a 3d array.')
end
[nx ny nc] = size(data);

if ~exist('mask','var') || isempty(mask)
    mask = any(data,3); % 2d mask [nx ny]
    warning('Argument ''mask'' not supplied - guessing.')
elseif isvector(mask)
    if ~isa(mask,'logical')
        index = unique(mask); % allow indices
        mask = false(1,ny); mask(index) = 1;
    end
    mask = repmat(reshape(mask,1,ny),nx,1);
end
mask = reshape(mask,nx,ny); % catch size mismatch

% convolution kernel indicies
[x y] = ndgrid(-ceil(opts.width/2):ceil(opts.width/2));
if opts.radial
    k = sqrt(x.^2+y.^2)<=opts.width/2;
else
    k = abs(x)<=opts.width/2 & abs(y)<=opts.width/2;
end
nk = nnz(k);
opts.kernel.x = x(k);
opts.kernel.y = y(k);

% dimensions of the dataset
opts.dims = [nx ny nc nk];
if opts.loraks; opts.dims = [opts.dims 2]; end

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

% memory required for data matrix
k = gather(data(1)) * 0; % single or double precision
bytes = 2 * prod(opts.dims) * getfield(whos('k'),'bytes');

% density of data matrix
density = nnz(mask) / numel(mask);

% display
disp(rmfield(opts,{'flip','kernel'}));
fprintf('Sampling density = %f\n',density);
fprintf('Matrix = %ix%i (%.1f Gb)\n',nx*ny,prod(opts.dims(3:end)),bytes/1e9);

%% see if gpu is possible

try
    gpu = gpuDevice;
    if gpu.AvailableMemory < 4*bytes; error('GPU memory too small'); end
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

ksp = zeros(nx,ny,nc,'like',data);

for iter = 1:opts.maxit

    % data consistency
    ksp = bsxfun(@times,data,mask) + bsxfun(@times,ksp,~mask);

    % calibration matrix
    A = make_data_matrix(ksp,opts);

    % row space and singular values
    if isempty(opts.cal)
        [V S] = svd(A'*A);
        S = sqrt(diag(S));
    else
        S = svd(A'*A);
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
            opts.noise = noise_floor / sqrt(2*density*nx*ny);
            disp(['Estimated noise std = ' num2str(opts.noise)]);
        end
    else
        noise_floor = opts.noise * sqrt(2*density*nx*ny);
    end

    % rank reduction by minimium variance filtering
    f = max(0,1-noise_floor.^2./S.^2);
    F = V * diag(f) * V';
    A = A * F;

    % hankel structure (average along anti-diagonals)   
    ksp = undo_data_matrix(A,opts);
    
    % convergence
    normA(iter) = sum(S);
    if iter==1; old = NaN; end
    tol(iter) = norm(ksp(:)-old(:))/norm(ksp(:));
    old = ksp;
    converged = tol(iter) < opts.tol;
    
    % display every few iterations
    if mod(iter,10)==1 || converged

        % plot singular values
        subplot(1,4,1); plot(S/S(1));
        hold on; plot(max(f,min(ylim)),'--'); hold off
        line(xlim,gather([1 1]*noise_floor/S(1)),'linestyle',':','color','black');
        legend({'singular vals.','min. var. filter','noise floor'});
        title(sprintf('rank %i/%i',nnz(f),numel(f))); xlim([0 numel(S)+1]);
        
        % show current kspace
        subplot(1,4,2); temp = log(sum(abs(ksp),3));
        imagesc(temp); xlabel('dim 2'); ylabel('dim 1'); title('kspace');
        
        % show current image
        subplot(1,4,3); imagesc(sum(abs(ifft2(ksp)),3));
        xlabel('dim 2'); ylabel('dim 1'); title(sprintf('iter %i',iter));
        
        % plot change in norm and tol
        subplot(1,4,4); warning('off','MATLAB:Axes:NegativeLimitsInLogAxis');
        [h,~,~] = plotyy(1:iter,max(tol,opts.tol),1:iter,normA);
        warning('off','MATLAB:Axes:NegativeLimitsInLogAxis');
        set(h(1),'YScale','log'); set(h(2),'YScale','log');
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
    B = conj(A(opts.flip.x,opts.flip.y,:,:));
    A = cat(4,A,B);
end

A = reshape(A,nx*ny,[]);

%% undo data matrix
function data = undo_data_matrix(A,opts)

nx = opts.dims(1);
ny = opts.dims(2);
nc = opts.dims(3);
nk = opts.dims(4);

A = reshape(A,nx,ny,nc,[]);

for k = 1:nk
    x = opts.kernel.x(k);
    y = opts.kernel.y(k);
    A(:,:,:,k) = circshift(A(:,:,:,k),-[x y]);

    if opts.loraks
        B = conj(A(opts.flip.x,opts.flip.y,:,k+nk));
        A(:,:,:,k+nk) = circshift(B,-[x y]);
    end
end

data = mean(A,4);